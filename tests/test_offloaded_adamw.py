"""Reference AdamW and exact resume controls; no language-quality claim."""
import copy
import unittest

import torch

from ilm.visual_lm.offloaded_adamw import OffloadedAdamW


class OffloadedAdamWTest(unittest.TestCase):
    def test_matches_torch_adamw_with_partial_chunks_and_unused_parameter(self):
        torch.manual_seed(59)
        ours=[torch.nn.Parameter(torch.randn(7,11)),torch.nn.Parameter(torch.randn(13))]
        reference=[torch.nn.Parameter(p.detach().clone()) for p in ours]
        a=OffloadedAdamW(ours,lr=.003,betas=(.8,.97),eps=2e-7,weight_decay=.04,chunk_elements=17)
        b=torch.optim.AdamW(reference,lr=.003,betas=(.8,.97),eps=2e-7,weight_decay=.04,foreach=False,fused=False)
        for step in range(5):
            for i,(p,q) in enumerate(zip(ours,reference)):
                g=None if step==2 and i==1 else torch.randn_like(p)
                p.grad=g;q.grad=None if g is None else g.clone()
            a.step();b.step()
            for p,q in zip(ours,reference):
                torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)
                for name in ('exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(a.state[p][name].reshape_as(p),b.state[q][name],rtol=1e-6,atol=1e-7)
        self.assertEqual(a.moment_bytes(),sum(p.numel()*8 for p in ours))

    def test_resume_preserves_host_state_and_next_update(self):
        torch.manual_seed(12);p=torch.nn.Parameter(torch.randn(31))
        a=OffloadedAdamW([p],chunk_elements=9);p.grad=torch.randn_like(p);a.step()
        q=torch.nn.Parameter(p.detach().clone());b=OffloadedAdamW([q],chunk_elements=2)
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        grad=torch.randn_like(p);p.grad=grad;q.grad=grad.clone();a.step();b.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertEqual(b.state[q]['exp_avg'].device.type,'cpu')
        self.assertEqual(b.state[q]['step'],2)

    def test_invalid_later_gradient_is_rejected_before_any_mutation(self):
        p=torch.nn.Parameter(torch.ones(4));q=torch.nn.Parameter(torch.ones(3))
        optimizer=OffloadedAdamW([p,q]);p.grad=torch.ones_like(p);q.grad=torch.full_like(q,float('nan'))
        with self.assertRaises(ValueError):optimizer.step()
        self.assertTrue(torch.equal(p,torch.ones_like(p)))
        self.assertEqual(len(optimizer.state),0)


if __name__=='__main__':unittest.main()

import copy
import unittest

import torch
from torch import nn

from ilm.visual_lm.nonlinear_visual_feedback import NonlinearVisualFeedback, relative_feedback_error


class NonlinearFeedbackTests(unittest.TestCase):
    def test_zero_branch_preserves_values_gradients_and_rng(self):
        torch.manual_seed(81)
        base = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 16))
        old = copy.deepcopy(base)
        before = torch.get_rng_state().clone()
        extended = NonlinearVisualFeedback(base, hidden_width=32)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x = torch.randn(17,3,8,requires_grad=True)
        a,b = old(x),extended(x)
        self.assertTrue(torch.equal(a,b))
        a.square().mean().backward(retain_graph=True)
        reference = [p.grad.clone() for p in old.parameters()]
        b.square().mean().backward()
        for left,right in zip(reference,extended.original.parameters()):
            torch.testing.assert_close(left,right.grad,atol=0,rtol=0)
        self.assertGreater(float(extended.project.weight.grad.norm()),0)
        self.assertEqual(float(extended.expand.weight.grad.norm()),0)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            self.assertTrue(torch.equal(old(x.detach()),extended(x.detach())))

    def test_real_optimizer_opens_branch_and_reset_is_exact(self):
        torch.manual_seed(82)
        extended = NonlinearVisualFeedback(nn.Sequential(nn.LayerNorm(8),nn.Linear(8,16)),hidden_width=32)
        initial = copy.deepcopy(extended.state_dict())
        opt = torch.optim.AdamW(extended.parameters(),lr=.001)
        x,y = torch.randn(64,8),torch.randn(64,16)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            relative_feedback_error(extended(x),y).mean().backward()
            if step==1:self.assertGreater(float(extended.expand.weight.grad.norm()),0)
            opt.step()
        self.assertTrue(any(not torch.equal(value,initial[name]) for name,value in extended.state_dict().items()))
        extended.load_state_dict(initial,strict=True)
        self.assertTrue(all(torch.equal(value,initial[name]) for name,value in extended.state_dict().items()))

    def test_nonlinear_coordinates_escape_fixed_affine_family(self):
        torch.manual_seed(83)
        base = nn.Sequential(nn.LayerNorm(8),nn.Linear(8,16))
        extended = NonlinearVisualFeedback(base,hidden_width=32)
        with torch.no_grad():extended.project.weight.normal_(0,.2)
        x = torch.randn(512,8)
        a,b = base(x).double(),extended(x).double()
        # The finite fixture witnesses removal of the structural affine bound;
        # it is not a rank claim about any trained production reader.
        rank_a = int(torch.linalg.matrix_rank(a-a.mean(0),atol=1e-4,rtol=0))
        rank_b = int(torch.linalg.matrix_rank(b-b.mean(0),atol=1e-4,rtol=0))
        self.assertLessEqual(rank_a,7)
        self.assertEqual(rank_b,16)

    def test_weighted_objective_keeps_ambiguous_targets(self):
        x = torch.tensor([[1.,2.],[1.,2.]],requires_grad=True)
        target = torch.tensor([[3.,4.],[-3.,4.]])
        weights = torch.tensor([.3,.7])
        got = (weights*relative_feedback_error(x,target)).sum()
        expected = sum(weights[i]*(x[i]-target[i]).square().sum()/target[i].square().sum() for i in range(2))
        torch.testing.assert_close(got,expected)
        a,=torch.autograd.grad(got,x,retain_graph=True)
        b,=torch.autograd.grad(expected,x)
        torch.testing.assert_close(a,b)
        with self.assertRaises(ValueError):relative_feedback_error(x[:0],target[:0])


if __name__=='__main__':unittest.main()

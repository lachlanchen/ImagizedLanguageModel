import copy
import unittest

import torch
from torch import nn

from ilm.visual_lm.profiled_visual_feedback import (
    FixedRidgeMoments,Float32VisualFeedback,feedback_features,install_native_readout,native_partial_loss,
)


class ProfiledFeedbackTests(unittest.TestCase):
    def test_envelope_gradient_matches_differentiating_solve(self):
        torch.manual_seed(213)
        x=torch.randn(23,4,dtype=torch.float64)
        a=torch.randn(4,7,dtype=torch.float64,requires_grad=True)
        y=torch.randn(23,3,dtype=torch.float64);c=torch.rand(23,dtype=torch.float64)+.1;ridge=.03
        phi=torch.cat((torch.ones(23,1,dtype=torch.float64),x,torch.nn.functional.gelu(x@a)),dim=-1)
        gram=phi.T@(c[:,None]*phi);rhs=phi.T@(c[:,None]*y)
        k=torch.linalg.solve(gram+ridge*torch.eye(len(gram),dtype=torch.float64),rhs)
        full=(c[:,None]*(phi@k-y).square()).sum()+ridge*k.square().sum()
        expected,=torch.autograd.grad(full,a,retain_graph=True)
        partial=native_partial_loss(phi,k,y,c)
        actual,=torch.autograd.grad(partial,a)
        torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-10)

    def test_streamed_feature_gradient_and_fixed_lambda(self):
        torch.manual_seed(214)
        head=Float32VisualFeedback(nn.Sequential(nn.LayerNorm(4),nn.Linear(4,3)),hidden_width=8)
        head.requires_grad_(False);head.expand.requires_grad_(True)
        x=torch.randn(17,4);y=torch.randn(17,3);c=torch.rand(17,dtype=torch.float64)+.01
        phi=feedback_features(head,x)
        moments=FixedRidgeMoments(phi.shape[1],3,'cpu');moments.add(phi.detach(),y,c)
        k,r=moments.solve_fixed(.0123)
        self.assertEqual(r['ridge'],.0123)
        native_partial_loss(phi,k,y,c).backward()
        expected=[p.grad.clone() for p in head.expand.parameters()]
        head.zero_grad(set_to_none=True)
        for first in range(0,17,5):
            native_partial_loss(feedback_features(head,x[first:first+5]),k,y[first:first+5],c[first:first+5]).backward()
        for a,b in zip(expected,head.expand.parameters()):
            torch.testing.assert_close(a,b.grad,atol=2e-6,rtol=2e-6)
        self.assertTrue(all(p.grad is None for p in head.original.parameters()))
        self.assertIsNone(head.project.weight.grad)

    def test_actual_fp32_readout_matches_native_matrix(self):
        torch.manual_seed(215)
        head=Float32VisualFeedback(nn.Sequential(nn.LayerNorm(4),nn.Linear(4,3)),hidden_width=8)
        x=torch.randn(19,4).bfloat16();k=torch.randn(13,3,dtype=torch.float64)
        scale=torch.rand(3)+.2;mean=torch.randn(3)
        before={n:v.clone() for n,v in head.state_dict().items() if n.startswith(('expand.','original.0.'))}
        install_native_readout(head,k,scale)
        with torch.autocast('cpu',dtype=torch.bfloat16):actual=head(x)*scale+mean
        self.assertEqual(actual.dtype,torch.float32)
        expected=feedback_features(head,x).double()@k+mean.double()
        torch.testing.assert_close(actual.double(),expected,atol=3e-6,rtol=3e-6)
        self.assertTrue(all(torch.equal(v,head.state_dict()[n]) for n,v in before.items()))

    def test_trial_state_reset_and_invalid_ridge(self):
        head=Float32VisualFeedback(nn.Sequential(nn.LayerNorm(4),nn.Linear(4,3)),hidden_width=8)
        saved=copy.deepcopy(head.state_dict())
        with torch.no_grad():head.expand.weight.add_(.1)
        head.load_state_dict(saved,strict=True)
        self.assertTrue(all(torch.equal(v,saved[n]) for n,v in head.state_dict().items()))
        moments=FixedRidgeMoments(2,1,'cpu')
        with self.assertRaises(ValueError):moments.solve_fixed(.1)
        moments.add(torch.ones(3,2),torch.ones(3,1),torch.ones(3))
        for ridge in [0.,-1.,float('nan')]:
            with self.assertRaises(ValueError):moments.solve_fixed(ridge)


if __name__=='__main__':unittest.main()

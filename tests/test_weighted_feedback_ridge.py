import unittest
import torch

from ilm.visual_lm.weighted_feedback_ridge import WeightedFeedbackMoments


class WeightedRidgeTests(unittest.TestCase):
    def test_streamed_solution_matches_augmented_least_squares(self):
        torch.manual_seed(193)
        for singular in (False,True):
            x=torch.randn(31,9,dtype=torch.float64)
            if singular:x[:,8]=x[:,0]+x[:,1]
            y=torch.randn(31,5,dtype=torch.float64);c=torch.rand(31,dtype=torch.float64)+.01
            originals=[a.clone() for a in (x,y,c)]
            moments=WeightedFeedbackMoments(9,5,'cpu')
            for first in range(0,31,7):moments.add(x[first:first+7],y[first:first+7],c[first:first+7])
            got,receipt=moments.solve()
            a=torch.cat((x*c.sqrt()[:,None],torch.eye(9,dtype=torch.float64)*receipt['ridge']**.5))
            b=torch.cat((y*c.sqrt()[:,None],torch.zeros(9,5,dtype=torch.float64)))
            expected=torch.linalg.lstsq(a,b).solution
            torch.testing.assert_close(got,expected,atol=1e-9,rtol=1e-9)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip((x,y,c),originals)))

    def test_native_affine_scaling_and_bias_are_inverted(self):
        torch.manual_seed(194)
        x=torch.randn(23,6,dtype=torch.float64);phi=torch.cat((torch.ones(23,1),x),dim=1)
        y=torch.randn(23,4,dtype=torch.float64);mean=torch.randn(4,dtype=torch.float64);scale=torch.rand(4,dtype=torch.float64)+.1
        mass=torch.rand(23,dtype=torch.float64);mass/=mass.sum();c=mass/y.square().sum(-1)
        moments=WeightedFeedbackMoments(7,4,'cpu');moments.add(phi,y-mean,c);solution,_=moments.solve()
        normalized=(phi@(solution/scale))
        physical=normalized*scale+mean
        torch.testing.assert_close(physical,phi@solution+mean,atol=1e-12,rtol=1e-12)
        direct=(mass*(physical-y).square().sum(-1)/y.square().sum(-1)).sum()
        reduced=(c*(phi@solution-(y-mean)).square().sum(-1)).sum()
        torch.testing.assert_close(direct,reduced,atol=1e-12,rtol=1e-12)

    def test_invalid_or_empty_support_rejected(self):
        m=WeightedFeedbackMoments(2,1,'cpu')
        with self.assertRaises(ValueError):m.solve()
        with self.assertRaises(ValueError):m.add(torch.ones(1,2),torch.ones(1,1),torch.tensor([0.]))
        m.add(torch.ones(1,2),torch.ones(1,1),torch.ones(1))
        with self.assertRaises(ValueError):m.solve(0)


if __name__=='__main__':unittest.main()

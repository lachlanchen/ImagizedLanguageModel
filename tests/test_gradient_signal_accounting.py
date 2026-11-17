"""Numerical checks of diagnostic arithmetic, not visual-language evidence."""
import unittest

import torch

from ilm.visual_lm.gradient_signal_accounting import PairMoments, adamw_displacement


class GradientAccountingTest(unittest.TestCase):
    def test_retained_moments_match_torch_adamw_without_mutation(self):
        gen = torch.Generator().manual_seed(202611171)
        for step in (0, 8172):
            weight = torch.randn(1031, generator=gen)
            gradient = torch.randn(1031, generator=gen) * .03
            mean = torch.randn(1031, generator=gen) * .02
            square = torch.rand(1031, generator=gen) * .1
            originals = [v.clone() for v in (weight, gradient, mean, square)]
            settings = dict(lr=2e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
            actual = adamw_displacement(weight, gradient, mean, square, step=step, **settings)
            p = torch.nn.Parameter(weight.clone())
            p.grad = gradient.clone()
            optimizer = torch.optim.AdamW([p], foreach=False, fused=False, **settings)
            optimizer.state[p] = dict(step=torch.tensor(float(step)),
                                      exp_avg=mean.clone(), exp_avg_sq=square.clone())
            optimizer.step()
            torch.testing.assert_close(actual, p.detach() - weight, rtol=0, atol=1e-7)
            for value, original in zip((weight, gradient, mean, square), originals):
                self.assertTrue(torch.equal(value, original))

    def test_unused_gradient_skips_momentum_and_weight_decay(self):
        w = torch.tensor([1., -2.])
        result = adamw_displacement(w, None, w, w.square(), step=8172,
            lr=2e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.assertTrue(torch.equal(result, torch.zeros_like(w)))

    def test_geometry_and_zero_gradient(self):
        moments = PairMoments()
        moments.add(torch.tensor([1., 2.]), torch.tensor([-1., -2.]))
        self.assertAlmostEqual(moments.result()['cosine'], -1.)
        self.assertEqual(moments.result()['inner_product'], -5.)
        empty = PairMoments()
        empty.add(torch.zeros(3), torch.ones(3))
        self.assertIsNone(empty.result()['cosine'])

    def test_aligned_gradient_can_oppose_retained_momentum(self):
        flow = torch.ones(4)
        direction = adamw_displacement(torch.ones(4), flow, -torch.ones(4), torch.ones(4),
            step=8172, lr=2e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.assertGreater(float(flow @ direction), 0.)


if __name__ == '__main__':
    unittest.main()

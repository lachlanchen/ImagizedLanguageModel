"""Numerical tests of the streaming probability law, not language benchmarks."""
import unittest

import torch
from torch.nn import functional as F

from ilm.visual_lm.predictive_state_distillation import complete_readout_kl, PredictiveStateObjective


class ReadoutKLTests(unittest.TestCase):
    def test_complete_law_and_first_derivative_match_dense(self):
        for rows in (1, 17):
            for block in (1, 6, 24):
                for temperature in (.5, 1., 3.):
                    with self.subTest(rows=rows, block=block, temperature=temperature):
                        torch.manual_seed(813)
                        x = torch.randn(7, 11, requires_grad=True)
                        y = torch.randn(7, 11)
                        w = torch.randn(rows, 11) * .3
                        originals = [v.detach().clone() for v in (x, y, w)]
                        pred = F.log_softmax(x @ w.T / temperature, dim=-1)
                        gold = F.log_softmax(y @ w.T / temperature, dim=-1)
                        dense = (gold.exp() * (gold - pred)).sum(-1).mean() * temperature ** 2
                        expected, = torch.autograd.grad(dense, x)
                        streamed = complete_readout_kl(x, y, w, temperature=temperature, block_rows=block)
                        actual, = torch.autograd.grad(streamed * 1.7, x)
                        torch.testing.assert_close(streamed, dense, atol=3e-6, rtol=3e-5)
                        torch.testing.assert_close(actual, expected * 1.7, atol=2e-6, rtol=3e-5)
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip((x, y, w), originals)))
                        self.assertIsNone(y.grad)
                        self.assertIsNone(w.grad)

    def test_identical_predictions_and_chunk_weighted_mean(self):
        torch.manual_seed(71)
        x = torch.randn(5, 9, requires_grad=True)
        w = torch.randn(13, 9)
        zero = complete_readout_kl(x, x.detach(), w, block_rows=4)
        grad, = torch.autograd.grad(zero, x)
        torch.testing.assert_close(zero, torch.tensor(0.), atol=1e-7, rtol=0)
        torch.testing.assert_close(grad, torch.zeros_like(x), atol=1e-7, rtol=0)
        y = torch.randn_like(x)
        whole = complete_readout_kl(x, y, w, block_rows=4)
        parts = (2 * complete_readout_kl(x[:2], y[:2], w, block_rows=4)
                 + 3 * complete_readout_kl(x[2:], y[2:], w, block_rows=4)) / 5
        torch.testing.assert_close(whole, parts, atol=1e-6, rtol=1e-6)

    def test_invalid_support_is_rejected(self):
        x, w = torch.ones(2, 3), torch.ones(5, 3)
        for kwargs in ({'temperature': 0}, {'temperature': float('nan')}, {'block_rows': 0}):
            with self.assertRaises(ValueError):
                complete_readout_kl(x, x, w, **kwargs)
        with self.assertRaises(ValueError):
            complete_readout_kl(x[:0], x[:0], w)
        with self.assertRaises(ValueError):
            complete_readout_kl(x, x, w.requires_grad_())
        with self.assertRaises(ValueError):
            complete_readout_kl(x * float('nan'), x, w.detach())

    def test_training_readout_is_not_a_saved_parameter(self):
        objective = PredictiveStateObjective(torch.randn(13, 7), context_width=5, teacher_width=7)
        self.assertEqual(set(objective.state_dict()), {'teacher_projection.weight'})
        self.assertEqual(sum(p.numel() for p in objective.parameters()), 35)


if __name__ == '__main__':
    unittest.main()

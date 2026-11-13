import numpy as np
import torch

from scripts.analyze_visual_continuation_trajectories import constant_optimum, quantile


def test_sequence_balanced_optimum_has_zero_independent_autograd_derivative():
    sequences = [np.array([[1, 0, 1, 0], [0, 1, 0, 0]]),
                 np.array([[1, 1, 0, 1]]), np.array([[0, 0, 1, 1]] * 3)]
    q, _, _ = constant_optimum(sequences)
    logits = torch.tensor(np.log(q / (1 - q)), dtype=torch.float64, requires_grad=True)
    losses = []
    for values in sequences:
        x = torch.tensor(values, dtype=torch.float64)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(logits.expand_as(x), x, reduction='none')
        losses.append(.5 * ((ce * x).sum() / x.sum().clamp_min(1)
                            + (ce * (1 - x)).sum() / (1 - x).sum().clamp_min(1)))
    torch.stack(losses).mean().backward()
    assert logits.grad.abs().max() < 1e-12
    assert not np.allclose(q, np.concatenate(sequences).mean(0))


def test_blank_and_fully_black_sequences_remain_in_population():
    q, a, b = constant_optimum([np.zeros((2, 4)), np.ones((1, 4))])
    np.testing.assert_allclose(q, .5)
    np.testing.assert_allclose(a, .125)
    np.testing.assert_allclose(b, .125)


def test_empirical_density_quantile_uses_inclusive_histogram_count():
    counts = np.array([1, 2, 1, 0])
    assert quantile(counts, .5) == 1
    assert quantile(counts, .99) == 2

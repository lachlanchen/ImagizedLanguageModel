import math
import torch

from ilm.visual_lm.context_contrast_writer import candidate_densities, select_group
from ilm.visual_lm.continuous_autoregressive_language import mixture_log_prob


def test_all_candidate_scores_match_the_scalar_mixture():
    torch.manual_seed(3)
    means = torch.randn(3, 4, 8)
    params = torch.randn(3, 4), means, torch.rand_like(means) + .2
    probabilities = candidate_densities(means, params)
    expected = torch.stack([mixture_log_prob(means[:, i], *params) for i in range(4)], 1)
    torch.testing.assert_close(probabilities, expected)
    selected, control, _ = select_group(params, params)
    assert torch.equal(selected, control)


def test_plausible_contrast_moves_towards_a_context_specific_mode():
    centers = torch.tensor([[[-5.], [0.], [8.]]])
    full = torch.tensor([[1., 0., -8.]]), centers, torch.ones_like(centers)
    short = torch.tensor([[3., -3., -20.]]), centers, torch.ones_like(centers)
    selected, control, eligible = select_group(full, short)
    assert (selected.item(), control.item(), eligible.item()) == (1, 0, 2)
    assert math.isfinite(candidate_densities(centers, short).min().item())

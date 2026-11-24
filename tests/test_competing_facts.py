"""Duplicated ink counts once as a surface class, across every valid occurrence."""

import torch
from scripts.train_competing_facts import alternative_mass


def test_interval_marginal_and_gradient():
    weights = torch.tensor(
        [[[0.12, 0.18], [0.20, 0.10], [0.30, 0.10]]], dtype=torch.float64
    )
    scores = weights.log().requires_grad_()
    p = alternative_mass(scores, torch.tensor([[0, 2, -1]]), torch.tensor([1]))
    torch.testing.assert_close(p.exp(), torch.tensor([0.42], dtype=torch.float64))
    p.sum().backward()
    expected = torch.zeros_like(scores)
    expected[0, 0, 0] = 0.12 / 0.42
    expected[0, 2, 0] = 0.30 / 0.42
    torch.testing.assert_close(scores.grad, expected)


def test_rival_uses_same_question_scores_and_other_surface():
    x = torch.tensor([[[0.7], [0.3]], [[0.2], [0.8]]]).log()
    at = torch.tensor([[0], [1]])
    lengths = torch.tensor([1, 1])
    positive = alternative_mass(x, at, lengths)
    negative = alternative_mass(x, at.flip(0), lengths)
    torch.testing.assert_close(positive.exp(), torch.tensor([0.7, 0.8]))
    torch.testing.assert_close(negative.exp(), torch.tensor([0.3, 0.2]))

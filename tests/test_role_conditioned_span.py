"""Check the full-span objective against explicit valid-pair enumeration."""

import torch
from ilm.visual_lm.bound_fact_reader import best_bounds
from ilm.visual_lm.role_conditioned_span import ParticipantHead, span_scores


def test_exhaustive_span_probabilities_gradients_and_decode():
    rng = torch.Generator().manual_seed(13)
    logits = torch.randn(
        3, 55, 2, dtype=torch.float64, generator=rng, requires_grad=True
    )
    lengths = [1, 8, 55]
    valid = torch.arange(55)[None, :] < torch.tensor(lengths)[:, None]
    scores = span_scores(logits, valid)
    partitions = scores.flatten(1).logsumexp(dim=1)
    bounds = best_bounds(logits.masked_fill(~valid[:, :, None], -1e4))
    brute = []
    for row, n in enumerate(lengths):
        pairs = [(i, j) for i in range(n) for j in range(i, min(i + 48, n))]
        joint = torch.stack([logits[row, i, 0] + logits[row, j, 1] for i, j in pairs])
        brute.append(joint.logsumexp(0))
        i, j = pairs[joint.argmax()]
        assert bounds[row].tolist() == [i, j + 1]
        assert int(torch.isfinite(scores[row]).sum()) == len(pairs)
    brute = torch.stack(brute)
    torch.testing.assert_close(brute, partitions)
    exact_grad = torch.autograd.grad(brute.sum(), logits)[0]
    actual_grad = torch.autograd.grad(partitions.sum(), logits)[0]
    torch.testing.assert_close(exact_grad, actual_grad)


def test_zero_residual_and_padding():
    model = ParticipantHead(12)
    page = torch.randn(2, 4, 773)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    role = torch.tensor([0, 1])
    prior = torch.randn(2, 4, 2)
    scores = model(page, valid, role, prior)
    torch.testing.assert_close(scores[valid], prior[valid])
    assert torch.all(scores[~valid] == -1e4)

import torch
from ilm.visual_lm.fact_channel import OpticalFactReader
from ilm.visual_lm.fact_ranking import SpanRanker, candidate_loss, candidates


def test_frozen_candidates_and_exact_ties():
    torch.manual_seed(31)
    net = OpticalFactReader().eval()
    features = torch.randn(17, 256)
    query = torch.tensor([[2, 3], [4, 0]])
    page = torch.tensor([[3, 4, 5, 0, 3, 4], [8, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        net.boundaries.weight.zero_()
        net.boundaries.bias.zero_()
        net.length_bias.zero_()
    emb, base, start, size = candidates(net, features, query, page)
    assert base.isfinite().sum(1).tolist() == [9, 1]
    assert list(zip(start[0, :9].tolist(), size[0, :9].tolist())) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (2, 1),
        (4, 1),
        (4, 2),
        (5, 1),
    ]
    for mode in ("endpoints", "whole"):
        model = SpanRanker(mode)
        torch.testing.assert_close(model(emb, base, size), base)
    torch.testing.assert_close(emb[0, 0, 0], emb[0, 0, 2])


def test_missing_targets_give_zero_gradient_and_occurrences_are_marginalized():
    z = torch.tensor([[1.0, -1.0, 2.0], [0.0, 2.0, 1.0]], requires_grad=True)
    y = torch.tensor([[True, False, True], [False, False, False]])
    loss = candidate_loss(z, y)
    ref = z[0].logsumexp(0) - z[0, [0, 2]].logsumexp(0)
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(
        torch.autograd.grad(loss, z)[0], torch.autograd.grad(ref, z)[0]
    )
    missing = candidate_loss(z, torch.zeros_like(y))
    assert missing.item() == 0
    assert torch.autograd.grad(missing, z)[0].count_nonzero() == 0

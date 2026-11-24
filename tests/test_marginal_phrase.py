from itertools import product
import torch
from ilm.visual_lm.marginal_phrase import SegmentationLikelihood


def test_partition_and_gradient_against_enumerated_log_paths():
    torch.manual_seed(81)
    lengths = torch.tensor([1, 3, 7, 9])
    x = torch.randn(4, 9, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.rand_like(x) > 0.4
    mask[:, :, 0] = True
    scores = x.masked_fill(~mask, -torch.inf)
    measured = SegmentationLikelihood.apply(scores, lengths)

    def enumerate_paths(b, t, end):
        if t == end:
            return [x.new_zeros(())]
        return [
            scores[b, t, k] + tail
            for k in range(5)
            if t + 2**k <= end and mask[b, t, k]
            for tail in enumerate_paths(b, t + 2**k, end)
        ]

    expected = torch.stack(
        [
            torch.stack(enumerate_paths(b, 0, n)).logsumexp(0)
            for b, n in enumerate(lengths.tolist())
        ]
    )
    weights = torch.tensor([1.0, 2.0, 0.7, 0.3])
    torch.testing.assert_close(measured, expected)
    ga = torch.autograd.grad((measured * weights).sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad((expected * weights).sum(), x)[0]
    torch.testing.assert_close(ga, gb)

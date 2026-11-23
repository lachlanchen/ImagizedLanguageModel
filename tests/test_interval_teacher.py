import itertools
import math
import pytest
import torch
from ilm.visual_lm.interval_teacher import interval_targets, path_loss


@pytest.mark.parametrize("length", [1, 3, 4])
def test_prefix_partition_matches_explicit_intervals(length):
    page = [4, 8, 4, 8, 4, 3]
    target = page[:length]
    source = torch.tensor([page + [-1, -1]])
    valid = torch.arange(8)[None, :] < len(page)
    targets = torch.tensor([target + [0]])
    t = torch.tensor(
        [[[0.2 * i, (-0.17 * i if i % 2 else 0.13 * i)] for i in range(8)]]
    )
    copies, stop = interval_targets(
        source, valid, targets, torch.tensor([length]), t, max_span=4
    )
    for k in range(length + 1):
        ends = 0.0
        more = [0.0] * 8
        for i, j in itertools.combinations_with_replacement(range(len(page)), 2):
            if j - i >= 4 or len(page[i : j + 1]) < k or page[i : i + k] != target[:k]:
                continue
            w = math.exp(float(t[0, i, 0] + t[0, j, 1]))
            if j - i + 1 == k:
                ends += w
            else:
                more[i] += w
        total = ends + sum(more)
        assert copies[0, k].tolist() == pytest.approx(
            [x / total for x in more], abs=2e-6
        )
        assert stop[0, k].item() == pytest.approx(ends / total, abs=2e-6)


def test_duplicates_padding_and_gradients():
    source = torch.tensor([[3, 4, 3, 4], [9, 8, 0, 0]])
    valid = source != 0
    y = torch.tensor([[3, 4, 0], [9, 0, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).bool()
    c, s = interval_targets(source, valid, y, mask.sum(1) - 1)
    assert c[0, 0].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert s.tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    pointer = torch.randn(2, 3, 4, requires_grad=True)
    gate = torch.randn(2, 3, requires_grad=True)
    stop = torch.randn(2, 3, requires_grad=True)
    loss = path_loss(
        pointer.masked_fill(~valid[:, None, :], -1e30),
        gate,
        stop,
        source,
        valid,
        y,
        mask,
        torch.randn(2, 4, 2),
        0.5,
    )
    loss.backward()
    assert all(
        torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for x in (pointer, gate, stop)
    )


def test_teacher_cross_entropy_scores_the_actual_copy_and_stop_routes():
    source = torch.tensor([[1, 2]])
    valid = source != 0
    answer = torch.tensor([[1, 0]])
    pointer = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    gate = torch.zeros(1, 2)
    stop_logp = torch.full((1, 2), math.log(0.25))
    mask = torch.ones((1, 2), dtype=torch.bool)
    # Equiprobable teacher spans [1], [2], [1,2]. After the prefix [1],
    # the copy and terminal events have probability one half each.
    teacher = torch.zeros(1, 2, 2, requires_grad=True)
    loss = path_loss(
        pointer, gate, stop_logp, source, valid, answer, mask, teacher, 1.0
    )
    expected = (-math.log(0.25) - 0.5 * math.log(0.25) - 0.5 * math.log(0.125)) / 2
    assert float(loss) == pytest.approx(expected)
    assert not loss.requires_grad

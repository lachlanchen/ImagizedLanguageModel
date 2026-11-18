import numpy as np
import torch

from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from scripts.train_anchor_gap_writer import targets


def test_supervision_covers_initial_internal_and_final_insertions():
    keep, left, right, offsets, close = targets(4, 6, [(0, 1), (3, 4)])
    assert keep.tolist() == [True, False, False, True]
    assert left.tolist() == [-1, -1, 0, 0, 0, 3, 3]
    assert right.tolist() == [0, 0, 3, 3, 3, 4, 4]
    assert offsets.tolist() == [0, 1, 0, 1, 2, 0, 1]
    assert close.tolist() == [False, True, False, False, True, False, True]


def test_equal_images_and_deletions_preserve_the_selected_occurrences():
    keep, _, _, offsets, close = targets(5, 2, [(1, 0), (3, 1)])
    assert keep.tolist() == [False, True, False, True, False]
    assert close.all() and not offsets.any()
    torch.manual_seed(1)
    torch.set_num_threads(1)
    pictures = torch.rand(2, 5, 1024) > 0.5
    pictures[0, 2] = pictures[0, 1]
    model = AnchorGapWriter(torch.arange(48), pictures[0, :2])
    with torch.no_grad():
        model.close_gap.weight.zero_()
        model.close_gap.bias.fill_(10.0)
    plan = torch.tensor(np.stack([keep, keep]))
    lengths = torch.tensor([5, 2])
    output, size, stopped, selected = model.write(
        pictures, lengths, policy="supplied", supplied=plan, limit=2
    )
    assert size.tolist() == [2, 1] and stopped.all()
    assert torch.equal(output[0, :2], pictures[0, [1, 3]])
    assert torch.equal(output[1, 0], pictures[1, 1]) and selected.sum() == 3
    _, size, stopped, _ = model.write(pictures, lengths, policy="all", limit=2)
    assert size.tolist() == [2, 2] and stopped.tolist() == [False, True]


def test_finite_gaps_backoff_and_unseen_ink(tmp_path):
    from ilm.visual_lm.finite_visual_gaps import FiniteVisualGaps

    cells = np.zeros((4, 128), np.uint8)
    cells[:, 0] = [1, 2, 3, 4]
    rules = {
        (0, (), 1): ((2,), 3, 3),
        (-1, (), -2): ((3,), 3, 5),
        (-2, (2,), -1): ((), 2, 2),
    }
    writer = FiniteVisualGaps(cells, rules)
    path = tmp_path / "gaps.gz"
    writer.save(path)
    restored = FiniteVisualGaps.restore(path)
    answer = restored.generate(cells[:3], np.array([True, True, False]))
    assert answer[:, 0].tolist() == [4, 1, 3, 2]
    unknown = np.full((1, 128), 123, np.uint8)
    assert np.array_equal(restored.generate(unknown, np.array([False])), unknown)

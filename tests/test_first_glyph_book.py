import pytest
import torch
from scripts.pointing_glyph_corpus import Row
from scripts.train_first_glyph_book import positions
from scripts.train_pointing_causal_writer import focused_action_loss


def test_weight_reaches_only_the_selected_unpadded_actions():
    a = [[5, 0], [1, 2, 0], [4, 0]]
    values = torch.arange(7, dtype=torch.float64, requires_grad=True)
    loss = focused_action_loss(values, a, [1, None, 0])
    assert loss.item() == -3
    loss.backward()
    assert values.grad.tolist() == [0, -0.5, 0, 0, 0, -0.5, 0]
    assert focused_action_loss(values, a, [None] * 3).item() == 0
    with pytest.raises(ValueError):
        focused_action_loss(values, a, [2, None, 0])


def test_both_question_languages_get_the_same_control_position():
    rows = [
        Row(k, "id", b"", b"", bytes(128 * 3), b"")
        for k in ("novel_chinese", "novel_second", "historical")
    ]
    assert positions(rows, "first") == [0, 0, None]
    i, j, none = positions(rows, "later")
    assert 1 <= i == j <= 3 and none is None

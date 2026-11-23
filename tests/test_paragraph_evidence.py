import pytest
import torch

from ilm.visual_lm.paragraph_evidence import (
    ParagraphRanker,
    PhysicalBank,
    lexical_score,
)
from scripts.paragraph_evidence_corpus import section_pair


def strip(*indices):
    return b"".join(bytes([index]) * 128 for index in indices)


def test_literal_run_and_blank_glyph():
    assert lexical_score(strip(1, 2, 3), strip(0, 1, 2, 3, 0)) == (3, 3)
    assert lexical_score(strip(1, 2, 3), strip(1, 0, 3)) == (1, 2)
    bank = PhysicalBank()
    ids = bank.index(strip(0, 1, 0))
    assert ids[0] == ids[2] and ids[0] != 0
    assert torch.count_nonzero(bank.pooled_ink("cpu")[ids[0]]) == 0
    with pytest.raises(ValueError):
        bank.index(b"short")


def test_padding_and_candidate_order_do_not_change_scores():
    model = ParagraphRanker().eval()
    q = strip(8, 9)
    a, b = strip(1), strip(8, 9, 12, 13)
    both = model.scores(q, [a, b])
    assert both == pytest.approx(
        [model.scores(q, [a])[0], model.scores(q, [b])[0]], abs=1e-5
    )
    assert model.scores(q, [b, a]) == pytest.approx(both[::-1], abs=1e-5)
    assert model.scores(q, [a] * 9) == pytest.approx([both[0]] * 9, abs=1e-5)


def test_natural_boundary_preserves_all_cells():
    page = "甲" * 17 + "。" + "乙" * 22
    assert section_pair(page) == [page[:18], page[18:]]
    assert section_pair(page[:12]) is None

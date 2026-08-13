from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.conditional_visual_density_ratio_data import (
    ConditionalVisualCandidateBank,
    ConditionalVisualRenderConfig,
    build_v29_candidate_bank,
    canonical_target_indices,
    conditional_visual_candidate_bank_receipt,
    conditional_visual_data_boundary_receipt,
    conditional_visual_natural_student_batch,
)
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics


def _statistics() -> VisualCharacterStatistics:
    return VisualCharacterStatistics(
        characters=("中", "文", "天"),
        counts=(9, 7, 5),
        bigram_rows={"中": ((1, 3),), "文": ((2, 2),)},
        visible_character_count=30,
        han_character_count=21,
    )


def test_v29_candidate_bank_is_rendered_images_with_external_receipt() -> None:
    bank = build_v29_candidate_bank(_statistics(), seed=20260923)
    assert isinstance(bank, ConditionalVisualCandidateBank)
    assert bank.images.shape == (2, 3, 1, 32, 32)
    assert bank.canonical.shape == (3, 1, 32, 32)
    receipt = conditional_visual_candidate_bank_receipt(
        bank, include_host_forms=False
    )
    assert receipt["bank_size"] == 3
    assert "host_forms" not in receipt
    assert receipt["checkpoint_contains_bank"] is False
    assert len(receipt["ordered_canonical_row_sha256"]) == 3


def test_v29_target_columns_are_derived_from_exact_pixels() -> None:
    bank = build_v29_candidate_bank(_statistics(), seed=20260923)
    targets = torch.stack((bank.canonical[2], bank.canonical[0]))
    assert torch.equal(
        canonical_target_indices(targets, bank.canonical),
        torch.tensor([2, 0]),
    )
    with pytest.raises(ValueError, match="exactly one"):
        canonical_target_indices(torch.rand(1, 1, 32, 32), bank.canonical)


def test_v29_student_batch_excludes_canonical_pixels_and_metadata() -> None:
    batch = {
        "first_context": torch.rand(2, 64, 1, 32, 32),
        "second_context": torch.rand(2, 64, 1, 32, 32),
        "canonical_target": torch.rand(2, 1, 32, 32),
        "metadata": [{"target": "中"}, {"target": "文"}],
    }
    student = conditional_visual_natural_student_batch(batch)
    assert set(student) == {"first_context", "second_context"}
    assert all(torch.is_floating_point(value) for value in student.values())


def test_v29_data_boundary_has_no_symbolic_student_route() -> None:
    receipt = conditional_visual_data_boundary_receipt()
    assert receipt["canonical_identity_derived_from_exact_pixels"] is True
    assert receipt["training_bank_is_host_only"] is True
    for key in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_external_language_model",
        "candidate_bank_deployed",
    ):
        assert receipt[key] is False


def test_v29_candidate_bank_validates_shapes() -> None:
    with pytest.raises(ValueError, match="invalid shape"):
        ConditionalVisualCandidateBank(
            images=torch.rand(1, 3, 1, 32, 32),
            canonical=torch.rand(3, 1, 32, 32),
            forms=("中", "文", "天"),
            counts=(9, 7, 5),
            font_paths=("a", "b"),
            seed=1,
        )
    with pytest.raises(ValueError, match="font sizes"):
        ConditionalVisualRenderConfig(minimum_font_size=29, maximum_font_size=28)

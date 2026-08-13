from __future__ import annotations

import torch

from ilm.visual_lm.dense_visual_future_data import (
    DenseVisualNaturalDataset,
    DenseVisualRenderConfig,
    canonical_pixel_groups,
    causal_position_weights,
    dense_visual_data_boundary_receipt,
    dense_visual_natural_collate,
    dense_visual_natural_student_batch,
    stratified_causal_positions,
)
from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import visual_cell_partition


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"v28-test-record-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find V28 identifier for {split}")


def _record(split: str = "train") -> VisualGrammarRecord:
    writing = (
        "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        "闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈"
        "剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔"
    )
    return VisualGrammarRecord(
        identifier=_identifier_for(split),
        text=writing,
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )


def test_v28_natural_dataset_exposes_images_but_no_student_metadata() -> None:
    dataset = DenseVisualNaturalDataset(
        [_record()],
        split="train",
        render_config=DenseVisualRenderConfig(augment=False),
        seed=20260917,
        length=2,
    )
    first = dataset[0]
    assert first["first_view"].shape == (68, 1, 32, 32)
    assert first["second_view"].shape == (68, 1, 32, 32)
    assert first["canonical"].shape == (68, 1, 32, 32)
    raw = dense_visual_natural_collate([first, dataset[1]])
    student = dense_visual_natural_student_batch(raw)
    assert set(student) == {"first_view", "second_view"}
    assert student["first_view"].shape == (2, 68, 1, 32, 32)
    assert "metadata" not in student
    assert "canonical" not in student


def test_v28_canonical_groups_are_derived_from_exact_pixels() -> None:
    canonical = torch.zeros(2, 3, 1, 32, 32)
    canonical[0, 1] = 1.0
    canonical[1, 0] = canonical[0, 0]
    canonical[1, 1, 0, 3, 7] = 1.0
    canonical[1, 2] = canonical[0, 1]
    groups = canonical_pixel_groups(canonical)
    assert groups.shape == (2, 3)
    assert groups[0, 0] == groups[1, 0]
    assert groups[0, 1] == groups[1, 2]
    assert groups[0, 0] != groups[1, 1]


def test_v28_stratified_positions_cover_every_quartile_and_fix_endpoint() -> None:
    first = stratified_causal_positions(
        generator=torch.Generator().manual_seed(19)
    )
    second = stratified_causal_positions(
        generator=torch.Generator().manual_seed(19)
    )
    assert torch.equal(first, second)
    assert first.shape == (16,)
    assert int((first < 16).sum()) == 4
    assert int(((first >= 16) & (first < 32)).sum()) == 4
    assert int(((first >= 32) & (first < 48)).sum()) == 4
    assert int((first >= 48).sum()) == 4
    assert int(first[-1]) == 63
    weights = causal_position_weights(first)
    assert torch.all(weights > 0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert weights[-1] == weights.max()


def test_v28_data_boundary_keeps_pixel_groups_out_of_model_input() -> None:
    receipt = dense_visual_data_boundary_receipt()
    assert receipt["canonical_identity_derived_from_exact_pixels"] is True
    assert receipt["canonical_groups_are_temporary_loss_only"] is True
    assert receipt["student_natural_keys"] == ["first_view", "second_view"]
    assert receipt["uses_character_ids"] is False

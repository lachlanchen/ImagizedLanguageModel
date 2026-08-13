from __future__ import annotations

import torch

from ilm.visual_lm.conditional_visual_field_flow_data import (
    ConditionalVisualFlowNaturalDataset,
    ConditionalVisualRenderConfig,
    conditional_visual_field_flow_data_boundary_receipt,
    conditional_visual_flow_natural_collate,
    conditional_visual_flow_natural_student_batch,
)
from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import visual_cell_partition


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"v31-test-record-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find V31 identifier for {split}")


def _record() -> VisualGrammarRecord:
    writing = (
        "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        "闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈"
        "剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔"
    )
    return VisualGrammarRecord(
        identifier=_identifier_for("train"),
        text=writing,
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )


def test_v31_natural_dataset_renders_context_and_real_cross_font_target() -> None:
    dataset = ConditionalVisualFlowNaturalDataset(
        [_record()],
        allowed_targets=set(_record().text),
        split="train",
        render_config=ConditionalVisualRenderConfig(augment=False),
        seed=20261111,
        length=2,
    )
    first = dataset[0]
    assert first["first_context"].shape == (64, 1, 32, 32)
    assert first["second_context"].shape == (64, 1, 32, 32)
    assert first["first_target"].shape == (1, 32, 32)
    assert first["second_target"].shape == (1, 32, 32)
    raw = conditional_visual_flow_natural_collate([first, dataset[1]])
    student = conditional_visual_flow_natural_student_batch(raw)
    assert set(student) == {
        "first_context",
        "second_context",
        "first_target",
        "second_target",
    }
    assert all(torch.is_floating_point(value) for value in student.values())
    assert "canonical_target" not in student
    assert "metadata" not in student


def test_v31_data_boundary_is_image_only_and_bank_free() -> None:
    receipt = conditional_visual_field_flow_data_boundary_receipt()
    assert receipt["output_is_candidate_independent_continuous_distribution"]
    assert receipt["student_natural_keys"] == [
        "first_context",
        "second_context",
        "first_target",
        "second_target",
    ]
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

from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord, load_visual_grammar_manifest
from ilm.visual_lm.visual_cell_data import (
    V25_DEVELOPMENT_FONTS,
    V25_FONT_PARTITIONS,
    V25_FROZEN_FONTS,
    V25_PARTITION_SALT,
    V25_TRAIN_FONTS,
    VisualCellRenderConfig,
    VisualCellStreamDataset,
    assert_image_only_student_batch,
    pack_visual_cells,
    render_visual_cell_stream,
    script_variants,
    student_visual_cell_batch,
    visual_cell_boundary_receipt,
    visual_cell_collate,
    visual_cell_partition,
    visual_cell_partition_receipt,
)


EXPECTED_PARTITION = {
    "identifiers": 7_017,
    "train_identifiers": 6_608,
    "development_identifiers": 190,
    "frozen_identifiers": 219,
    "train_identifiers_sha256": (
        "a3eb1af8e6860aca8589574f41fb2a00a7dcb0e38478e051b701aa8b67541654"
    ),
    "development_identifiers_sha256": (
        "46f3364ddbcdef7a3b6f9147a3ccfab7246deb059516c2524f9026ee01c0b221"
    ),
    "frozen_identifiers_sha256": (
        "7426e2440d378f77c1577698020cd1c266eeed0317c200e9a535d6b5a85cf94f"
    ),
}


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"test-record-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find deterministic identifier for {split}")


def _record(split: str) -> VisualGrammarRecord:
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


def test_v25_partition_matches_preregistration_without_rendering() -> None:
    records = load_visual_grammar_manifest(
        "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
    )
    receipt = visual_cell_partition_receipt(records)
    assert receipt["salt"] == V25_PARTITION_SALT
    assert receipt["frozen_images_instantiated"] is False
    for key, expected in EXPECTED_PARTITION.items():
        assert receipt[key] == expected


def test_v25_fonts_are_disjoint_and_complete() -> None:
    assert V25_FONT_PARTITIONS == {
        "train": V25_TRAIN_FONTS,
        "development": V25_DEVELOPMENT_FONTS,
        "frozen": V25_FROZEN_FONTS,
    }
    all_fonts = (*V25_TRAIN_FONTS, *V25_DEVELOPMENT_FONTS, *V25_FROZEN_FONTS)
    assert len(set(all_fonts)) == len(all_fonts) == 8


def test_text_becomes_a_clean_three_dimensional_visual_time_stream() -> None:
    config = VisualCellRenderConfig(augment=False, script_views="original")
    cells = render_visual_cell_stream(
        "视觉语言模型",
        config=config,
        font_path=V25_DEVELOPMENT_FONTS[0],
        variant=19,
    )
    assert cells.shape == (6, 1, 32, 32)
    assert cells.dtype == torch.float32
    assert torch.all((0.0 <= cells) & (cells <= 1.0))
    assert torch.all(cells.flatten(1).sum(dim=1) > 5.0)
    assert not torch.equal(cells[0], cells[1])

    page = pack_visual_cells(cells, columns=4, gutter=1)
    assert page.size == (4 * 32 + 3, 2 * 32 + 1)


def test_dataset_exposes_only_float_images_to_the_student() -> None:
    dataset = VisualCellStreamDataset(
        [_record("development")],
        split="development",
        render_config=VisualCellRenderConfig(
            augment=False,
            script_views="original",
        ),
        seed=41,
        length=2,
        expose_evaluation_labels=True,
    )
    batch = visual_cell_collate([dataset[0], dataset[1]])
    assert batch["context"].shape == (2, 64, 1, 32, 32)
    assert batch["target"].shape == (2, 64, 1, 32, 32)
    assert batch["reference_context"].shape == (2, 64, 1, 32, 32)
    assert batch["reference_target"].shape == (2, 64, 1, 32, 32)
    assert "target_characters" in batch["metadata"][0]

    student = student_visual_cell_batch(batch)
    assert set(student) == {
        "context",
        "target",
        "reference_context",
        "reference_target",
    }
    assert_image_only_student_batch(student)
    assert all(torch.is_floating_point(value) for value in student.values())

    with pytest.raises(ValueError, match="only the four"):
        assert_image_only_student_batch({**student, "metadata": batch["metadata"]})


def test_frozen_images_are_refused_until_explicit_authorization() -> None:
    with pytest.raises(PermissionError, match="remain sealed"):
        VisualCellStreamDataset(
            [_record("frozen")],
            split="frozen",
            render_config=VisualCellRenderConfig(
                augment=False,
                script_views="original",
            ),
        )


def test_simplified_view_is_an_offline_image_preparation_variant() -> None:
    record = VisualGrammarRecord(
        identifier="script-test",
        text="學習視覺語言模型並生成圖像",
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )
    variants = dict(script_variants(record, mode="original+simplified"))
    assert variants["original"] == "學習視覺語言模型並生成圖像"
    assert variants["simplified"] == "学习视觉语言模型并生成图像"


def test_boundary_receipt_forbids_symbolic_and_identity_channels() -> None:
    receipt = visual_cell_boundary_receipt()
    assert receipt["native_sample_shape"] == [65, 1, 32, 32]
    assert receipt["sequence_axis_is_visual_time"] is True
    assert receipt["geometric_depth_is_one"] is True
    assert receipt["rereads_generated_pixels"] is True
    forbidden = (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_ocr",
        "uses_color_identity_channel",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    )
    assert all(receipt[key] is False for key in forbidden)


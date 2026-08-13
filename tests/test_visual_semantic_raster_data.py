from __future__ import annotations

import json

import pytest
import torch

from ilm.visual_lm.visual_semantic_raster_data import (
    V32_DEVELOPMENT_FONTS,
    V32_TRAIN_FONTS,
    VisualRasterContinuationDataset,
    VisualRasterInstructionDataset,
    VisualRasterRecord,
    VisualRasterRenderConfig,
    VisualRasterWarmupDataset,
    VisualTextRecord,
    load_visual_raster_instructions,
    load_visual_raster_paraphrases,
    normalize_visible_text,
    render_answer_cells,
    render_prompt_strip,
    visual_raster_partition,
    visual_semantic_raster_collate,
    visual_semantic_raster_data_boundary_receipt,
    visual_semantic_raster_student_batch,
)


def _identifier_for(split: str, stream: str) -> str:
    for index in range(100_000):
        identifier = f"v32-test-{stream}-{index}"
        if visual_raster_partition(identifier, stream=stream) == split:
            return identifier
    raise AssertionError(f"could not find V32 identifier for {stream}/{split}")


def _instruction(split: str = "train") -> VisualRasterRecord:
    return VisualRasterRecord(
        identifier=_identifier_for(split, "instruction"),
        prompt="问：天空是什么颜色？",
        answer="蓝色",
        language="zh",
        source="unit-test",
        rights="test-only",
    )


def test_v32_normalization_preserves_historical_codepoints() -> None:
    text = "  甲\n\U00020069\t乙\x00 "
    assert normalize_visible_text(text) == "甲 \U00020069 乙"


def test_v32_prompt_render_is_fixed_visual_strip_and_can_cross_patch_boundaries() -> None:
    config = VisualRasterRenderConfig(
        maximum_prompt_patches=24,
        augment=False,
        nonaligned_origin_probability=1.0,
    )
    pixels, mask, metadata = render_prompt_strip(
        "天地玄黄宇宙洪荒",
        config=config,
        font_path=V32_TRAIN_FONTS[0],
        variant=7,
        force_origin=7,
    )
    assert pixels.shape == (3, 16, 384)
    assert mask.shape == (24,)
    assert torch.is_floating_point(pixels)
    assert 0.0 <= float(pixels.min()) <= float(pixels.max()) <= 1.0
    assert metadata["origin"] == 7
    assert int(mask.sum()) > 1


def test_v32_answer_render_uses_pixels_and_a_post_answer_stop_position() -> None:
    config = VisualRasterRenderConfig(maximum_answer_cells=8, augment=False)
    cells, answer_mask, stop_targets, stop_mask, metadata = render_answer_cells(
        "蓝色",
        config=config,
        font_path=V32_TRAIN_FONTS[0],
        variant=11,
    )
    assert cells.shape == (8, 1, 24, 24)
    assert answer_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert stop_targets.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert stop_mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert float(cells[:2].sum()) > 0.0
    assert float(cells[2:].sum()) == 0.0
    assert metadata["normalized_length"] == 2


def test_v32_instruction_dataset_removes_metadata_at_student_boundary() -> None:
    dataset = VisualRasterInstructionDataset(
        [_instruction()],
        split="train",
        render_config=VisualRasterRenderConfig(
            maximum_prompt_patches=32,
            maximum_answer_cells=8,
            augment=False,
        ),
        seed=32,
    )
    raw = visual_semantic_raster_collate([dataset[0], dataset[0]])
    student = visual_semantic_raster_student_batch(raw)
    assert "metadata" in raw
    assert "metadata" not in student
    assert all(torch.is_floating_point(value) for value in student.values())
    assert student["prompt_pixels"].shape == (2, 3, 16, 512)
    assert student["answer_cells"].shape == (2, 8, 1, 24, 24)


def test_v32_continuation_partition_is_source_disjoint() -> None:
    records = [
        VisualTextRecord(
            identifier=_identifier_for("train", "public-domain"),
            text="天地玄黄宇宙洪荒日月盈昃辰宿列张" * 4,
            language="zh-Hant",
            source="unit-test",
            rights="test-only",
        ),
        VisualTextRecord(
            identifier=_identifier_for("development", "public-domain"),
            text="寒来暑往秋收冬藏闰余成岁律吕调阳" * 4,
            language="zh-Hant",
            source="unit-test",
            rights="test-only",
        ),
    ]
    dataset = VisualRasterContinuationDataset(
        records,
        split="development",
        render_config=VisualRasterRenderConfig(
            maximum_prompt_patches=48,
            maximum_answer_cells=8,
            augment=False,
        ),
        seed=91,
        length=2,
    )
    sample = dataset[0]
    assert "development" in visual_raster_partition(
        sample["metadata"]["continuation"]["source_identifier"],
        stream="public-domain",
    )
    assert sample["answer_mask"].sum() >= 1


def test_v32_continuation_respects_declared_prompt_capacity() -> None:
    record = VisualTextRecord(
        identifier=_identifier_for("train", "public-domain"),
        text="天地玄黄宇宙洪荒日月盈昃辰宿列张" * 16,
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )
    dataset = VisualRasterContinuationDataset(
        [record],
        split="train",
        render_config=VisualRasterRenderConfig(
            maximum_prompt_patches=32,
            maximum_answer_cells=8,
            augment=False,
        ),
        seed=17,
        length=8,
        maximum_prompt_cells=24,
    )
    for sample in dataset:
        assert sample["metadata"]["continuation"]["prompt_cells"] <= 24
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_v32_warmup_renders_answers_without_spending_work_on_prompts() -> None:
    record = VisualTextRecord(
        identifier=_identifier_for("train", "public-domain"),
        text="天地玄黄宇宙洪荒日月盈昃辰宿列张" * 2,
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )
    config = VisualRasterRenderConfig(
        maximum_prompt_patches=16,
        maximum_answer_cells=8,
        augment=False,
    )
    sample = VisualRasterWarmupDataset(
        [record],
        split="train",
        render_config=config,
        seed=32,
        length=2,
    )[0]
    assert torch.equal(sample["prompt_pixels"], torch.ones(3, 16, 256))
    assert sample["prompt_mask"].sum() == 0
    assert sample["answer_mask"].sum() >= 1
    assert sample["metadata"]["warmup_only"] is True


def test_v32_instruction_and_paraphrase_loaders_keep_text_host_side(tmp_path) -> None:
    instruction_path = tmp_path / "instructions.json"
    instruction_path.write_text(
        json.dumps(
            [
                {"instruction": "计算二加二", "input": "", "output": "四"},
                {"instruction": "空答案", "input": "", "output": ""},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    records = load_visual_raster_instructions(instruction_path)
    assert len(records) == 1
    assert records[0].identifier == "alpaca-zh:0"

    paraphrase_path = tmp_path / "paraphrases.jsonl"
    paraphrase_path.write_text(
        json.dumps(
            {"identifier": "any-source:0", "paraphrase": "二与二相加是多少？"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paraphrases = load_visual_raster_paraphrases(paraphrase_path, records)
    assert len(paraphrases) == 1
    assert paraphrases[0].answer == "四"
    assert "二与二" in paraphrases[0].prompt


def test_v32_student_boundary_rejects_symbolic_or_integer_tensors() -> None:
    config = VisualRasterRenderConfig(
        maximum_prompt_patches=32,
        maximum_answer_cells=8,
        augment=False,
    )
    dataset = VisualRasterInstructionDataset(
        [_instruction()],
        split="train",
        render_config=config,
        seed=7,
    )
    raw = visual_semantic_raster_collate([dataset[0]])
    raw["prompt_mask"] = raw["prompt_mask"].long()
    with pytest.raises(TypeError, match="floating tensor"):
        visual_semantic_raster_student_batch(raw)

    receipt = visual_semantic_raster_data_boundary_receipt()
    assert receipt["primary_output_is_raster"]
    assert receipt["metadata_enters_student"] is False
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


def test_v32_development_font_is_distinct_from_training_fonts() -> None:
    assert set(V32_DEVELOPMENT_FONTS).isdisjoint(V32_TRAIN_FONTS)

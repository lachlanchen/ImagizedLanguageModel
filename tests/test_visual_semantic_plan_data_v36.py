from __future__ import annotations

from pathlib import Path

import torch

from ilm.visual_lm.visual_semantic_plan_data import (
    V36_CHUNKS,
    V36_PATCHES,
    V36_WIDTH,
    VisualSemanticPlanAnswerDataset,
    VisualSemanticPlanPromptDataset,
    VisualSemanticPlanRenderConfig,
    render_visual_semantic_plan_record,
    render_visual_sentence_strip,
    split_visual_answer_chunks,
    visual_semantic_plan_answer_collate,
    visual_semantic_plan_collate,
    visual_semantic_plan_data_boundary_receipt,
    visual_semantic_plan_pixel_batch,
    visual_semantic_plan_prompt_collate,
    visual_semantic_plan_stream_record_index,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord


FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _record() -> VisualRasterRecord:
    return VisualRasterRecord(
        identifier="test:1",
        prompt="\u95ee\uff1a\u8bf7\u89e3\u91ca\u8fd9\u4e2a\u5b57\u3002",
        answer="\u8a00\u8bed\u8868\u8fbe\u3002",
        language="zh",
        source="unit-test",
        rights="test-only",
    )


def test_v36_strip_and_spatial_chunks_are_pixel_only() -> None:
    if not Path(FONT).is_file():
        return
    config = VisualSemanticPlanRenderConfig(augment=False)
    pixels, mask, _ = render_visual_sentence_strip(
        "\u8a00\u8bed\u6587\u5b57",
        config=config,
        font_path=FONT,
        variant=7,
        force_origin=0,
    )
    chunks, chunk_mask = split_visual_answer_chunks(pixels, mask)
    assert pixels.shape == (3, 16, V36_WIDTH)
    assert mask.shape == (V36_PATCHES,)
    assert chunks.shape == (V36_CHUNKS, 3, 16, V36_WIDTH)
    assert chunk_mask.shape == (V36_CHUNKS, V36_PATCHES)
    assert torch.equal(chunks[0, :, :, :256], pixels[:, :, :256])
    assert torch.all(chunks[0, :, :, 256:] == 1.0)
    assert torch.equal(chunk_mask[0, :16], mask[:16])
    assert torch.all(chunk_mask[0, 16:] == 0.0)


def test_v36_collate_excludes_metadata_from_pixel_batch() -> None:
    if not Path(FONT).is_file():
        return
    sample = render_visual_semantic_plan_record(
        _record(),
        split="train",
        config=VisualSemanticPlanRenderConfig(augment=False),
        variant=11,
    )
    collated = visual_semantic_plan_collate([sample, sample])
    pixels = visual_semantic_plan_pixel_batch(collated)
    assert "metadata" in collated
    assert "metadata" not in pixels
    assert all(torch.is_floating_point(value) for value in pixels.values())
    assert pixels["prompt_pixels"].shape == (2, 3, 16, V36_WIDTH)
    assert pixels["answer_chunk_pixels"].shape == (
        2,
        V36_CHUNKS,
        3,
        16,
        V36_WIDTH,
    )


def test_v36_separated_prompt_and_answer_streams() -> None:
    if not Path(FONT).is_file():
        return
    config = VisualSemanticPlanRenderConfig(augment=False)
    prompt_dataset = VisualSemanticPlanPromptDataset(
        [_record()],
        split="train",
        render_config=config,
        seed=23,
        length=2,
    )
    answer_dataset = VisualSemanticPlanAnswerDataset(
        [_record()],
        split="train",
        render_config=config,
        seed=29,
    )

    prompt_sample = prompt_dataset[0]
    repeated_prompt = prompt_dataset[0]
    answer_sample = answer_dataset[0]
    prompt_batch = visual_semantic_plan_prompt_collate([prompt_sample])
    answer_batch = visual_semantic_plan_answer_collate([answer_sample])

    assert torch.equal(prompt_sample["prompt_pixels"], repeated_prompt["prompt_pixels"])
    assert prompt_sample["metadata"]["identifier"] == "test:1"
    assert answer_sample["metadata"]["identifier"] == "test:1"
    assert set(prompt_batch) == {
        "prompt_pixels",
        "prompt_mask",
        "prompt_view_pixels",
        "prompt_view_mask",
        "metadata",
    }
    assert set(answer_batch) == {
        "answer_pixels",
        "answer_mask",
        "answer_chunk_pixels",
        "answer_chunk_mask",
        "answer_length",
        "metadata",
    }
    assert prompt_batch["prompt_pixels"].shape == (1, 3, 16, V36_WIDTH)
    assert answer_batch["answer_chunk_pixels"].shape == (
        1,
        V36_CHUNKS,
        3,
        16,
        V36_WIDTH,
    )


def test_v36_data_boundary_has_no_symbolic_student_interface() -> None:
    receipt = visual_semantic_plan_data_boundary_receipt()
    assert receipt["deployable_keys"] == ["prompt_pixels", "prompt_mask"]
    assert receipt["uses_strings"] is False
    assert receipt["uses_token_ids"] is False
    assert receipt["uses_unicode_ids"] is False
    assert receipt["uses_character_ids"] is False
    assert receipt["candidate_bank_deployed"] is False


def test_v36_stream_is_a_permutation_before_repeating() -> None:
    first_epoch = [
        visual_semantic_plan_stream_record_index(index, records=17, seed=36)
        for index in range(17)
    ]
    second_epoch = [
        visual_semantic_plan_stream_record_index(index, records=17, seed=36)
        for index in range(17, 34)
    ]
    assert sorted(first_epoch) == list(range(17))
    assert sorted(second_epoch) == list(range(17))
    assert first_epoch == second_epoch
    crossing = [
        visual_semantic_plan_stream_record_index(index, records=17, seed=36)
        for index in range(12, 22)
    ]
    assert len(set(crossing)) == len(crossing)

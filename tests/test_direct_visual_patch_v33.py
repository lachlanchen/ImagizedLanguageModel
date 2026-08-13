from __future__ import annotations

import inspect

import torch

from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchRenderConfig,
    direct_patch_collate,
    direct_patch_data_boundary_receipt,
    direct_patch_student_batch,
    render_direct_patch_instruction,
)
from ilm.visual_lm.direct_visual_patch_lm import (
    DirectVisualPatchConfig,
    DirectVisualPatchLM,
    _resize_projection_kernel,
    direct_visual_patch_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord


def tiny_model() -> DirectVisualPatchLM:
    return DirectVisualPatchLM(
        DirectVisualPatchConfig(
            patch_size=32,
            maximum_patches=12,
            hidden_size=64,
            layers=2,
            attention_heads=4,
            key_value_heads=4,
            intermediate_size=128,
        )
    )


def test_forward_and_generation_are_pixels_only() -> None:
    model = tiny_model().eval()
    pixels = torch.ones(2, 1, 32, 32 * 4)
    pixels[:, :, 8:24, 6:27] = 0.0
    mask = torch.ones(2, 4)
    output = model(pixels, mask)
    assert output.patch_logits.shape == (2, 4, 1, 32, 32)
    assert output.stop_logits.shape == (2, 4)
    generation = model.generate(
        pixels,
        mask,
        maximum_new_patches=2,
        minimum_new_patches=2,
    )
    assert generation.patches.shape == (2, 2, 1, 32, 32)
    assert generation.strips().shape == (2, 1, 32, 64)
    assert generation.lengths.tolist() == [2, 2]
    assert tuple(inspect.signature(model.generate).parameters) == (
        "pixels",
        "patch_mask",
        "maximum_new_patches",
        "minimum_new_patches",
        "pixel_threshold",
        "stop_threshold",
    )


def test_boundary_has_no_symbolic_parameter_tables() -> None:
    receipt = direct_visual_patch_boundary_receipt(tiny_model())
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["uses_strings"] is False
    assert receipt["uses_token_ids"] is False
    assert receipt["uses_unicode_ids"] is False
    assert receipt["uses_ocr"] is False
    assert receipt["uses_runtime_teacher"] is False


def test_projection_resize_has_expected_shape_and_scaling() -> None:
    source = torch.ones(8, 1, 8, 16)
    resized_input = _resize_projection_kernel(
        source,
        size=32,
        preserve_input_variance=True,
    )
    resized_output = _resize_projection_kernel(
        source,
        size=32,
        preserve_input_variance=False,
    )
    assert resized_input.shape == (8, 1, 32, 32)
    assert resized_output.shape == (8, 1, 32, 32)
    assert torch.allclose(resized_input.mean(), torch.tensor(2**-1.5))
    assert torch.allclose(resized_output.mean(), torch.tensor(1.0))


def test_instruction_renderer_separates_metadata_from_student() -> None:
    record = VisualRasterRecord(
        identifier="unit:capital",
        prompt="问：法国的首都是什么？",
        answer="巴黎。",
        language="zh",
        source="unit",
        rights="unit",
    )
    config = DirectPatchRenderConfig(
        maximum_patches=16,
        maximum_prompt_patches=12,
        maximum_answer_patches=4,
        augment=False,
    )
    sample = render_direct_patch_instruction(
        record,
        split="train",
        config=config,
        variant=7,
    )
    batch = direct_patch_collate([sample, sample])
    student = direct_patch_student_batch(batch)
    receipt = direct_patch_data_boundary_receipt(batch)
    assert "metadata" not in student
    assert receipt["metadata_excluded"] is True
    assert receipt["student_contains_strings"] is False
    assert student["pixels"].shape == (2, 1, 32, 32 * 16)
    prompt_count = sample["metadata"]["prompt_patches"]
    answer_count = sample["metadata"]["answer_patches"]
    assert sample["next_patch_mask"][prompt_count - 1].item() == 1.0
    assert sample["stop_targets"][prompt_count + answer_count - 1].item() == 1.0


def test_invalid_raster_shape_is_rejected() -> None:
    model = tiny_model()
    pixels = torch.ones(1, 1, 31, 64)
    mask = torch.ones(1, 2)
    try:
        model(pixels, mask)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid V33 raster shape was accepted")

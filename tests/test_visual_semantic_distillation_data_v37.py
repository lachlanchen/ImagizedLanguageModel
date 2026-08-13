from __future__ import annotations

from dataclasses import replace

import torch

from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_TRAIN_FONTS,
    V37_WIDTH,
    VisualSemanticDistillationRenderConfig,
    render_visual_semantic_distillation_strip,
    visual_semantic_distillation_data_boundary_receipt,
    visual_semantic_distillation_stream_record_index,
)


def test_mask_is_invariant_to_pixel_augmentation() -> None:
    augmented = VisualSemanticDistillationRenderConfig(augment=True)
    clean = replace(augmented, augment=False)
    arguments = {
        "text": "视觉语言模型读取文字图像。",
        "font_path": V37_TRAIN_FONTS[0],
        "font_size": 10,
        "variant": 123_456,
        "force_origin": 7,
    }
    noisy_pixels, noisy_mask, noisy_meta = render_visual_semantic_distillation_strip(
        config=augmented, **arguments
    )
    clean_pixels, clean_mask, clean_meta = render_visual_semantic_distillation_strip(
        config=clean, **arguments
    )

    assert noisy_pixels.shape == (3, V37_PATCH_SIZE, V37_WIDTH)
    assert clean_pixels.shape == noisy_pixels.shape
    assert torch.equal(noisy_mask, clean_mask)
    assert noisy_meta["active_patches"] == clean_meta["active_patches"]
    assert noisy_meta["mask_source"] == "clean-pre-augmentation-raster"
    assert not torch.equal(noisy_pixels, clean_pixels)


def test_background_shift_cannot_activate_blank_patches() -> None:
    config = VisualSemanticDistillationRenderConfig(augment=True)
    _, mask, metadata = render_visual_semantic_distillation_strip(
        "中",
        config=config,
        font_path=V37_TRAIN_FONTS[0],
        font_size=10,
        variant=2,
        force_origin=0,
    )

    assert 1 <= int(mask.sum()) <= 2
    assert not bool(mask[2:].any())
    assert metadata["active_patches"] == int(mask.sum())


def test_stream_visits_each_record_once_per_cycle() -> None:
    indices = [
        visual_semantic_distillation_stream_record_index(
            index,
            records=17,
            seed=20263700,
        )
        for index in range(34)
    ]

    assert len(set(indices[:17])) == 17
    assert indices[:17] == indices[17:]


def test_data_boundary_is_image_only() -> None:
    receipt = visual_semantic_distillation_data_boundary_receipt()

    assert receipt["deployable_keys"] == ["prompt_pixels", "prompt_mask"]
    assert receipt["mask_shape"] == [V37_PATCHES]
    assert receipt["mask_source"] == "clean-pre-augmentation-raster"
    assert not receipt["uses_strings"]
    assert not receipt["uses_token_ids"]
    assert not receipt["uses_unicode_ids"]
    assert not receipt["uses_ocr"]

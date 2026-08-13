from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from .visual_semantic_raster_data import (
    VisualRasterRecord,
    load_visual_raster_instructions,
    normalize_visible_text,
    visual_raster_partition,
)


V36_ARCHITECTURE = "visual-semantic-plan-v36"
V36_PATCH_SIZE = 16
V36_PATCHES = 64
V36_WIDTH = V36_PATCH_SIZE * V36_PATCHES
V36_CHUNKS = 4
V36_CHUNK_PATCHES = V36_PATCHES // V36_CHUNKS
V36_PLAN_SLOTS = V36_CHUNKS + 1
V36_SPLITS = ("train", "development", "sealed")

V36_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
)
V36_DEVELOPMENT_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
V36_SEALED_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
)

V36_PIXEL_KEYS = (
    "prompt_pixels",
    "prompt_mask",
    "prompt_view_pixels",
    "prompt_view_mask",
    "answer_pixels",
    "answer_mask",
    "answer_chunk_pixels",
    "answer_chunk_mask",
    "answer_length",
)


@dataclass(frozen=True)
class VisualSemanticPlanRenderConfig:
    patch_size: int = V36_PATCH_SIZE
    patches: int = V36_PATCHES
    font_size: int = 11
    maximum_origin: int = 15
    augment: bool = True

    def __post_init__(self) -> None:
        if self.patch_size != V36_PATCH_SIZE:
            raise ValueError("V36 fixes 16-pixel patches")
        if self.patches != V36_PATCHES:
            raise ValueError("V36 fixes 64 visual positions")
        if not 8 <= self.font_size <= 15:
            raise ValueError("V36 font size must be in [8,15]")
        if not 0 <= self.maximum_origin < self.patch_size:
            raise ValueError("V36 origin must stay inside the first patch")

    @property
    def width(self) -> int:
        return self.patch_size * self.patches


def visual_semantic_plan_fonts(split: str) -> tuple[str, ...]:
    if split == "train":
        candidates = V36_TRAIN_FONTS
    elif split == "development":
        candidates = V36_DEVELOPMENT_FONTS
    elif split == "sealed":
        candidates = V36_SEALED_FONTS
    else:
        raise ValueError(f"unknown V36 split: {split}")
    available = tuple(path for path in candidates if Path(path).is_file())
    if not available:
        raise FileNotFoundError(f"no V36 fonts are available for split={split!r}")
    return available


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return ImageFont.truetype(path, size=size)


def _text_geometry(
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    image = Image.new("L", (1, 1), color=255)
    return ImageDraw.Draw(image).textbbox((0, 0), text, font=font)


def visual_text_fits(
    text: str,
    *,
    config: VisualSemanticPlanRenderConfig,
    font_path: str,
    origin: int,
) -> bool:
    normalized = normalize_visible_text(text)
    if not normalized:
        return False
    left, _, right, _ = _text_geometry(
        normalized,
        font=_font(font_path, config.font_size),
    )
    return origin + right - left <= config.width


def render_visual_sentence_strip(
    text: str,
    *,
    config: VisualSemanticPlanRenderConfig,
    font_path: str,
    variant: int,
    force_origin: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    normalized = normalize_visible_text(text)
    if not normalized:
        raise ValueError("V36 cannot render empty visible text")
    rng = random.Random(int(variant))
    origin = (
        rng.randint(0, config.maximum_origin)
        if force_origin is None
        else int(force_origin)
    )
    if not 0 <= origin <= config.maximum_origin:
        raise ValueError("V36 origin is outside the configured range")

    font = _font(font_path, config.font_size)
    left, top, right, bottom = _text_geometry(normalized, font=font)
    text_width = right - left
    if origin + text_width > config.width:
        raise ValueError("V36 text does not fit the fixed visual strip")
    text_height = bottom - top
    image = Image.new(
        "L",
        (config.width, config.patch_size),
        color=255,
    )
    draw = ImageDraw.Draw(image)
    y = (config.patch_size - text_height) // 2 - top
    draw.text((origin - left, y), normalized, font=font, fill=0)

    if config.augment:
        if rng.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.30)))
        array = np.asarray(image, dtype=np.float32) / 255.0
        contrast = rng.uniform(0.94, 1.06)
        array = 0.5 + (array - 0.5) * contrast
        if rng.random() < 0.15:
            noise = np.random.default_rng(int(variant)).normal(
                0.0,
                rng.uniform(0.002, 0.012),
                array.shape,
            )
            array = array + noise
        array = np.clip(array, 0.0, 1.0)
    else:
        array = np.asarray(image, dtype=np.float32) / 255.0

    ink = 1.0 - array
    patch_ink = ink.reshape(
        config.patch_size,
        config.patches,
        config.patch_size,
    ).max(axis=(0, 2))
    mask = torch.from_numpy((patch_ink > 1.0 / 255.0).astype(np.float32))
    if not bool(mask.any()):
        raise ValueError("V36 rendered strip contains no visible ink")
    pixels = torch.from_numpy(array.copy()).unsqueeze(0).repeat(3, 1, 1)
    return pixels, mask, {
        "font_path": font_path,
        "font_size": config.font_size,
        "origin": origin,
        "normalized_length": len(normalized),
        "active_patches": int(mask.sum().item()),
    }


def split_visual_answer_chunks(
    answer_pixels: torch.Tensor,
    answer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (3, V36_PATCH_SIZE, V36_WIDTH)
    if answer_pixels.shape != expected or answer_mask.shape != (V36_PATCHES,):
        raise ValueError("V36 answer strip has an invalid shape")
    chunks = torch.ones(
        V36_CHUNKS,
        3,
        V36_PATCH_SIZE,
        V36_WIDTH,
        dtype=answer_pixels.dtype,
    )
    masks = torch.zeros(V36_CHUNKS, V36_PATCHES, dtype=answer_mask.dtype)
    chunk_width = V36_CHUNK_PATCHES * V36_PATCH_SIZE
    for index in range(V36_CHUNKS):
        source_start = index * chunk_width
        source_end = source_start + chunk_width
        chunks[index, :, :, :chunk_width] = answer_pixels[
            :,
            :,
            source_start:source_end,
        ]
        masks[index, :V36_CHUNK_PATCHES] = answer_mask[
            index * V36_CHUNK_PATCHES : (index + 1) * V36_CHUNK_PATCHES
        ]
    return chunks, masks


def visual_semantic_plan_record_fits(
    record: VisualRasterRecord,
    *,
    split: str,
    config: VisualSemanticPlanRenderConfig,
) -> bool:
    origin = config.maximum_origin if split == "train" else 0
    for font_path in visual_semantic_plan_fonts(split):
        if not visual_text_fits(
            record.prompt,
            config=config,
            font_path=font_path,
            origin=origin,
        ):
            return False
        if not visual_text_fits(
            record.answer,
            config=config,
            font_path=font_path,
            origin=origin,
        ):
            return False
    return True


def render_visual_semantic_plan_record(
    record: VisualRasterRecord,
    *,
    split: str,
    config: VisualSemanticPlanRenderConfig,
    variant: int,
) -> dict[str, Any]:
    fonts = visual_semantic_plan_fonts(split)
    rng = random.Random(int(variant))
    prompt_font = fonts[rng.randrange(len(fonts))]
    view_font = fonts[rng.randrange(len(fonts))]
    answer_font = fonts[rng.randrange(len(fonts))]
    force_origin = None if split == "train" else 0

    prompt_pixels, prompt_mask, prompt_meta = render_visual_sentence_strip(
        record.prompt,
        config=config,
        font_path=prompt_font,
        variant=variant,
        force_origin=force_origin,
    )
    prompt_view_pixels, prompt_view_mask, prompt_view_meta = (
        render_visual_sentence_strip(
            record.prompt,
            config=config,
            font_path=view_font,
            variant=variant + 17,
            force_origin=force_origin,
        )
    )
    answer_pixels, answer_mask, answer_meta = render_visual_sentence_strip(
        record.answer,
        config=config,
        font_path=answer_font,
        variant=variant + 31,
        force_origin=force_origin,
    )
    answer_chunk_pixels, answer_chunk_mask = split_visual_answer_chunks(
        answer_pixels,
        answer_mask,
    )
    return {
        "prompt_pixels": prompt_pixels,
        "prompt_mask": prompt_mask,
        "prompt_view_pixels": prompt_view_pixels,
        "prompt_view_mask": prompt_view_mask,
        "answer_pixels": answer_pixels,
        "answer_mask": answer_mask,
        "answer_chunk_pixels": answer_chunk_pixels,
        "answer_chunk_mask": answer_chunk_mask,
        "answer_length": answer_mask.sum().float(),
        "metadata": {
            "identifier": record.identifier,
            "language": record.language,
            "source": record.source,
            "rights": record.rights,
            "prompt": prompt_meta,
            "prompt_view": prompt_view_meta,
            "answer": answer_meta,
        },
    }


def load_v36_instruction_records(path: str | Path) -> list[VisualRasterRecord]:
    return load_visual_raster_instructions(
        path,
        maximum_prompt_characters=160,
        maximum_answer_cells=32,
    )


class VisualSemanticPlanDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VisualRasterRecord],
        *,
        split: str,
        render_config: VisualSemanticPlanRenderConfig,
        seed: int,
        length: int | None = None,
        include_all_records: bool = False,
    ) -> None:
        if split not in V36_SPLITS:
            raise ValueError(f"unknown V36 split: {split}")
        partitioned = [
            record
            for record in records
            if include_all_records
            or visual_raster_partition(record.identifier, stream="instruction")
            == split
        ]
        self.rejected_identifiers = tuple(
            record.identifier
            for record in partitioned
            if not visual_semantic_plan_record_fits(
                record,
                split=split,
                config=render_config,
            )
        )
        rejected = set(self.rejected_identifiers)
        selected = [record for record in partitioned if record.identifier not in rejected]
        if not selected:
            raise ValueError(f"V36 instruction split {split!r} is empty")
        self.records = tuple(selected)
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = len(selected) if length is None else int(length)
        if self.length < 1:
            raise ValueError("V36 dataset length must be positive")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        rng = random.Random(self.seed + index * 1_000_003)
        record = (
            self.records[index]
            if self.length == len(self.records)
            else self.records[rng.randrange(len(self.records))]
        )
        return render_visual_semantic_plan_record(
            record,
            split=self.split,
            config=self.render_config,
            variant=rng.randrange(2**31),
        )


def visual_semantic_plan_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V36 batch")
    return {
        key: torch.stack([item[key] for item in batch]) for key in V36_PIXEL_KEYS
    } | {"metadata": [item.get("metadata", {}) for item in batch]}


def visual_semantic_plan_pixel_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    result = {key: batch[key] for key in V36_PIXEL_KEYS}
    for key, value in result.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V36 pixel value {key!r} must be floating")
    batch_size = result["prompt_pixels"].shape[0]
    strip_shape = (batch_size, 3, V36_PATCH_SIZE, V36_WIDTH)
    for key in ("prompt_pixels", "prompt_view_pixels", "answer_pixels"):
        if result[key].shape != strip_shape:
            raise ValueError(f"V36 {key} has an invalid shape")
    mask_shape = (batch_size, V36_PATCHES)
    for key in ("prompt_mask", "prompt_view_mask", "answer_mask"):
        if result[key].shape != mask_shape:
            raise ValueError(f"V36 {key} has an invalid shape")
    if result["answer_chunk_pixels"].shape != (
        batch_size,
        V36_CHUNKS,
        3,
        V36_PATCH_SIZE,
        V36_WIDTH,
    ):
        raise ValueError("V36 answer chunk pixels have an invalid shape")
    if result["answer_chunk_mask"].shape != (
        batch_size,
        V36_CHUNKS,
        V36_PATCHES,
    ):
        raise ValueError("V36 answer chunk masks have an invalid shape")
    if result["answer_length"].shape != (batch_size,):
        raise ValueError("V36 visual answer lengths have an invalid shape")
    return result


def visual_semantic_plan_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V36_ARCHITECTURE,
        "pixel_keys": list(V36_PIXEL_KEYS),
        "deployable_keys": ["prompt_pixels", "prompt_mask"],
        "prompt_shape": [3, V36_PATCH_SIZE, V36_WIDTH],
        "answer_shape": [3, V36_PATCH_SIZE, V36_WIDTH],
        "answer_chunk_shape": [V36_CHUNKS, 3, V36_PATCH_SIZE, V36_WIDTH],
        "strings_exist_only_before_pixel_batch": True,
        "metadata_enters_student": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "candidate_bank_deployed": False,
    }

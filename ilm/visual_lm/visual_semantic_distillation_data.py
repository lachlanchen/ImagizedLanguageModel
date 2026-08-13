from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
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


V37_ARCHITECTURE = "visual-semantic-distillation-v37"
V37_TARGET_ARCHITECTURE = "visual-semantic-distillation-target-bank-v37"
V37_PATCH_SIZE = 16
V37_PATCHES = 64
V37_WIDTH = V37_PATCH_SIZE * V37_PATCHES
V37_SEMANTIC_DIM = 1024
V37_SPLITS = ("train", "development", "sealed")

V37_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
)
V37_DEVELOPMENT_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
V37_HELD_FONT = "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf"
V37_SEALED_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc"

V37_PIXEL_KEYS = (
    "prompt_pixels",
    "prompt_mask",
    "prompt_view_pixels",
    "prompt_view_mask",
    "answer_pixels",
    "answer_mask",
    "answer_view_pixels",
    "answer_view_mask",
)


@dataclass(frozen=True)
class VisualSemanticDistillationRenderConfig:
    patch_size: int = V37_PATCH_SIZE
    patches: int = V37_PATCHES
    minimum_font_size: int = 8
    maximum_font_size: int = 11
    evaluation_font_size: int = 10
    maximum_origin: int = 15
    augment: bool = True

    def __post_init__(self) -> None:
        if self.patch_size != V37_PATCH_SIZE or self.patches != V37_PATCHES:
            raise ValueError("V37 fixes a 16 by 1024 visual strip")
        if not 8 <= self.minimum_font_size <= self.maximum_font_size <= 15:
            raise ValueError("V37 font-size range is invalid")
        if not self.minimum_font_size <= self.evaluation_font_size <= 15:
            raise ValueError("V37 evaluation font size is invalid")
        if not 0 <= self.maximum_origin < self.patch_size:
            raise ValueError("V37 origin must stay inside the first patch")

    @property
    def width(self) -> int:
        return self.patch_size * self.patches


def _require_font(path: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return path


def visual_semantic_distillation_fonts(split: str) -> tuple[str, ...]:
    if split == "train":
        candidates = V37_TRAIN_FONTS
    elif split == "development":
        candidates = (V37_DEVELOPMENT_FONT,)
    elif split == "sealed":
        candidates = (V37_SEALED_FONT,)
    else:
        raise ValueError(f"unknown V37 split: {split}")
    return tuple(_require_font(path) for path in candidates)


@lru_cache(maxsize=128)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_require_font(path), size=size)


def _text_geometry(
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    image = Image.new("L", (1, 1), color=255)
    return ImageDraw.Draw(image).textbbox((0, 0), text, font=font)


def visual_text_fits_v37(
    text: str,
    *,
    config: VisualSemanticDistillationRenderConfig,
    font_path: str,
    font_size: int,
    origin: int,
) -> bool:
    normalized = normalize_visible_text(text)
    if not normalized:
        return False
    left, _, right, _ = _text_geometry(
        normalized,
        font=_font(font_path, font_size),
    )
    return origin + right - left <= config.width


def _clean_patch_mask(
    array: np.ndarray, *, config: VisualSemanticDistillationRenderConfig
) -> torch.Tensor:
    if array.shape != (config.patch_size, config.width):
        raise ValueError("V37 clean raster has an invalid shape")
    ink = 1.0 - array
    patch_ink = ink.reshape(
        config.patch_size,
        config.patches,
        config.patch_size,
    ).max(axis=(0, 2))
    mask = torch.from_numpy((patch_ink > 1.0 / 255.0).astype(np.float32))
    if not bool(mask.any()):
        raise ValueError("V37 rendered strip contains no visible ink")
    return mask


def render_visual_semantic_distillation_strip(
    text: str,
    *,
    config: VisualSemanticDistillationRenderConfig,
    font_path: str,
    font_size: int,
    variant: int,
    force_origin: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    normalized = normalize_visible_text(text)
    if not normalized:
        raise ValueError("V37 cannot render empty visible text")
    if not config.minimum_font_size <= font_size <= config.maximum_font_size:
        if font_size != config.evaluation_font_size:
            raise ValueError("V37 font size is outside the configured range")
    rng = random.Random(int(variant))
    origin = (
        rng.randint(0, config.maximum_origin)
        if force_origin is None
        else int(force_origin)
    )
    if not 0 <= origin <= config.maximum_origin:
        raise ValueError("V37 origin is outside the configured range")

    face = _font(font_path, font_size)
    left, top, right, bottom = _text_geometry(normalized, font=face)
    if origin + right - left > config.width:
        raise ValueError("V37 text does not fit the fixed visual strip")
    text_height = bottom - top
    clean = Image.new("L", (config.width, config.patch_size), color=255)
    draw = ImageDraw.Draw(clean)
    y = (config.patch_size - text_height) // 2 - top
    draw.text((origin - left, y), normalized, font=face, fill=0)

    clean_array = np.asarray(clean, dtype=np.float32) / 255.0
    mask = _clean_patch_mask(clean_array, config=config)
    image = clean
    if config.augment:
        if rng.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.35)))
        array = np.asarray(image, dtype=np.float32) / 255.0
        contrast = rng.uniform(0.92, 1.08)
        array = 0.5 + (array - 0.5) * contrast
        if rng.random() < 0.20:
            noise = np.random.default_rng(int(variant)).normal(
                0.0,
                rng.uniform(0.002, 0.010),
                array.shape,
            )
            array = array + noise
        array = np.clip(array, 0.0, 1.0)
    else:
        array = clean_array

    pixels = torch.from_numpy(array.copy()).unsqueeze(0).repeat(3, 1, 1)
    return (
        pixels,
        mask,
        {
            "font_path": font_path,
            "font_size": font_size,
            "origin": origin,
            "normalized_length": len(normalized),
            "active_patches": int(mask.sum().item()),
            "mask_source": "clean-pre-augmentation-raster",
        },
    )


def _selection_fonts(split: str) -> tuple[str, ...]:
    if split == "development":
        return (_require_font(V37_DEVELOPMENT_FONT), _require_font(V37_HELD_FONT))
    return visual_semantic_distillation_fonts(split)


def visual_semantic_distillation_record_fits(
    record: VisualRasterRecord,
    *,
    split: str,
    config: VisualSemanticDistillationRenderConfig,
) -> bool:
    font_size = (
        config.maximum_font_size if split == "train" else config.evaluation_font_size
    )
    origin = config.maximum_origin if split == "train" else 0
    return all(
        visual_text_fits_v37(
            text,
            config=config,
            font_path=font_path,
            font_size=font_size,
            origin=origin,
        )
        for font_path in _selection_fonts(split)
        for text in (record.prompt, record.answer)
    )


def load_v37_instruction_records(path: str | Path) -> list[VisualRasterRecord]:
    return load_visual_raster_instructions(
        path,
        maximum_prompt_characters=160,
        maximum_answer_cells=32,
    )


def select_v37_instruction_records(
    records: Sequence[VisualRasterRecord],
    *,
    split: str,
    render_config: VisualSemanticDistillationRenderConfig,
    include_all_records: bool = False,
) -> tuple[tuple[VisualRasterRecord, ...], tuple[str, ...]]:
    if split not in V37_SPLITS:
        raise ValueError(f"unknown V37 split: {split}")
    partitioned = [
        record
        for record in records
        if include_all_records
        or visual_raster_partition(record.identifier, stream="instruction") == split
    ]
    rejected = tuple(
        record.identifier
        for record in partitioned
        if not visual_semantic_distillation_record_fits(
            record,
            split=split,
            config=render_config,
        )
    )
    rejected_set = set(rejected)
    selected = tuple(
        record for record in partitioned if record.identifier not in rejected_set
    )
    if not selected:
        raise ValueError(f"V37 instruction split {split!r} is empty")
    return selected, rejected


@lru_cache(maxsize=4_096)
def _visual_stream_affine_permutation(size: int, seed: int) -> tuple[int, int]:
    if size < 1:
        raise ValueError("V37 visual stream permutation is invalid")
    if size == 1:
        return 0, 1
    rng = random.Random(int(seed))
    offset = rng.randrange(size)
    stride = rng.randrange(1, size)
    while gcd(stride, size) != 1:
        stride = stride % (size - 1) + 1
    return offset, stride


def visual_semantic_distillation_stream_record_index(
    index: int,
    *,
    records: int,
    seed: int,
) -> int:
    if index < 0 or records < 1:
        raise ValueError("V37 visual stream index is invalid")
    _, position = divmod(index, records)
    offset, stride = _visual_stream_affine_permutation(records, seed)
    return (offset + position * stride) % records


def _render_view(
    text: str,
    *,
    split: str,
    config: VisualSemanticDistillationRenderConfig,
    rng: random.Random,
    variant: int,
    font_override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    fonts = (
        (_require_font(font_override),)
        if font_override is not None
        else visual_semantic_distillation_fonts(split)
    )
    font_path = fonts[rng.randrange(len(fonts))]
    if split == "train":
        font_size = rng.randint(config.minimum_font_size, config.maximum_font_size)
        force_origin = None
    else:
        font_size = config.evaluation_font_size
        force_origin = 0
    return render_visual_semantic_distillation_strip(
        text,
        config=config,
        font_path=font_path,
        font_size=font_size,
        variant=variant,
        force_origin=force_origin,
    )


def render_visual_semantic_distillation_record(
    record: VisualRasterRecord,
    *,
    split: str,
    config: VisualSemanticDistillationRenderConfig,
    variant: int,
    font_override: str | None = None,
) -> dict[str, Any]:
    rng = random.Random(int(variant))
    fields: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "identifier": record.identifier,
        "language": record.language,
        "source": record.source,
        "rights": record.rights,
    }
    view_specs = (
        ("prompt", "prompt_pixels", "prompt_mask", 0),
        ("prompt", "prompt_view_pixels", "prompt_view_mask", 17),
        ("answer", "answer_pixels", "answer_mask", 31),
        ("answer", "answer_view_pixels", "answer_view_mask", 47),
    )
    for source_field, pixel_key, mask_key, offset in view_specs:
        pixels, mask, view_metadata = _render_view(
            getattr(record, source_field),
            split=split,
            config=config,
            rng=rng,
            variant=variant + offset,
            font_override=font_override,
        )
        fields[pixel_key] = pixels
        fields[mask_key] = mask
        metadata[pixel_key.removesuffix("_pixels")] = view_metadata
    fields["metadata"] = metadata
    return fields


def canonical_answer_length_v37(
    record: VisualRasterRecord,
    *,
    split: str,
    config: VisualSemanticDistillationRenderConfig,
) -> float:
    if split == "train":
        font_path = _require_font(V37_TRAIN_FONTS[0])
    elif split == "development":
        font_path = _require_font(V37_DEVELOPMENT_FONT)
    elif split == "sealed":
        font_path = _require_font(V37_SEALED_FONT)
    else:
        raise ValueError(f"unknown V37 split: {split}")
    clean_config = VisualSemanticDistillationRenderConfig(
        minimum_font_size=config.minimum_font_size,
        maximum_font_size=config.maximum_font_size,
        evaluation_font_size=config.evaluation_font_size,
        maximum_origin=config.maximum_origin,
        augment=False,
    )
    _, mask, _ = render_visual_semantic_distillation_strip(
        record.answer,
        config=clean_config,
        font_path=font_path,
        font_size=config.evaluation_font_size,
        variant=0,
        force_origin=0,
    )
    return float(mask.sum())


class VisualSemanticDistillationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[VisualRasterRecord],
        *,
        split: str,
        render_config: VisualSemanticDistillationRenderConfig,
        seed: int,
        length: int | None = None,
        include_all_records: bool = False,
        font_override: str | None = None,
    ) -> None:
        self.records, self.rejected_identifiers = select_v37_instruction_records(
            records,
            split=split,
            render_config=render_config,
            include_all_records=include_all_records,
        )
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = len(self.records) if length is None else int(length)
        self.font_override = font_override
        if self.length < 1:
            raise ValueError("V37 dataset length must be positive")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        record_index = visual_semantic_distillation_stream_record_index(
            index,
            records=len(self.records),
            seed=self.seed,
        )
        record = self.records[record_index]
        variant = self.seed + index * 1_000_003
        return render_visual_semantic_distillation_record(
            record,
            split=self.split,
            config=self.render_config,
            variant=variant,
            font_override=self.font_override,
        )


def visual_semantic_distillation_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V37 batch")
    return {
        key: torch.stack([item[key] for item in batch]) for key in V37_PIXEL_KEYS
    } | {"metadata": [item.get("metadata", {}) for item in batch]}


def visual_semantic_distillation_pixel_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    result = {key: batch[key] for key in V37_PIXEL_KEYS}
    batch_size = result["prompt_pixels"].shape[0]
    strip_shape = (batch_size, 3, V37_PATCH_SIZE, V37_WIDTH)
    mask_shape = (batch_size, V37_PATCHES)
    for key, value in result.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V37 pixel value {key!r} must be floating")
        expected = mask_shape if key.endswith("_mask") else strip_shape
        if value.shape != expected:
            raise ValueError(f"V37 {key} has an invalid shape")
    return result


def visual_semantic_distillation_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V37_ARCHITECTURE,
        "pixel_keys": list(V37_PIXEL_KEYS),
        "deployable_keys": ["prompt_pixels", "prompt_mask"],
        "prompt_shape": [3, V37_PATCH_SIZE, V37_WIDTH],
        "mask_shape": [V37_PATCHES],
        "mask_source": "clean-pre-augmentation-raster",
        "strings_exist_only_before_tensor_boundary": True,
        "metadata_enters_model": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "candidate_bank_deployed": False,
    }

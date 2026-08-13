from __future__ import annotations

import hashlib
import math
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
    VisualTextRecord,
    normalize_visible_text,
)


V33_PATCH_SIZE = 32
V33_MAXIMUM_PATCHES = 96
V33_MAXIMUM_PROMPT_PATCHES = 64
V33_MAXIMUM_ANSWER_PATCHES = 31
V33_SPLITS = ("train", "development", "sealed")
V33_STUDENT_KEYS = (
    "pixels",
    "patch_mask",
    "next_patch_mask",
    "reconstruction_mask",
    "stop_targets",
    "stop_mask",
)

V33_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
)
V33_DEVELOPMENT_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
V33_SEALED_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
)


@dataclass(frozen=True)
class DirectPatchRenderConfig:
    patch_size: int = V33_PATCH_SIZE
    maximum_patches: int = V33_MAXIMUM_PATCHES
    maximum_prompt_patches: int = V33_MAXIMUM_PROMPT_PATCHES
    maximum_answer_patches: int = V33_MAXIMUM_ANSWER_PATCHES
    minimum_font_size: int = 24
    maximum_font_size: int = 28
    augment: bool = True

    def __post_init__(self) -> None:
        if self.patch_size != V33_PATCH_SIZE:
            raise ValueError("V33 fixes 32-pixel patches")
        if not 4 <= self.maximum_patches <= V33_MAXIMUM_PATCHES:
            raise ValueError("V33 maximum patch count is invalid")
        if not 2 <= self.maximum_prompt_patches < self.maximum_patches:
            raise ValueError("V33 prompt patch count is invalid")
        if not 1 <= self.maximum_answer_patches < self.maximum_patches:
            raise ValueError("V33 answer patch count is invalid")
        if self.maximum_prompt_patches + self.maximum_answer_patches > self.maximum_patches:
            raise ValueError("V33 prompt and answer regions exceed the strip")
        if not 16 <= self.minimum_font_size <= self.maximum_font_size <= 31:
            raise ValueError("V33 font range is invalid")


def direct_patch_partition(identifier: str, *, stream: str) -> str:
    if stream not in {"public-domain", "instruction"}:
        raise ValueError("V33 stream must be public-domain or instruction")
    digest = hashlib.sha256(f"v33:{stream}:{identifier}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    if stream == "public-domain":
        if fraction < 0.96:
            return "train"
        if fraction < 0.98:
            return "development"
        return "sealed"
    if fraction < 0.94:
        return "train"
    if fraction < 0.97:
        return "development"
    return "sealed"


def _font_paths(split: str) -> tuple[str, ...]:
    paths = {
        "train": V33_TRAIN_FONTS,
        "development": V33_DEVELOPMENT_FONTS,
        "sealed": V33_SEALED_FONTS,
    }.get(split)
    if paths is None:
        raise ValueError(f"unknown V33 split: {split}")
    available = tuple(path for path in paths if Path(path).is_file())
    if not available:
        raise FileNotFoundError(f"V33 has no available font for split={split!r}")
    return available


def _patchify_strip(image: Image.Image, patch_size: int) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).unsqueeze(0)
    return tensor.unfold(-1, patch_size, patch_size).permute(2, 0, 1, 3).contiguous()


def _render_region(
    text: str,
    *,
    patch_count: int,
    config: DirectPatchRenderConfig,
    font_path: str,
    font_size: int,
    origin: int,
    variant: int,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    normalized = normalize_visible_text(text)
    if not normalized:
        raise ValueError("V33 cannot render empty text")
    if not 0 <= origin < config.patch_size:
        raise ValueError("V33 origin must lie inside the first patch")
    font = ImageFont.truetype(font_path, size=font_size)
    width = patch_count * config.patch_size
    image = Image.new("L", (width, config.patch_size), 255)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), normalized, font=font)
    text_width = right - left
    text_height = bottom - top
    if origin + text_width > width:
        raise ValueError("V33 rendered text exceeds its patch region")
    y = (config.patch_size - text_height) // 2 - top
    draw.text((origin - left, y), normalized, font=font, fill=0)

    rng = random.Random(variant)
    if config.augment and rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.30)))
    array = np.asarray(image, dtype=np.float32)
    if config.augment:
        contrast = rng.uniform(0.92, 1.08)
        array = 127.5 + (array - 127.5) * contrast
        if rng.random() < 0.12:
            noise = np.random.default_rng(variant).normal(
                0.0,
                rng.uniform(0.25, 2.0),
                array.shape,
            )
            array += noise
    binary = (np.clip(array, 0, 255) >= 127.5).astype(np.uint8) * 255
    image = Image.fromarray(binary, mode="L")
    occupied = max(1, math.ceil((origin + text_width) / config.patch_size))
    return _patchify_strip(image, config.patch_size), occupied, {
        "font_path": font_path,
        "font_size": font_size,
        "origin": origin,
        "text_width": text_width,
        "occupied_patches": occupied,
        "normalized_length": len(normalized),
    }


def render_direct_patch_instruction(
    record: VisualRasterRecord,
    *,
    split: str,
    config: DirectPatchRenderConfig,
    variant: int,
) -> dict[str, Any]:
    rng = random.Random(variant)
    fonts = _font_paths(split)
    font_path = fonts[rng.randrange(len(fonts))]
    font_size = rng.randint(config.minimum_font_size, config.maximum_font_size)
    prompt_text = f"{normalize_visible_text(record.prompt)} 答："
    answer_text = normalize_visible_text(record.answer)
    prompt_patches, prompt_count, prompt_meta = _render_region(
        prompt_text,
        patch_count=config.maximum_prompt_patches,
        config=config,
        font_path=font_path,
        font_size=font_size,
        origin=rng.randrange(config.patch_size),
        variant=variant,
    )
    answer_patches, answer_count, answer_meta = _render_region(
        answer_text,
        patch_count=config.maximum_answer_patches,
        config=config,
        font_path=font_path,
        font_size=font_size,
        origin=rng.randrange(config.patch_size),
        variant=variant + 1,
    )
    total = prompt_count + answer_count
    if total > config.maximum_patches:
        raise ValueError("V33 rendered instruction exceeds the full strip")
    patches = torch.ones(
        config.maximum_patches,
        1,
        config.patch_size,
        config.patch_size,
    )
    patches[:prompt_count] = prompt_patches[:prompt_count]
    patches[prompt_count:total] = answer_patches[:answer_count]
    patch_mask = torch.zeros(config.maximum_patches)
    patch_mask[:total] = 1.0
    next_patch_mask = torch.zeros(config.maximum_patches)
    next_patch_mask[prompt_count - 1 : total - 1] = 1.0
    reconstruction_mask = patch_mask.clone()
    stop_targets = torch.zeros(config.maximum_patches)
    stop_targets[total - 1] = 1.0
    stop_mask = torch.zeros(config.maximum_patches)
    stop_mask[prompt_count - 1 : total] = 1.0
    strip = patches.permute(1, 2, 0, 3).reshape(
        1,
        config.patch_size,
        config.maximum_patches * config.patch_size,
    )
    return {
        "pixels": strip,
        "patch_mask": patch_mask,
        "next_patch_mask": next_patch_mask,
        "reconstruction_mask": reconstruction_mask,
        "stop_targets": stop_targets,
        "stop_mask": stop_mask,
        "metadata": {
            "identifier": record.identifier,
            "language": record.language,
            "source": record.source,
            "rights": record.rights,
            "prompt_patches": prompt_count,
            "answer_patches": answer_count,
            "prompt": prompt_meta,
            "answer": answer_meta,
        },
    }


def render_direct_patch_continuation(
    record: VisualTextRecord,
    *,
    split: str,
    config: DirectPatchRenderConfig,
    variant: int,
) -> dict[str, Any]:
    rng = random.Random(variant)
    fonts = _font_paths(split)
    font_path = fonts[rng.randrange(len(fonts))]
    font_size = rng.randint(config.minimum_font_size, config.maximum_font_size)
    normalized = normalize_visible_text(record.text)
    if len(normalized) < 4:
        raise ValueError("V33 continuation text is too short")
    maximum_characters = max(8, config.maximum_patches)
    if len(normalized) > maximum_characters:
        start = rng.randrange(len(normalized) - maximum_characters + 1)
        normalized = normalized[start : start + maximum_characters]
    rendered, occupied, meta = _render_region(
        normalized,
        patch_count=config.maximum_patches,
        config=config,
        font_path=font_path,
        font_size=font_size,
        origin=rng.randrange(config.patch_size),
        variant=variant,
    )
    if occupied < 2:
        raise ValueError("V33 continuation requires at least two patches")
    patch_mask = torch.zeros(config.maximum_patches)
    patch_mask[:occupied] = 1.0
    next_patch_mask = torch.zeros(config.maximum_patches)
    next_patch_mask[: occupied - 1] = 1.0
    reconstruction_mask = patch_mask.clone()
    stop_targets = torch.zeros(config.maximum_patches)
    stop_mask = torch.zeros(config.maximum_patches)
    strip = rendered.permute(1, 2, 0, 3).reshape(
        1,
        config.patch_size,
        config.maximum_patches * config.patch_size,
    )
    return {
        "pixels": strip,
        "patch_mask": patch_mask,
        "next_patch_mask": next_patch_mask,
        "reconstruction_mask": reconstruction_mask,
        "stop_targets": stop_targets,
        "stop_mask": stop_mask,
        "metadata": {
            "identifier": record.identifier,
            "language": record.language,
            "source": record.source,
            "rights": record.rights,
            "occupied_patches": occupied,
            "render": meta,
        },
    }


class DirectPatchInstructionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[VisualRasterRecord],
        *,
        split: str,
        config: DirectPatchRenderConfig,
        variants_per_record: int = 64,
        seed: int = 20_263_300,
    ) -> None:
        self.records = tuple(
            record
            for record in records
            if direct_patch_partition(record.identifier, stream="instruction") == split
        )
        if not self.records:
            raise ValueError(f"V33 instruction split {split!r} is empty")
        if variants_per_record < 1:
            raise ValueError("V33 variants per record must be positive")
        self.split = split
        self.config = config
        self.variants_per_record = variants_per_record
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.variants_per_record

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index % len(self.records)]
        cycle = index // len(self.records)
        variant = self.seed + cycle * len(self.records) + index % len(self.records)
        return render_direct_patch_instruction(
            record,
            split=self.split,
            config=self.config,
            variant=variant,
        )


class DirectPatchContinuationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[VisualTextRecord],
        *,
        split: str,
        config: DirectPatchRenderConfig,
        variants_per_record: int = 64,
        seed: int = 20_263_300,
    ) -> None:
        self.records = tuple(
            record
            for record in records
            if direct_patch_partition(record.identifier, stream="public-domain") == split
        )
        if not self.records:
            raise ValueError(f"V33 continuation split {split!r} is empty")
        if variants_per_record < 1:
            raise ValueError("V33 variants per record must be positive")
        self.split = split
        self.config = config
        self.variants_per_record = variants_per_record
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.variants_per_record

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index % len(self.records)]
        cycle = index // len(self.records)
        variant = self.seed + cycle * len(self.records) + index % len(self.records)
        return render_direct_patch_continuation(
            record,
            split=self.split,
            config=self.config,
            variant=variant,
        )


def direct_patch_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("V33 cannot collate an empty batch")
    return {
        key: torch.stack([sample[key] for sample in samples])
        for key in V33_STUDENT_KEYS
    } | {"metadata": [sample["metadata"] for sample in samples]}


def direct_patch_student_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    result = {key: batch[key] for key in V33_STUDENT_KEYS}
    if set(result) != set(V33_STUDENT_KEYS):
        raise RuntimeError("V33 student batch has an invalid boundary")
    if not all(isinstance(value, torch.Tensor) for value in result.values()):
        raise TypeError("V33 student batch must contain tensors only")
    return result


def direct_patch_data_boundary_receipt(batch: Mapping[str, Any]) -> dict[str, Any]:
    student = direct_patch_student_batch(batch)
    return {
        "student_keys": sorted(student),
        "metadata_excluded": "metadata" not in student,
        "all_student_values_are_tensors": all(
            isinstance(value, torch.Tensor) for value in student.values()
        ),
        "student_contains_strings": any(
            isinstance(value, str) for value in student.values()
        ),
        "shapes": {key: list(value.shape) for key, value in student.items()},
    }


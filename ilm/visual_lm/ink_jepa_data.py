from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset



RETINAL_CJK_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc",
)
RETINAL_CJK_AVAILABLE_FONTS = tuple(
    path for path in RETINAL_CJK_FONT_PATHS if Path(path).exists()
)


@dataclass(frozen=True)
class VisualGrammarRecord:
    identifier: str
    text: str
    language: str
    source: str
    rights: str


@dataclass(frozen=True)
class RetinalRenderConfig:
    height: int = 192
    width: int = 768
    font_size: int = 25
    minimum_font_size: int = 21
    margin: int = 12
    cell_padding: int = 3
    augment: bool = True

    def __post_init__(self) -> None:
        if self.height < 96 or self.width < 256:
            raise ValueError("retinal page is too small")
        if self.minimum_font_size > self.font_size:
            raise ValueError("minimum_font_size cannot exceed font_size")

    @property
    def cell_width(self) -> int:
        return self.font_size + self.cell_padding

    @property
    def line_height(self) -> int:
        return self.font_size + max(5, self.font_size // 4)

    @property
    def columns(self) -> int:
        return max(1, (self.width - 2 * self.margin) // self.cell_width)

    @property
    def rows(self) -> int:
        return max(1, (self.height - 2 * self.margin) // self.line_height)

    @property
    def capacity(self) -> int:
        return self.columns * self.rows


def _stable_fraction(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


@lru_cache(maxsize=128)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _font(variant: int, size: int) -> ImageFont.ImageFont:
    if not RETINAL_CJK_AVAILABLE_FONTS:
        return ImageFont.load_default()
    return _load_font(
        RETINAL_CJK_AVAILABLE_FONTS[variant % len(RETINAL_CJK_AVAILABLE_FONTS)],
        size,
    )


@lru_cache(maxsize=65_536)
def _font_supports_character(path: str, character: str) -> bool:
    font = _load_font(path, 24)
    missing = bytes(font.getmask("\U0010ffff", mode="L"))
    return bytes(font.getmask(character, mode="L")) != missing


@lru_cache(maxsize=1)
def _shared_retinal_codepoints() -> frozenset[int] | None:
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    shared: set[int] | None = None
    for path in RETINAL_CJK_AVAILABLE_FONTS:
        if Path(path).suffix.lower() == ".ttc":
            font = TTFont(path, fontNumber=0, lazy=True)
        else:
            font = TTFont(path, lazy=True)
        try:
            coverage = set(font.getBestCmap() or {})
        finally:
            font.close()
        shared = coverage if shared is None else shared.intersection(coverage)
    return frozenset(shared or ())


@lru_cache(maxsize=65_536)
def retinal_character_supported(character: str) -> bool:
    """Return whether every selected retinal face has a real glyph."""

    if len(character) != 1:
        raise ValueError("retinal character coverage accepts one character")
    if character.isspace():
        return True
    if not RETINAL_CJK_AVAILABLE_FONTS:
        return ord(character) < 128
    shared = _shared_retinal_codepoints()
    if shared is not None:
        return ord(character) in shared
    return all(
        _font_supports_character(path, character)
        for path in RETINAL_CJK_AVAILABLE_FONTS
    )


@lru_cache(maxsize=1)
def retinal_font_manifest() -> tuple[dict[str, str | int], ...]:
    output: list[dict[str, str | int]] = []
    for value in RETINAL_CJK_AVAILABLE_FONTS:
        path = Path(value)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        output.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return tuple(output)


def normalize_visual_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _page_segment(text: str, offset: int, capacity: int) -> str:
    if len(text) <= capacity:
        return text
    offset = max(0, min(offset, len(text) - capacity))
    if offset > 0:
        boundary = max(text.rfind(mark, max(0, offset - 24), offset + 1) for mark in "。！？；\n")
        if boundary >= 0:
            offset = boundary + 1
    return text[offset : offset + capacity]


def render_retinal_page(
    text: str,
    *,
    config: RetinalRenderConfig,
    variant: int,
) -> torch.Tensor:
    """Render aligned writing cells; return continuous ink intensity in [0, 1]."""

    rng = random.Random(variant)
    image = Image.new("L", (config.width, config.height), 255)
    draw = ImageDraw.Draw(image)
    size = rng.randint(config.minimum_font_size, config.font_size)
    font = _font(variant, size)
    row = 0
    column = 0
    for character in text:
        if character == "\n":
            row += 1
            column = 0
            if row >= config.rows:
                break
            continue
        if column >= config.columns:
            row += 1
            column = 0
        if row >= config.rows:
            break
        if not character.isspace():
            left = config.margin + column * config.cell_width
            top = config.margin + row * config.line_height
            box = draw.textbbox((0, 0), character, font=font)
            glyph_width = box[2] - box[0]
            glyph_height = box[3] - box[1]
            x = left + (config.cell_width - glyph_width) / 2 - box[0]
            y = top + (config.line_height - glyph_height) / 2 - box[1]
            if config.augment:
                x += rng.uniform(-0.8, 0.8)
                y += rng.uniform(-0.6, 0.6)
            draw.text((x, y), character, font=font, fill=0)
        column += 1

    if config.augment:
        if rng.random() < 0.45:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 1.28))
        if rng.random() < 0.30:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.50)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if config.augment and rng.random() < 0.35:
        noise = np.random.default_rng(variant).normal(0.0, rng.uniform(0.002, 0.015), array.shape)
        array = np.clip(array + noise, 0.0, 1.0)
    return torch.from_numpy(1.0 - array.astype(np.float32))[None]


def load_visual_grammar_manifest(path: str | Path) -> list[VisualGrammarRecord]:
    records: list[VisualGrammarRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = normalize_visual_text(str(item.get("text", "")))
            if not text:
                continue
            identifier = str(item.get("id", item.get("identifier", f"line:{line_number}")))
            records.append(
                VisualGrammarRecord(
                    identifier=identifier,
                    text=text,
                    language=str(item.get("language", "zh")),
                    source=str(item.get("source", Path(path).name)),
                    rights=str(item.get("rights", "unspecified")),
                )
            )
    if not records:
        raise ValueError(f"visual grammar manifest contains no usable records: {path}")
    return records


class VisualGrammarDataset(Dataset):
    """Return cross-render image pairs without exposing strings to the model."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        render_config: RetinalRenderConfig,
        split: str,
        validation_fraction: float = 0.03,
        length: int | None = None,
        seed: int = 0,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        selected: list[VisualGrammarRecord] = []
        for record in records:
            validation = _stable_fraction(record.identifier) < validation_fraction
            if split == "all" or (split == "validation" and validation) or (split == "train" and not validation):
                selected.append(record)
        if not selected:
            raise ValueError(f"no visual grammar records selected for split={split}")
        self.records = selected
        self.config = render_config
        self.length = int(length) if length is not None else len(selected)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        record = self.records[index % len(self.records)] if self.length <= len(self.records) else rng.choice(self.records)
        maximum_offset = max(0, len(record.text) - self.config.capacity)
        offset = rng.randint(0, maximum_offset) if maximum_offset else 0
        segment = _page_segment(record.text, offset, self.config.capacity)
        first_variant = rng.randrange(2**31)
        second_variant = rng.randrange(2**31)
        return {
            "view_a": render_retinal_page(segment, config=self.config, variant=first_variant),
            "view_b": render_retinal_page(segment, config=self.config, variant=second_variant),
            "metadata": {
                "id": record.identifier,
                "source": record.source,
                "language": record.language,
                "rights": record.rights,
                "offset": offset,
                "visible_characters": len(segment),
            },
        }


def visual_grammar_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "view_a": torch.stack([item["view_a"] for item in batch]),
        "view_b": torch.stack([item["view_b"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def retinal_layout(text: str, config: RetinalRenderConfig) -> list[tuple[int, int, int]]:
    """Return source offsets and visual cells using the renderer's layout rules."""

    positions: list[tuple[int, int, int]] = []
    row = 0
    column = 0
    for source_offset, character in enumerate(text):
        if character == "\n":
            row += 1
            column = 0
            if row >= config.rows:
                break
            continue
        if column >= config.columns:
            row += 1
            column = 0
        if row >= config.rows:
            break
        if not character.isspace():
            positions.append((source_offset, row, column))
        column += 1
    return positions


def retinal_cell_bounds(
    *,
    row: int,
    column: int,
    config: RetinalRenderConfig,
) -> tuple[int, int, int, int]:
    if not 0 <= row < config.rows or not 0 <= column < config.columns:
        raise ValueError(f"retinal cell ({row}, {column}) is outside the page")
    left = config.margin + column * config.cell_width
    top = config.margin + row * config.line_height
    right = min(config.width, left + config.cell_width)
    bottom = min(config.height, top + config.line_height)
    return left, top, right, bottom


def extract_retinal_fovea(
    page: torch.Tensor,
    *,
    row: int,
    column: int,
    config: RetinalRenderConfig,
    fovea_size: int,
) -> torch.Tensor:
    if page.ndim != 3 or page.shape[0] != 1:
        raise ValueError("retinal page must have shape [1, height, width]")
    left, top, right, bottom = retinal_cell_bounds(row=row, column=column, config=config)
    crop = page[:, top:bottom, left:right]
    if crop.shape[-2] > fovea_size or crop.shape[-1] > fovea_size:
        crop = F.interpolate(
            crop[None],
            size=(min(fovea_size, crop.shape[-2]), min(fovea_size, crop.shape[-1])),
            mode="bilinear",
            align_corners=False,
        )[0]
    output = page.new_zeros(1, fovea_size, fovea_size)
    top_padding = (fovea_size - crop.shape[-2]) // 2
    left_padding = (fovea_size - crop.shape[-1]) // 2
    output[:, top_padding : top_padding + crop.shape[-2], left_padding : left_padding + crop.shape[-1]] = crop
    return output


def place_retinal_fovea(
    page: torch.Tensor,
    fovea: torch.Tensor,
    *,
    row: int,
    column: int,
    config: RetinalRenderConfig,
) -> torch.Tensor:
    """Paste continuous ink into one visual cell while preserving existing ink."""

    if page.ndim != 3 or page.shape[0] != 1:
        raise ValueError("retinal page must have shape [1, height, width]")
    if fovea.ndim != 3 or fovea.shape[0] != 1 or fovea.shape[-2] != fovea.shape[-1]:
        raise ValueError("fovea must have shape [1, size, size]")
    left, top, right, bottom = retinal_cell_bounds(row=row, column=column, config=config)
    cell_height = bottom - top
    cell_width = right - left
    scaled_height = min(fovea.shape[-2], cell_height)
    scaled_width = min(fovea.shape[-1], cell_width)
    top_padding = (fovea.shape[-2] - scaled_height) // 2
    left_padding = (fovea.shape[-1] - scaled_width) // 2
    crop = fovea[
        :,
        top_padding : top_padding + scaled_height,
        left_padding : left_padding + scaled_width,
    ]
    if (scaled_height, scaled_width) != (cell_height, cell_width):
        crop = F.interpolate(
            crop[None],
            size=(cell_height, cell_width),
            mode="bilinear",
            align_corners=False,
        )[0]
    output = page.clone()
    output[:, top:bottom, left:right] = torch.maximum(output[:, top:bottom, left:right], crop.clamp(0, 1))
    return output


def advance_retinal_cursor(
    row: int,
    column: int,
    config: RetinalRenderConfig,
) -> tuple[int, int] | None:
    retinal_cell_bounds(row=row, column=column, config=config)
    column += 1
    if column >= config.columns:
        row += 1
        column = 0
    if row >= config.rows:
        return None
    return row, column


def retinal_cursor_after_text(
    text: str,
    config: RetinalRenderConfig,
) -> tuple[int, int] | None:
    """Compute the write position in the boundary renderer, never inside the model."""

    row = 0
    column = 0
    for character in text:
        if character == "\n":
            row += 1
            column = 0
        else:
            if column >= config.columns:
                row += 1
                column = 0
            if row >= config.rows:
                return None
            column += 1
        if row >= config.rows:
            return None
    if column >= config.columns:
        row += 1
        column = 0
    return None if row >= config.rows else (row, column)


def infer_retinal_cursor(
    page: torch.Tensor,
    config: RetinalRenderConfig,
    *,
    minimum_mean_ink: float = 0.002,
) -> tuple[int, int] | None:
    """Find the first visual cell after the last occupied cell without OCR."""

    if page.ndim != 3 or page.shape != (1, config.height, config.width):
        raise ValueError("retinal page dimensions do not match the render configuration")
    last_occupied: tuple[int, int] | None = None
    for row in range(config.rows):
        for column in range(config.columns):
            left, top, right, bottom = retinal_cell_bounds(row=row, column=column, config=config)
            if float(page[:, top:bottom, left:right].float().mean()) >= minimum_mean_ink:
                last_occupied = (row, column)
    if last_occupied is None:
        return 0, 0
    return advance_retinal_cursor(*last_occupied, config)


def future_retinal_masks(
    *,
    row: int,
    column: int,
    config: RetinalRenderConfig,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    future = torch.zeros(config.height, config.width, dtype=torch.float32)
    left, top, right, bottom = retinal_cell_bounds(row=row, column=column, config=config)
    future[top:bottom, left:] = 1.0
    future[bottom:] = 1.0
    target = torch.zeros_like(future)
    target[top:bottom, left:right] = 1.0
    hidden_mask = F.max_pool2d(future[None, None], patch_size, patch_size)[0, 0].bool()
    target_mask = F.max_pool2d(target[None, None], patch_size, patch_size)[0, 0].bool()
    return hidden_mask, target_mask


class FovealContinuationDataset(Dataset):
    """Create image-prefix to next-ink samples without returning character labels."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        render_config: RetinalRenderConfig,
        patch_size: int,
        fovea_size: int,
        split: str,
        validation_fraction: float = 0.03,
        length: int | None = None,
        minimum_context_cells: int = 8,
        seed: int = 0,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        selected = []
        for record in records:
            validation = _stable_fraction(record.identifier) < validation_fraction
            if split == "all" or (split == "validation" and validation) or (split == "train" and not validation):
                selected.append(record)
        if not selected:
            raise ValueError(f"no continuation records selected for split={split}")
        if render_config.height % patch_size or render_config.width % patch_size:
            raise ValueError("retinal render dimensions must be divisible by patch_size")
        self.records = selected
        self.config = render_config
        self.patch_size = int(patch_size)
        self.fovea_size = int(fovea_size)
        self.length = int(length) if length is not None else len(selected)
        self.minimum_context_cells = int(minimum_context_cells)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        for _ in range(32):
            record = self.records[index % len(self.records)] if self.length <= len(self.records) else rng.choice(self.records)
            maximum_offset = max(0, len(record.text) - self.config.capacity)
            offset = rng.randint(0, maximum_offset) if maximum_offset else 0
            segment = _page_segment(record.text, offset, self.config.capacity)
            positions = retinal_layout(segment, self.config)
            if len(positions) > self.minimum_context_cells:
                break
        else:
            raise ValueError("could not find a record with enough visible continuation cells")
        position_index = rng.randint(self.minimum_context_cells, len(positions) - 1)
        source_offset, row, column = positions[position_index]
        variant = rng.randrange(2**31)
        full_page = render_retinal_page(segment, config=self.config, variant=variant)
        context = render_retinal_page(segment[:source_offset], config=self.config, variant=variant)
        target = extract_retinal_fovea(
            full_page,
            row=row,
            column=column,
            config=self.config,
            fovea_size=self.fovea_size,
        )
        hidden_mask, target_mask = future_retinal_masks(
            row=row,
            column=column,
            config=self.config,
            patch_size=self.patch_size,
        )
        return {
            "context": context,
            "target": target,
            "hidden_mask": hidden_mask,
            "target_mask": target_mask,
            "metadata": {
                "id": record.identifier,
                "source": record.source,
                "rights": record.rights,
                "row": row,
                "column": column,
                "context_cells": position_index,
            },
        }


def foveal_continuation_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "hidden_mask": torch.stack([item["hidden_mask"] for item in batch]),
        "target_mask": torch.stack([item["target_mask"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }

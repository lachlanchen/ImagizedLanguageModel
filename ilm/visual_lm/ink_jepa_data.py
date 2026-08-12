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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset

from .folio_data import FOLIO_AVAILABLE_FONTS


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
    if not FOLIO_AVAILABLE_FONTS:
        return ImageFont.load_default()
    return _load_font(FOLIO_AVAILABLE_FONTS[variant % len(FOLIO_AVAILABLE_FONTS)], size)


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

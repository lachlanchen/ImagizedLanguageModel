from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset


FOLIO_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/arphic-bkai00mp/bkai00mp.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
)
FOLIO_AVAILABLE_FONTS = tuple(path for path in FOLIO_FONT_PATHS if Path(path).exists())


@dataclass(frozen=True)
class FolioRenderConfig:
    height: int = 192
    width: int = 768
    font_size: int = 28
    minimum_font_size: int = 18
    margin: int = 16
    augment: bool = True

    def __post_init__(self) -> None:
        if self.height < 96 or self.width < 256:
            raise ValueError("folio canvas is too small for visible writing")


def stable_fraction(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


@lru_cache(maxsize=128)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _font(variant: int, size: int) -> ImageFont.FreeTypeFont:
    if not FOLIO_AVAILABLE_FONTS:
        return ImageFont.load_default()
    path = FOLIO_AVAILABLE_FONTS[variant % len(FOLIO_AVAILABLE_FONTS)]
    return _load_font(path, size)


def _lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    output: list[str] = []
    for paragraph in text.replace("\r", "").split("\n"):
        if not paragraph:
            output.append("")
            continue
        current: list[str] = []
        current_width = 0.0
        for character in paragraph:
            advance = font.getlength(character)
            if current and current_width + advance > width:
                output.append("".join(current).rstrip())
                first = character.lstrip() if character.isspace() else character
                current = list(first)
                current_width = font.getlength(first) if first else 0.0
            else:
                current.append(character)
                current_width += advance
        if current:
            output.append("".join(current).rstrip())
    return output


def render_folio(
    text: str,
    *,
    config: FolioRenderConfig,
    variant: int,
    augment: bool | None = None,
) -> torch.Tensor:
    """Render a plain writing field; returned value is ink in [0, 1]."""

    use_augmentation = config.augment if augment is None else bool(augment)
    rng = random.Random(variant)
    image = Image.new("L", (config.width, config.height), 255)
    draw = ImageDraw.Draw(image)
    available_width = config.width - config.margin * 2
    available_height = config.height - config.margin * 2
    selected_font: ImageFont.FreeTypeFont | None = None
    selected_lines: list[str] = []
    selected_spacing = 0
    for size in range(config.font_size, config.minimum_font_size - 1, -1):
        candidate_font = _font(variant, size)
        candidate_lines = _lines(draw, text, candidate_font, available_width)
        spacing = max(3, size // 4)
        if len(candidate_lines) * (size + spacing) <= available_height:
            selected_font = candidate_font
            selected_lines = candidate_lines
            selected_spacing = spacing
            break
    if selected_font is None:
        selected_font = _font(variant, config.minimum_font_size)
        selected_spacing = max(3, config.minimum_font_size // 4)
        selected_lines = _lines(draw, text, selected_font, available_width)
        maximum_lines = max(1, available_height // (config.minimum_font_size + selected_spacing))
        selected_lines = selected_lines[:maximum_lines]

    y = config.margin + rng.randint(-2, 2) if use_augmentation else config.margin
    x = config.margin + rng.randint(-2, 3) if use_augmentation else config.margin
    for line in selected_lines:
        draw.text((x, y), line, font=selected_font, fill=0)
        y += selected_font.size + selected_spacing

    if use_augmentation:
        if rng.random() < 0.40:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.76, 1.24))
        if rng.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.55)))
        if rng.random() < 0.20:
            image = image.rotate(rng.uniform(-0.35, 0.35), resample=Image.Resampling.BICUBIC, fillcolor=255)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if use_augmentation and rng.random() < 0.35:
        noise = np.random.default_rng(variant).normal(0.0, rng.uniform(0.002, 0.018), array.shape)
        array = np.clip(array + noise, 0.0, 1.0)
    return torch.from_numpy(1.0 - array.astype(np.float32))[None]


def split_folio_text(text: str, *, config: FolioRenderConfig, variant: int) -> list[str]:
    """Split by visual fit, without requiring character or word tokenization."""

    text = text.strip()
    if not text:
        return [""]
    image = Image.new("L", (config.width, config.height), 255)
    draw = ImageDraw.Draw(image)
    font = _font(variant, config.minimum_font_size)
    spacing = max(3, config.minimum_font_size // 4)
    lines = _lines(draw, text, font, config.width - config.margin * 2)
    lines_per_page = max(1, (config.height - config.margin * 2) // (font.size + spacing))
    return [
        "\n".join(lines[offset : offset + lines_per_page])
        for offset in range(0, len(lines), lines_per_page)
    ]


def render_folio_pages(
    text: str,
    *,
    config: FolioRenderConfig,
    variant: int,
) -> list[torch.Tensor]:
    page_config = replace(config, font_size=config.minimum_font_size, augment=False)
    return [
        render_folio(page, config=page_config, variant=variant + index, augment=False)
        for index, page in enumerate(split_folio_text(text, config=config, variant=variant))
    ]


def folio_tensor_to_image(ink: torch.Tensor) -> Image.Image:
    if ink.ndim == 4:
        if ink.shape[0] != 1:
            raise ValueError("only one folio can be visualized at once")
        ink = ink[0]
    if ink.ndim != 3 or ink.shape[0] != 1:
        raise ValueError("expected [1, height, width] ink tensor")
    array = ((1.0 - ink[0].detach().float().cpu().clamp(0, 1).numpy()) * 255).round().astype(np.uint8)
    return Image.fromarray(array, "L")


def load_teacher_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "folio-semantic-teacher-v1":
        raise ValueError("unsupported folio teacher cache")
    if not isinstance(payload.get("documents"), list):
        raise ValueError("teacher cache has no document manifest")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("teacher cache has invalid embeddings")
    if len(payload["documents"]) != embeddings.shape[0]:
        raise ValueError("teacher documents and embeddings are misaligned")
    return payload


def semantic_residual_fields(teacher_cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the corpus-common direction before visual distillation."""

    embeddings = teacher_cache["embeddings"].float()
    mean = teacher_cache.get("embedding_mean")
    if not isinstance(mean, torch.Tensor) or mean.shape != (embeddings.shape[1],):
        mean = embeddings.mean(dim=0)
    residuals = torch.nn.functional.normalize(embeddings - mean.float(), dim=-1)
    return residuals, mean.float()


class FolioSemanticDataset(Dataset):
    """Expose pixels and continuous teacher fields, never linguistic IDs."""

    def __init__(
        self,
        teacher_cache: dict[str, Any],
        *,
        render_config: FolioRenderConfig,
        split: str,
        validation_fraction: float = 0.05,
        seed: int = 0,
        length: int | None = None,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        documents = teacher_cache["documents"]
        selected_indices = []
        for index, document in enumerate(documents):
            is_validation = stable_fraction(str(document["record_identifier"])) < validation_fraction
            if split == "all" or (split == "validation" and is_validation) or (split == "train" and not is_validation):
                selected_indices.append(index)
        if not selected_indices:
            raise ValueError(f"teacher cache has no documents for split={split}")
        self.documents: Sequence[dict[str, Any]] = documents
        self.embeddings, self.embedding_mean = semantic_residual_fields(teacher_cache)
        self.indices = selected_indices
        self.render_config = render_config
        self.seed = int(seed)
        self.length = int(length) if length is not None else len(selected_indices)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        selected = self.indices[index % len(self.indices)] if self.length <= len(self.indices) else rng.choice(self.indices)
        document = self.documents[selected]
        first_variant = rng.randrange(2**31)
        second_variant = rng.randrange(2**31)
        return {
            "view_a": render_folio(
                str(document["text"]),
                config=self.render_config,
                variant=first_variant,
            ),
            "view_b": render_folio(
                str(document["text"]),
                config=self.render_config,
                variant=second_variant,
            ),
            "teacher": self.embeddings[selected],
            "metadata": {
                "document_index": selected,
                "record_identifier": document["record_identifier"],
                "kind": document["kind"],
                "language": document["language"],
            },
        }


def folio_semantic_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "view_a": torch.stack([item["view_a"] for item in batch]),
        "view_b": torch.stack([item["view_b"] for item in batch]),
        "teacher": torch.stack([item["teacher"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }

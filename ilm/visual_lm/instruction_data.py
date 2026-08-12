from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset

from .dataset import pil_to_tensor


FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/arphic-bkai00mp/bkai00mp.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
)

PALETTES = (
    ("#fbfaf6", "#171717", "#8a5a2b"),
    ("#ffffff", "#111827", "#1d4ed8"),
    ("#f7f1df", "#28231d", "#8b1e1e"),
    ("#f3f4f6", "#202124", "#30506d"),
)


@dataclass(frozen=True)
class VisualInstructionRecord:
    identifier: str
    instruction: str
    context: str
    response: str
    language: str
    source: str

    @property
    def prompt(self) -> str:
        if self.context:
            return f"{self.instruction}\n\n{self.context}"
        return self.instruction


@dataclass(frozen=True)
class InstructionRenderConfig:
    image_size: int = 384
    margin: int = 24
    base_font_size: int = 20
    minimum_font_size: int = 13
    augment: bool = True

    def __post_init__(self) -> None:
        if self.image_size < 256:
            raise ValueError("instruction pages require image_size >= 256")


def _stable_fraction(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    if Path(path).exists():
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _font_for_variant(variant: int, size: int) -> ImageFont.FreeTypeFont:
    available = [path for path in FONT_PATHS if Path(path).exists()]
    if not available:
        return ImageFont.load_default()
    return _load_font(available[variant % len(available)], size)


def _visual_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.replace("\r", "").split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and draw.textlength(candidate, font=font) > width:
                lines.append(current.rstrip())
                current = character.lstrip() if character.isspace() else character
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    width: int,
    height: int,
    config: InstructionRenderConfig,
    variant: int,
) -> tuple[ImageFont.FreeTypeFont, list[str], int, bool]:
    for size in range(config.base_font_size, config.minimum_font_size - 1, -1):
        font = _font_for_variant(variant, size)
        spacing = max(3, size // 4)
        lines = _visual_lines(draw, text, font, width)
        if len(lines) * (size + spacing) <= height:
            return font, lines, spacing, False

    font = _font_for_variant(variant, config.minimum_font_size)
    spacing = max(3, config.minimum_font_size // 4)
    lines = _visual_lines(draw, text, font, width)
    maximum_lines = max(1, height // (config.minimum_font_size + spacing))
    truncated = len(lines) > maximum_lines
    lines = lines[:maximum_lines]
    if truncated and lines:
        ellipsis = "……"
        while lines[-1] and draw.textlength(lines[-1] + ellipsis, font=font) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += ellipsis
    return font, lines, spacing, truncated


def render_instruction_page(
    text: str,
    *,
    role: str,
    language: str,
    config: InstructionRenderConfig,
    variant: int,
    page_index: int = 0,
    page_count: int = 1,
) -> tuple[Image.Image, dict[str, Any]]:
    rng = random.Random(variant)
    background, ink, accent = PALETTES[variant % len(PALETTES)]
    image = Image.new("RGB", (config.image_size, config.image_size), background)
    draw = ImageDraw.Draw(image)
    margin = config.margin
    header_height = max(42, config.image_size // 9)

    draw.rectangle((0, 0, config.image_size, header_height), fill=accent)
    role_label = {"prompt": "问 / PROMPT", "answer": "答 / ANSWER"}.get(role, role.upper())
    label_font = _font_for_variant(variant + 3, max(14, config.image_size // 24))
    draw.text((margin, 10), role_label, font=label_font, fill="#ffffff")
    page_label = f"{page_index + 1}/{page_count}" if page_count > 1 else language.upper()
    page_width = draw.textlength(page_label, font=label_font)
    draw.text((config.image_size - margin - page_width, 10), page_label, font=label_font, fill="#ffffff")

    body_x = margin
    body_y = header_height + margin
    body_width = config.image_size - margin * 2
    body_height = config.image_size - body_y - margin
    body_font, lines, spacing, truncated = _fit_text(
        draw,
        text,
        width=body_width,
        height=body_height,
        config=config,
        variant=variant,
    )
    cursor_y = body_y
    for line in lines:
        draw.text((body_x, cursor_y), line, font=body_font, fill=ink)
        cursor_y += body_font.size + spacing

    if config.augment:
        if rng.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))
        if rng.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.05, 0.35)))
        if rng.random() < 0.20:
            array = np.asarray(image, dtype=np.int16)
            noise = np.random.default_rng(variant).normal(0.0, rng.uniform(0.4, 1.8), array.shape)
            image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")

    return image, {
        "truncated": truncated,
        "font_size": body_font.size,
        "line_count": len(lines),
        "language": language,
        "role": role,
    }


def _convert_script(text: str, conversion: str) -> str:
    if conversion == "original":
        return text
    try:
        from opencc import OpenCC

        return OpenCC(conversion).convert(text)
    except Exception:
        return text


def load_alpaca_records(
    path: str | Path,
    *,
    language: str,
    source: str | None = None,
    max_prompt_chars: int | None = None,
    max_response_chars: int | None = None,
    limit: int | None = None,
) -> list[VisualInstructionRecord]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data", data.get("instances", []))
    if not isinstance(data, list):
        raise ValueError(f"Unsupported instruction dataset structure in {path}")
    records: list[VisualInstructionRecord] = []
    source_name = source or path.stem
    for index, item in enumerate(data):
        instruction = str(item.get("instruction", item.get("prompt", ""))).strip()
        context = str(item.get("input", item.get("context", ""))).strip()
        response = str(item.get("output", item.get("response", ""))).strip()
        if not instruction or not response:
            continue
        prompt_length = len(instruction) + len(context)
        if max_prompt_chars is not None and prompt_length > max_prompt_chars:
            continue
        if max_response_chars is not None and len(response) > max_response_chars:
            continue
        records.append(
            VisualInstructionRecord(
                identifier=f"{source_name}:{index}",
                instruction=instruction,
                context=context,
                response=response,
                language=language,
                source=source_name,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


class VisualInstructionDataset(Dataset):
    """Render instruction/response strings into paired writing images.

    Strings are used only by this offline raster data boundary. Returned model
    tensors contain pixels and metadata only; there are no token IDs.
    """

    def __init__(
        self,
        records: Sequence[VisualInstructionRecord],
        *,
        render_config: InstructionRenderConfig,
        length: int | None = None,
        seed: int = 0,
        split: str = "train",
        validation_fraction: float = 0.02,
        script_augmentation: bool = True,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        selected: list[VisualInstructionRecord] = []
        for record in records:
            is_validation = _stable_fraction(record.identifier) < validation_fraction
            if split == "all" or (split == "validation" and is_validation) or (split == "train" and not is_validation):
                selected.append(record)
        if not selected and records:
            ranked = sorted(records, key=lambda record: _stable_fraction(record.identifier))
            if split == "validation":
                selected = [ranked[0]]
            elif split == "train":
                selected = [ranked[-1]]
        if not selected:
            raise ValueError(f"No records selected for split={split!r}")
        self.records = selected
        self.cfg = render_config
        self.length = int(length) if length is not None else len(selected)
        self.seed = int(seed)
        self.script_augmentation = bool(script_augmentation)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 1_000_003)
        record = self.records[index % len(self.records)] if self.length <= len(self.records) else rng.choice(self.records)
        variant = rng.randrange(2**31)
        conversion = "original"
        if self.script_augmentation and record.language.startswith("zh"):
            conversion = rng.choices(("original", "s2t", "t2s"), weights=(0.50, 0.25, 0.25), k=1)[0]
        prompt_text = _convert_script(record.prompt, conversion)
        response_text = _convert_script(record.response, conversion)
        prompt, prompt_meta = render_instruction_page(
            prompt_text,
            role="prompt",
            language=record.language,
            config=self.cfg,
            variant=variant,
        )
        target, target_meta = render_instruction_page(
            response_text,
            role="answer",
            language=record.language,
            config=self.cfg,
            variant=variant + 17,
        )
        return {
            "prompt": pil_to_tensor(prompt),
            "target": pil_to_tensor(target),
            "metadata": {
                "id": record.identifier,
                "source": record.source,
                "language": record.language,
                "script_conversion": conversion,
                "prompt_chars": len(prompt_text),
                "response_chars": len(response_text),
                "prompt_truncated": prompt_meta["truncated"],
                "target_truncated": target_meta["truncated"],
            },
        }


class MixedVisualDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        *,
        weights: Sequence[float] | None = None,
        length: int,
        seed: int = 0,
    ):
        if not datasets:
            raise ValueError("MixedVisualDataset requires at least one dataset")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("MixedVisualDataset cannot contain an empty dataset")
        self.datasets = list(datasets)
        self.weights = list(weights or [1.0] * len(datasets))
        if len(self.weights) != len(self.datasets) or any(weight <= 0 for weight in self.weights):
            raise ValueError("weights must be positive and match datasets")
        self.length = int(length)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 104_729)
        dataset = rng.choices(self.datasets, weights=self.weights, k=1)[0]
        return dataset[rng.randrange(len(dataset))]

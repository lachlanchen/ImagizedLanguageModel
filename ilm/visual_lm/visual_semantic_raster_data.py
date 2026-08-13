from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset


V32_ARCHITECTURE = "visual-semantic-raster-transducer-v32"
V32_PROMPT_PATCH = 16
V32_MAX_PROMPT_PATCHES = 192
V32_ANSWER_CELL = 24
V32_MAX_ANSWER_CELLS = 32
V32_SPLITS = ("train", "development", "sealed")

V32_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
)
V32_DEVELOPMENT_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
V32_SEALED_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)

V32_STUDENT_KEYS = (
    "prompt_pixels",
    "prompt_mask",
    "answer_cells",
    "answer_mask",
    "stop_targets",
    "stop_mask",
)


@dataclass(frozen=True)
class VisualRasterRecord:
    identifier: str
    prompt: str
    answer: str
    language: str
    source: str
    rights: str


@dataclass(frozen=True)
class VisualTextRecord:
    identifier: str
    text: str
    language: str
    source: str
    rights: str


@dataclass(frozen=True)
class VisualRasterRenderConfig:
    prompt_patch_size: int = V32_PROMPT_PATCH
    maximum_prompt_patches: int = V32_MAX_PROMPT_PATCHES
    answer_cell_size: int = V32_ANSWER_CELL
    maximum_answer_cells: int = V32_MAX_ANSWER_CELLS
    prompt_minimum_font_size: int = 11
    prompt_maximum_font_size: int = 14
    answer_minimum_font_size: int = 18
    answer_maximum_font_size: int = 21
    nonaligned_origin_probability: float = 0.25
    augment: bool = True

    def __post_init__(self) -> None:
        if self.prompt_patch_size != V32_PROMPT_PATCH:
            raise ValueError("V32 fixes 16-pixel prompt patches")
        if not 8 <= self.maximum_prompt_patches <= V32_MAX_PROMPT_PATCHES:
            raise ValueError("V32 prompt patch count must be in [8,192]")
        if self.answer_cell_size != V32_ANSWER_CELL:
            raise ValueError("V32 fixes 24-pixel answer cells")
        if not 1 <= self.maximum_answer_cells <= V32_MAX_ANSWER_CELLS:
            raise ValueError("V32 answer length must be in [1,32]")
        if not 8 <= self.prompt_minimum_font_size <= self.prompt_maximum_font_size <= 16:
            raise ValueError("V32 prompt font range is invalid")
        if not 12 <= self.answer_minimum_font_size <= self.answer_maximum_font_size <= 23:
            raise ValueError("V32 answer font range is invalid")
        if not 0.0 <= self.nonaligned_origin_probability <= 1.0:
            raise ValueError("V32 origin probability must be in [0,1]")

    @property
    def prompt_width(self) -> int:
        return self.prompt_patch_size * self.maximum_prompt_patches


_WHITESPACE = re.compile(r"\s+")


def normalize_visible_text(text: str) -> str:
    """Collapse layout whitespace without normalizing historical codepoints."""

    cleaned: list[str] = []
    for character in str(text).replace("\r", " ").replace("\n", " "):
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs"} and not character.isspace():
            continue
        cleaned.append(character)
    return _WHITESPACE.sub(" ", "".join(cleaned)).strip()


def _stable_fraction(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def visual_raster_partition(identifier: str, *, stream: str) -> str:
    fraction = _stable_fraction(f"{stream}:{identifier}")
    if stream == "public-domain":
        if fraction < 0.96:
            return "train"
        if fraction < 0.98:
            return "development"
        return "sealed"
    if stream == "instruction":
        if fraction < 0.94:
            return "train"
        if fraction < 0.97:
            return "development"
        return "sealed"
    raise ValueError("V32 partition stream must be public-domain or instruction")


def _font_paths(split: str) -> tuple[str, ...]:
    if split == "train":
        paths = V32_TRAIN_FONTS
    elif split == "development":
        paths = V32_DEVELOPMENT_FONTS
    elif split == "sealed":
        paths = V32_SEALED_FONTS
    else:
        raise ValueError(f"unknown V32 split: {split}")
    available = tuple(path for path in paths if Path(path).exists())
    if not available:
        raise FileNotFoundError(f"no V32 fonts are available for split={split!r}")
    return available


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return ImageFont.truetype(path, size=size)


def _centered_text_position(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
    height: int,
    *,
    shift_x: int = 0,
    shift_y: int = 0,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    glyph_width = right - left
    glyph_height = bottom - top
    x = (width - glyph_width) // 2 - left + shift_x
    y = (height - glyph_height) // 2 - top + shift_y
    return x, y


def render_prompt_strip(
    text: str,
    *,
    config: VisualRasterRenderConfig,
    font_path: str,
    variant: int,
    force_origin: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    normalized = normalize_visible_text(text)
    if not normalized:
        raise ValueError("V32 prompt text is empty")
    rng = random.Random(int(variant))
    font_size = rng.randint(
        config.prompt_minimum_font_size,
        config.prompt_maximum_font_size,
    )
    font = _load_font(font_path, font_size)
    if force_origin is None:
        if rng.random() < config.nonaligned_origin_probability:
            origin = rng.randint(1, config.prompt_patch_size - 1)
        else:
            origin = 0
    else:
        origin = int(force_origin)
    if not 0 <= origin < config.prompt_patch_size:
        raise ValueError("V32 prompt origin must be inside the first patch")

    background = 255
    ink = 0
    image = Image.new(
        "L",
        (config.prompt_width, config.prompt_patch_size),
        color=background,
    )
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), normalized, font=font)
    text_width = right - left
    if origin + text_width >= config.prompt_width:
        raise ValueError("V32 prompt does not fit the fixed visual strip")
    text_height = bottom - top
    y = (config.prompt_patch_size - text_height) // 2 - top
    draw.text((origin - left, y), normalized, font=font, fill=ink)

    if config.augment:
        if rng.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.35)))
        array = np.asarray(image, dtype=np.float32) / 255.0
        contrast = rng.uniform(0.92, 1.08)
        array = 0.5 + (array - 0.5) * contrast
        if rng.random() < 0.15:
            noise = np.random.default_rng(variant).normal(
                0.0,
                rng.uniform(0.002, 0.015),
                array.shape,
            )
            array = array + noise
        array = np.clip(array, 0.0, 1.0)
    else:
        array = np.asarray(image, dtype=np.float32) / 255.0

    ink_array = 1.0 - array
    patch_ink = ink_array.reshape(
        config.prompt_patch_size,
        config.maximum_prompt_patches,
        config.prompt_patch_size,
    ).max(axis=(0, 2))
    mask = torch.from_numpy((patch_ink > 1.0 / 255.0).astype(np.float32))
    if not bool(mask.any()):
        raise ValueError("V32 rendered prompt contains no visible ink")
    pixels = torch.from_numpy(array.copy()).unsqueeze(0).repeat(3, 1, 1)
    return pixels, mask, {
        "font_path": font_path,
        "font_size": font_size,
        "origin": origin,
        "normalized_length": len(normalized),
        "occupied_patches": int(mask.sum().item()),
    }


def _render_answer_cell(
    character: str,
    *,
    cell_size: int,
    font: ImageFont.FreeTypeFont,
    shift_x: int,
    shift_y: int,
) -> torch.Tensor:
    image = Image.new("L", (cell_size, cell_size), color=255)
    if character != " ":
        draw = ImageDraw.Draw(image)
        x, y = _centered_text_position(
            draw,
            character,
            font,
            cell_size,
            cell_size,
            shift_x=shift_x,
            shift_y=shift_y,
        )
        draw.text((x, y), character, font=font, fill=0)
    intensity = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()) / 255.0
    return (1.0 - intensity).unsqueeze(0)


def render_answer_cells(
    text: str,
    *,
    config: VisualRasterRenderConfig,
    font_path: str,
    variant: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    normalized = normalize_visible_text(text)
    if not normalized:
        raise ValueError("V32 answer text is empty")
    if len(normalized) > config.maximum_answer_cells:
        raise ValueError("V32 answer exceeds the maximum raster-cell count")
    rng = random.Random(int(variant))
    font_size = rng.randint(
        config.answer_minimum_font_size,
        config.answer_maximum_font_size,
    )
    font = _load_font(font_path, font_size)
    cells = torch.zeros(
        config.maximum_answer_cells,
        1,
        config.answer_cell_size,
        config.answer_cell_size,
        dtype=torch.float32,
    )
    for index, character in enumerate(normalized):
        shift_x = rng.randint(-1, 1) if config.augment else 0
        shift_y = rng.randint(-1, 1) if config.augment else 0
        cells[index] = _render_answer_cell(
            character,
            cell_size=config.answer_cell_size,
            font=font,
            shift_x=shift_x,
            shift_y=shift_y,
        )
    answer_mask = torch.zeros(config.maximum_answer_cells, dtype=torch.float32)
    answer_mask[: len(normalized)] = 1.0
    stop_targets = torch.zeros(config.maximum_answer_cells + 1, dtype=torch.float32)
    stop_targets[len(normalized)] = 1.0
    stop_mask = torch.zeros(config.maximum_answer_cells + 1, dtype=torch.float32)
    stop_mask[: len(normalized) + 1] = 1.0
    return cells, answer_mask, stop_targets, stop_mask, {
        "font_path": font_path,
        "font_size": font_size,
        "normalized_length": len(normalized),
    }


def render_visual_raster_record(
    record: VisualRasterRecord,
    *,
    split: str,
    config: VisualRasterRenderConfig,
    variant: int,
) -> dict[str, Any]:
    fonts = _font_paths(split)
    rng = random.Random(int(variant))
    font_path = fonts[rng.randrange(len(fonts))]
    prompt_pixels, prompt_mask, prompt_meta = render_prompt_strip(
        record.prompt,
        config=config,
        font_path=font_path,
        variant=variant,
    )
    answer_cells, answer_mask, stop_targets, stop_mask, answer_meta = (
        render_answer_cells(
            record.answer,
            config=config,
            font_path=font_path,
            variant=variant + 17,
        )
    )
    return {
        "prompt_pixels": prompt_pixels,
        "prompt_mask": prompt_mask,
        "answer_cells": answer_cells,
        "answer_mask": answer_mask,
        "stop_targets": stop_targets,
        "stop_mask": stop_mask,
        "metadata": {
            "identifier": record.identifier,
            "language": record.language,
            "source": record.source,
            "rights": record.rights,
            "prompt": prompt_meta,
            "answer": answer_meta,
        },
    }


def load_visual_raster_instructions(
    path: str | Path,
    *,
    maximum_prompt_characters: int = 160,
    maximum_answer_cells: int = V32_MAX_ANSWER_CELLS,
) -> list[VisualRasterRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("data", payload.get("instances", []))
    if not isinstance(payload, list):
        raise ValueError(f"unsupported V32 instruction data in {path}")
    records: list[VisualRasterRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            continue
        instruction = normalize_visible_text(str(item.get("instruction", "")))
        context = normalize_visible_text(str(item.get("input", "")))
        answer = normalize_visible_text(str(item.get("output", "")))
        if not instruction or not answer:
            continue
        prompt = f"问：{instruction}"
        if context:
            prompt += f" 条件：{context}"
        if len(prompt) > maximum_prompt_characters or len(answer) > maximum_answer_cells:
            continue
        records.append(
            VisualRasterRecord(
                identifier=f"alpaca-zh:{index}",
                prompt=prompt,
                answer=answer,
                language="zh",
                source="GPT-4-LLM alpaca_gpt4_data_zh",
                rights="CC BY-NC 4.0; research use only",
            )
        )
    if not records:
        raise ValueError("V32 selected no instruction records")
    return records


def load_visual_text_records(path: str | Path) -> list[VisualTextRecord]:
    path = Path(path)
    records: list[VisualTextRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            text = normalize_visible_text(str(item.get("text", "")))
            if len(text) < 10:
                continue
            records.append(
                VisualTextRecord(
                    identifier=str(item["id"]),
                    text=text,
                    language=str(item.get("language", "zh")),
                    source=str(item.get("source", path.name)),
                    rights=str(item.get("rights", "unrecorded")),
                )
            )
    if not records:
        raise ValueError("V32 selected no public-domain visual text records")
    return records


def load_visual_raster_paraphrases(
    path: str | Path,
    instruction_records: Sequence[VisualRasterRecord],
) -> list[VisualRasterRecord]:
    by_identifier = {record.identifier: record for record in instruction_records}
    records: list[VisualRasterRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_identifier = str(item.get("identifier", ""))
            try:
                source_index = int(raw_identifier.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                continue
            source = by_identifier.get(f"alpaca-zh:{source_index}")
            paraphrase = normalize_visible_text(str(item.get("paraphrase", "")))
            if source is None or not paraphrase:
                continue
            records.append(
                VisualRasterRecord(
                    identifier=f"paraphrase:{source.identifier}",
                    prompt=f"问：{paraphrase}",
                    answer=source.answer,
                    language=source.language,
                    source="fixed Qwen paraphrase holdout",
                    rights=source.rights,
                )
            )
    return records


class VisualRasterInstructionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VisualRasterRecord],
        *,
        split: str,
        render_config: VisualRasterRenderConfig,
        seed: int,
        length: int | None = None,
        include_all_records: bool = False,
    ) -> None:
        if split not in V32_SPLITS:
            raise ValueError(f"unknown V32 split: {split}")
        selected = [
            record
            for record in records
            if include_all_records
            or visual_raster_partition(record.identifier, stream="instruction") == split
        ]
        if not selected:
            raise ValueError(f"V32 instruction split {split!r} is empty")
        self.records = tuple(selected)
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = len(selected) if length is None else int(length)
        if self.length < 1:
            raise ValueError("V32 instruction dataset length must be positive")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 1_000_003)
        record = (
            self.records[index]
            if self.length == len(self.records)
            else self.records[rng.randrange(len(self.records))]
        )
        return render_visual_raster_record(
            record,
            split=self.split,
            config=self.render_config,
            variant=rng.randrange(2**31),
        )


class VisualRasterContinuationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VisualTextRecord],
        *,
        split: str,
        render_config: VisualRasterRenderConfig,
        seed: int,
        length: int,
        minimum_prompt_cells: int = 8,
    ) -> None:
        if split not in V32_SPLITS:
            raise ValueError(f"unknown V32 split: {split}")
        selected = [
            record
            for record in records
            if visual_raster_partition(record.identifier, stream="public-domain") == split
            and len(record.text) >= minimum_prompt_cells + 1
        ]
        if not selected or length < 1:
            raise ValueError(f"V32 continuation split {split!r} is empty")
        self.records = tuple(selected)
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = int(length)
        self.minimum_prompt_cells = int(minimum_prompt_cells)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 1_000_033)
        for _ in range(256):
            record = self.records[rng.randrange(len(self.records))]
            maximum_prompt = min(160, len(record.text) - 1)
            if maximum_prompt < self.minimum_prompt_cells:
                continue
            prompt_length = rng.randint(self.minimum_prompt_cells, maximum_prompt)
            maximum_answer = min(
                self.render_config.maximum_answer_cells,
                len(record.text) - prompt_length,
            )
            if maximum_answer < 1:
                continue
            answer_length = rng.randint(1, maximum_answer)
            maximum_start = len(record.text) - prompt_length - answer_length
            start = rng.randint(0, maximum_start)
            prompt = record.text[start : start + prompt_length]
            answer = record.text[
                start + prompt_length : start + prompt_length + answer_length
            ]
            sample = render_visual_raster_record(
                VisualRasterRecord(
                    identifier=f"{record.identifier}:{start}:{prompt_length}:{answer_length}",
                    prompt=prompt,
                    answer=answer,
                    language=record.language,
                    source=record.source,
                    rights=record.rights,
                ),
                split=self.split,
                config=self.render_config,
                variant=rng.randrange(2**31),
            )
            sample["metadata"]["continuation"] = {
                "source_identifier": record.identifier,
                "offset": start,
                "prompt_cells": prompt_length,
                "answer_cells": answer_length,
            }
            return sample
        raise RuntimeError("V32 could not render a valid continuation sample")


def visual_semantic_raster_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V32 batch")
    return {
        key: torch.stack([item[key] for item in batch]) for key in V32_STUDENT_KEYS
    } | {"metadata": [item.get("metadata", {}) for item in batch]}


def visual_semantic_raster_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V32_STUDENT_KEYS}
    for key, value in student.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V32 student value {key!r} must be a floating tensor")
    prompt = student["prompt_pixels"]
    if prompt.ndim != 4 or prompt.shape[1] != 3 or prompt.shape[2] != V32_PROMPT_PATCH:
        raise ValueError("V32 prompt pixels must be [B,3,16,W]")
    if prompt.shape[-1] % V32_PROMPT_PATCH:
        raise ValueError("V32 prompt width must divide into 16-pixel patches")
    if student["prompt_mask"].shape != (
        prompt.shape[0],
        prompt.shape[-1] // V32_PROMPT_PATCH,
    ):
        raise ValueError("V32 prompt mask does not match prompt patches")
    answer = student["answer_cells"]
    if answer.ndim != 5 or tuple(answer.shape[-3:]) != (
        1,
        V32_ANSWER_CELL,
        V32_ANSWER_CELL,
    ):
        raise ValueError("V32 answer cells must end in [1,24,24]")
    if answer.shape[1] > V32_MAX_ANSWER_CELLS:
        raise ValueError("V32 answer cells exceed the fixed maximum")
    if student["answer_mask"].shape != answer.shape[:2]:
        raise ValueError("V32 answer mask does not match answer cells")
    stop_shape = (answer.shape[0], answer.shape[1] + 1)
    if student["stop_targets"].shape != stop_shape or student["stop_mask"].shape != stop_shape:
        raise ValueError("V32 stop tensors must include one post-answer position")
    return student


def visual_semantic_raster_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V32_ARCHITECTURE,
        "student_keys": list(V32_STUDENT_KEYS),
        "prompt_shape": [3, V32_PROMPT_PATCH, V32_PROMPT_PATCH * V32_MAX_PROMPT_PATCHES],
        "answer_shape": [V32_MAX_ANSWER_CELLS, 1, V32_ANSWER_CELL, V32_ANSWER_CELL],
        "primary_output_is_raster": True,
        "strings_exist_only_before_student_batch": True,
        "metadata_enters_student": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_one_hot_character_labels": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }


from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from fontTools.ttLib import TTFont
from PIL import ImageFont
from torch.utils.data import Dataset

from .visual_answer_trajectory import V39_ARCHITECTURE, V39_MAX_SEGMENTS
from .visual_semantic_distillation_data import (
    VisualSemanticDistillationRenderConfig,
    render_visual_semantic_distillation_strip,
    visual_semantic_distillation_stream_record_index,
)
from .visual_semantic_raster_data import normalize_visible_text, visual_raster_partition


V39_MAX_SEGMENT_UNITS = 48
V39_TARGET_ARCHITECTURE = "visual-answer-trajectory-target-bank-v39"

V39_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
V39_DEVELOPMENT_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
V39_HELD_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Black.ttc"
V39_SEALED_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc"

V39_PIXEL_KEYS = (
    "prompt_anchor_pixels",
    "prompt_anchor_mask",
    "prompt_view_pixels",
    "prompt_view_mask",
    "segment_anchor_pixels",
    "segment_anchor_mask",
    "segment_view_pixels",
    "segment_view_mask",
)

_STRONG_BREAKS = frozenset("。！？!?；;")
_SOFT_BREAKS = frozenset("，、：,:")
_CLOSERS = frozenset("”’」』）》】〕〉")


@dataclass(frozen=True)
class VisualAnswerTrajectoryRecord:
    identifier: str
    prompt: str
    answer: str
    segments: tuple[str, ...]
    language: str
    source: str
    rights: str

    def __post_init__(self) -> None:
        if not self.identifier or not self.prompt or not self.answer:
            raise ValueError("V39 record fields cannot be empty")
        if not 1 <= len(self.segments) <= V39_MAX_SEGMENTS:
            raise ValueError("V39 record must contain one to sixteen segments")
        if any(not segment for segment in self.segments):
            raise ValueError("V39 answer segments cannot be empty")


def _require_font(path: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return path


@lru_cache(maxsize=32)
def _geometry_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_require_font(path), size=size)


def _visible_unit_extents(
    path: str,
    size: int,
    units: set[str],
) -> dict[str, int]:
    if not units:
        raise ValueError("V39 cannot measure an empty visible-unit set")
    options = {"fontNumber": 0} if Path(path).suffix.lower() == ".ttc" else {}
    font = TTFont(_require_font(path), lazy=True, **options)
    try:
        units_per_em = int(font["head"].unitsPerEm)
        cmap = font.getBestCmap() or {}
        metrics = font["hmtx"].metrics
        missing_advance = int(metrics.get(".notdef", (units_per_em, 0))[0])

        def unit_advance(unit: str) -> int:
            total = 0
            for character in unit:
                glyph_name = cmap.get(ord(character))
                total += int(metrics.get(glyph_name, (missing_advance, 0))[0])
            return total

        advances = {unit: unit_advance(unit) for unit in units}
    finally:
        font.close()
    # One raster pixel per visible unit covers hinting and side-bearing rounding.
    return {
        unit: max(1, math.ceil(advance * size / units_per_em) + 1)
        for unit, advance in advances.items()
    }


def visual_answer_trajectory_fonts(split: str) -> tuple[str, ...]:
    if split == "train":
        candidates = V39_TRAIN_FONTS
    elif split == "development":
        candidates = (V39_DEVELOPMENT_FONT, V39_HELD_FONT)
    elif split == "sealed":
        candidates = (V39_SEALED_FONT,)
    else:
        raise ValueError(f"unknown V39 split: {split}")
    return tuple(_require_font(path) for path in candidates)


def _is_unit_continuation(character: str) -> bool:
    value = ord(character)
    return (
        unicodedata.category(character).startswith("M")
        or character == "\u200d"
        or 0xFE00 <= value <= 0xFE0F
        or 0xE0100 <= value <= 0xE01EF
    )


def _visible_units(text: str) -> list[str]:
    cleaned: list[str] = []
    for character in str(text).replace("\r\n", "\n").replace("\r", "\n"):
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs"} and character != "\n":
            continue
        if character in "\t\f\v":
            character = " "
        if _is_unit_continuation(character) and cleaned:
            cleaned[-1] += character
        else:
            cleaned.append(character)

    normalized: list[str] = []
    prior_space = False
    prior_break = False
    for unit in cleaned:
        if unit == "\n":
            if normalized and not prior_break:
                normalized.append(unit)
            prior_space = False
            prior_break = True
        elif unit.isspace():
            if normalized and not prior_space and not prior_break:
                normalized.append(" ")
            prior_space = True
        else:
            normalized.append(unit)
            prior_space = False
            prior_break = False
    while normalized and normalized[-1].isspace():
        normalized.pop()
    return normalized


def segment_visual_answer(
    text: str,
    *,
    maximum_units: int = V39_MAX_SEGMENT_UNITS,
    target_units: int = 36,
    minimum_break_units: int = 12,
) -> tuple[str, ...]:
    """Split visible writing into sentence-scale spans before rasterization."""

    if not 8 <= minimum_break_units <= target_units <= maximum_units:
        raise ValueError("V39 segment geometry is invalid")
    units = _visible_units(text)
    if not units:
        return ()
    output: list[str] = []
    buffer: list[str] = []
    strong: list[int] = []
    soft: list[int] = []

    def emit(position: int) -> None:
        piece = "".join(buffer[:position]).replace("\n", " ").strip()
        del buffer[:position]
        while buffer and (buffer[0].isspace() or buffer[0] == "\n"):
            del buffer[0]
        strong[:] = [value - position for value in strong if value > position]
        soft[:] = [value - position for value in soft if value > position]
        if piece:
            output.append(piece)

    for index, unit in enumerate(units):
        buffer.append(unit)
        position = len(buffer)
        if unit in _STRONG_BREAKS or unit == "\n":
            strong.append(position)
        elif unit in _CLOSERS and strong and strong[-1] == position - 1:
            strong[-1] = position
        elif unit in _SOFT_BREAKS or unit.isspace():
            soft.append(position)

        next_unit = units[index + 1] if index + 1 < len(units) else ""
        at_terminal_break = (
            (unit in _STRONG_BREAKS or unit == "\n" or unit in _CLOSERS)
            and next_unit not in _CLOSERS
        )
        if at_terminal_break and len(buffer) >= target_units:
            emit(len(buffer))
        elif len(buffer) >= maximum_units:
            candidates = [value for value in strong if value >= minimum_break_units]
            if not candidates:
                candidates = [value for value in soft if value >= minimum_break_units]
            emit(max(candidates) if candidates else maximum_units)

    if buffer:
        piece = "".join(buffer).replace("\n", " ").strip()
        if piece:
            output.append(piece)

    merged: list[str] = []
    for piece in output:
        units_in_piece = len(_visible_units(piece))
        prior_units = len(_visible_units(merged[-1])) if merged else 0
        if (
            merged
            and units_in_piece < minimum_break_units
            and prior_units + units_in_piece <= maximum_units
        ):
            merged[-1] += piece
        else:
            merged.append(piece)
    return tuple(merged)


def load_v39_instruction_records(
    path: str | Path,
    *,
    maximum_prompt_characters: int = 160,
    maximum_segments: int = V39_MAX_SEGMENTS,
) -> list[VisualAnswerTrajectoryRecord]:
    if maximum_prompt_characters < 8 or maximum_segments != V39_MAX_SEGMENTS:
        raise ValueError("V39 instruction limits are invalid")
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("data", payload.get("instances", []))
    if not isinstance(payload, list):
        raise ValueError(f"unsupported V39 instruction data in {path}")
    records: list[VisualAnswerTrajectoryRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            continue
        instruction = normalize_visible_text(str(item.get("instruction", "")))
        context = normalize_visible_text(str(item.get("input", "")))
        raw_answer = str(item.get("output", ""))
        answer = normalize_visible_text(raw_answer)
        segments = segment_visual_answer(raw_answer)
        if not instruction or not answer or not segments:
            continue
        prompt = f"问：{instruction}"
        if context:
            prompt += f" 条件：{context}"
        if len(prompt) > maximum_prompt_characters or len(segments) > maximum_segments:
            continue
        records.append(
            VisualAnswerTrajectoryRecord(
                identifier=f"alpaca-zh:{index}",
                prompt=prompt,
                answer=answer,
                segments=segments,
                language="zh",
                source="GPT-4-LLM alpaca_gpt4_data_zh",
                rights="CC BY-NC 4.0; research use only",
            )
        )
    if not records:
        raise ValueError("V39 selected no instruction records")
    return records


def visual_answer_trajectory_record_fits(
    record: VisualAnswerTrajectoryRecord,
    *,
    split: str,
    render_config: VisualSemanticDistillationRenderConfig,
    unit_extents: Mapping[str, Mapping[str, int]] | None = None,
) -> bool:
    fonts = visual_answer_trajectory_fonts(split)
    if split == "train":
        font_size = render_config.maximum_font_size
        origin = render_config.maximum_origin
    else:
        font_size = render_config.evaluation_font_size
        origin = 0
    if unit_extents is None:
        units = {
            unit
            for text in (record.prompt, *record.segments)
            for unit in _visible_units(text)
        }
        unit_extents = {
            font_path: _visible_unit_extents(font_path, font_size, units)
            for font_path in fonts
        }
    for font_path in fonts:
        face = _geometry_font(font_path, font_size)
        font_extents = unit_extents[font_path]
        for text in (record.prompt, *record.segments):
            visible_units = _visible_units(text)
            conservative_width = origin + sum(
                int(font_extents[unit]) for unit in visible_units
            )
            if conservative_width <= render_config.width:
                continue
            normalized = normalize_visible_text(text)
            left, _top, right, _bottom = face.getbbox(normalized)
            if origin + right - left > render_config.width:
                return False
    return True


def select_v39_instruction_records(
    records: Sequence[VisualAnswerTrajectoryRecord],
    *,
    split: str,
    render_config: VisualSemanticDistillationRenderConfig,
    include_all_records: bool = False,
) -> tuple[tuple[VisualAnswerTrajectoryRecord, ...], tuple[str, ...]]:
    if split not in {"train", "development", "sealed"}:
        raise ValueError(f"unknown V39 split: {split}")
    partitioned = [
        record
        for record in records
        if include_all_records
        or visual_raster_partition(record.identifier, stream="instruction") == split
    ]
    if not partitioned:
        raise ValueError(f"V39 instruction split {split!r} is empty")
    fonts = visual_answer_trajectory_fonts(split)
    font_size = (
        render_config.maximum_font_size
        if split == "train"
        else render_config.evaluation_font_size
    )
    units = {
        unit
        for record in partitioned
        for text in (record.prompt, *record.segments)
        for unit in _visible_units(text)
    }
    unit_extents = {
        font_path: _visible_unit_extents(font_path, font_size, units)
        for font_path in fonts
    }
    rejected = tuple(
        record.identifier
        for record in partitioned
        if not visual_answer_trajectory_record_fits(
            record,
            split=split,
            render_config=render_config,
            unit_extents=unit_extents,
        )
    )
    rejected_set = set(rejected)
    selected = tuple(
        record for record in partitioned if record.identifier not in rejected_set
    )
    if not selected:
        raise ValueError(f"V39 instruction split {split!r} is empty")
    return selected, rejected


@lru_cache(maxsize=4)
def _opencc(conversion: str):
    from opencc import OpenCC

    return OpenCC(conversion)


def convert_visual_script(text: str, conversion: str) -> str:
    if conversion == "original":
        return text
    if conversion not in {"s2t", "t2s"}:
        raise ValueError("V39 script conversion is invalid")
    return _opencc(conversion).convert(text)


def _render_view(
    text: str,
    *,
    config: VisualSemanticDistillationRenderConfig,
    font_path: str,
    rng: random.Random,
    variant: int,
    clean: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if clean:
        view_config = replace(config, augment=False)
        font_size = config.evaluation_font_size
        origin = 0
    else:
        view_config = config
        font_size = rng.randint(config.minimum_font_size, config.maximum_font_size)
        origin = None
    return render_visual_semantic_distillation_strip(
        text,
        config=view_config,
        font_path=font_path,
        font_size=font_size,
        variant=variant,
        force_origin=origin,
    )


def render_visual_answer_trajectory_record(
    record: VisualAnswerTrajectoryRecord,
    *,
    config: VisualSemanticDistillationRenderConfig,
    variant: int,
) -> dict[str, Any]:
    rng = random.Random(int(variant))
    fonts = list(visual_answer_trajectory_fonts("train"))
    rng.shuffle(fonts)
    if len(fonts) < 4 or len(set(fonts[:4])) != 4:
        raise RuntimeError("V39 requires four distinct training font paths")
    segment_index = rng.randrange(len(record.segments))
    conversion = rng.choices(
        ("original", "s2t", "t2s"),
        weights=(0.50, 0.25, 0.25),
        k=1,
    )[0]
    prompt_view = convert_visual_script(record.prompt, conversion)
    segment = record.segments[segment_index]
    segment_view = convert_visual_script(segment, conversion)
    specs = (
        ("prompt_anchor", record.prompt, fonts[0], True, 0),
        ("prompt_view", prompt_view, fonts[1], False, 17),
        ("segment_anchor", segment, fonts[2], True, 31),
        ("segment_view", segment_view, fonts[3], False, 47),
    )
    fields: dict[str, Any] = {"segment_index": segment_index}
    views: dict[str, Any] = {}
    for name, text, font_path, clean, offset in specs:
        pixels, mask, metadata = _render_view(
            text,
            config=config,
            font_path=font_path,
            rng=rng,
            variant=variant + offset,
            clean=clean,
        )
        fields[f"{name}_pixels"] = pixels
        fields[f"{name}_mask"] = mask
        views[name] = metadata | {"clean_anchor": clean}
    fields["metadata"] = {
        "identifier": record.identifier,
        "language": record.language,
        "source": record.source,
        "rights": record.rights,
        "segments": len(record.segments),
        "sampled_segment_index": segment_index,
        "script_conversion": conversion,
        "distinct_font_paths": len(
            {metadata["font_path"] for metadata in views.values()}
        ),
        "views": views,
    }
    return fields


class VisualAnswerTrajectoryDataset(Dataset):
    """Deterministic pixel-only V39 training stream."""

    def __init__(
        self,
        records: Sequence[VisualAnswerTrajectoryRecord],
        *,
        render_config: VisualSemanticDistillationRenderConfig,
        seed: int,
        length: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("V39 dataset requires records")
        self.records = tuple(records)
        self.render_config = render_config
        self.seed = int(seed)
        self.length = len(records) if length is None else int(length)
        if self.length < 1:
            raise ValueError("V39 dataset length must be positive")

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
        epoch, _ = divmod(index, len(self.records))
        variant_digest = hashlib.sha256(
            f"v39:{self.seed}:{epoch}:{index}".encode("utf-8")
        ).digest()
        variant = int.from_bytes(variant_digest[:8], "big") % (2**31)
        return render_visual_answer_trajectory_record(
            self.records[record_index],
            config=self.render_config,
            variant=variant,
        )


def visual_answer_trajectory_collate(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not items:
        raise ValueError("V39 cannot collate an empty batch")
    batch: dict[str, Any] = {
        key: torch.stack([item[key] for item in items]) for key in V39_PIXEL_KEYS
    }
    batch["segment_indices"] = torch.tensor(
        [int(item["segment_index"]) for item in items],
        dtype=torch.long,
    )
    batch["identifiers"] = tuple(
        str(item["metadata"]["identifier"]) for item in items
    )
    batch["metadata"] = tuple(item["metadata"] for item in items)
    return batch


def visual_answer_trajectory_tensor_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    expected = set(V39_PIXEL_KEYS) | {"segment_indices"}
    missing = expected.difference(batch)
    if missing:
        raise KeyError(f"V39 batch lacks tensor fields: {sorted(missing)}")
    result = {key: batch[key] for key in expected}
    if any(not isinstance(value, torch.Tensor) for value in result.values()):
        raise TypeError("V39 tensor boundary contains a non-tensor")
    if result["segment_indices"].dtype != torch.long:
        raise TypeError("V39 segment indices must be long geometry indices")
    if any(
        not torch.is_floating_point(value)
        for key, value in result.items()
        if key != "segment_indices"
    ):
        raise TypeError("V39 pixel paths must be floating point")
    return result


def canonical_v39_segment_lengths(
    record: VisualAnswerTrajectoryRecord,
    *,
    render_config: VisualSemanticDistillationRenderConfig,
    font_path: str = V39_DEVELOPMENT_FONT,
) -> tuple[float, ...]:
    lengths: list[float] = []
    clean = replace(render_config, augment=False)
    for index, segment in enumerate(record.segments):
        _pixels, mask, _metadata = render_visual_semantic_distillation_strip(
            segment,
            config=clean,
            font_path=_require_font(font_path),
            font_size=clean.evaluation_font_size,
            variant=index,
            force_origin=0,
        )
        lengths.append(float(mask.sum()))
    return tuple(lengths)


def visual_answer_trajectory_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V39_ARCHITECTURE,
        "maximum_segments": V39_MAX_SEGMENTS,
        "maximum_visible_units_per_segment": V39_MAX_SEGMENT_UNITS,
        "paired_visual_paths": 4,
        "distinct_training_font_paths_per_item": 4,
        "offline_script_conversions": ["original", "s2t", "t2s"],
        "tensor_keys": list(V39_PIXEL_KEYS) + ["segment_indices"],
        "deployable_keys": ["prompt_pixels", "prompt_mask"],
        "segment_index_role": "train-only output-position geometry",
        "uses_strings_after_tensor_boundary": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_runtime_script_converter": False,
        "target_answer_pixels_deployed": False,
    }


__all__ = [
    "V39_DEVELOPMENT_FONT",
    "V39_HELD_FONT",
    "V39_MAX_SEGMENT_UNITS",
    "V39_PIXEL_KEYS",
    "V39_SEALED_FONT",
    "V39_TARGET_ARCHITECTURE",
    "V39_TRAIN_FONTS",
    "VisualAnswerTrajectoryDataset",
    "VisualAnswerTrajectoryRecord",
    "canonical_v39_segment_lengths",
    "convert_visual_script",
    "load_v39_instruction_records",
    "render_visual_answer_trajectory_record",
    "segment_visual_answer",
    "select_v39_instruction_records",
    "visual_answer_trajectory_collate",
    "visual_answer_trajectory_data_boundary_receipt",
    "visual_answer_trajectory_fonts",
    "visual_answer_trajectory_record_fits",
    "visual_answer_trajectory_tensor_batch",
]

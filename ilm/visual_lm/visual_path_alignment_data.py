from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .visual_path_alignment import V38_ARCHITECTURE
from .visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_SEALED_FONT,
    V37_WIDTH,
    VisualSemanticDistillationRenderConfig,
    load_v37_instruction_records,
    render_visual_semantic_distillation_strip,
    visual_semantic_distillation_stream_record_index,
    visual_text_fits_v37,
)
from .visual_semantic_raster_data import VisualRasterRecord, visual_raster_partition


V38_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
V38_DEVELOPMENT_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
V38_HELD_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Black.ttc"
V38_SEALED_FONT = V37_SEALED_FONT

V38_PIXEL_KEYS = (
    "prompt_anchor_pixels",
    "prompt_anchor_mask",
    "prompt_view_pixels",
    "prompt_view_mask",
    "semantic_view_pixels",
    "semantic_view_mask",
    "answer_anchor_pixels",
    "answer_anchor_mask",
    "answer_view_pixels",
    "answer_view_mask",
)


@dataclass(frozen=True)
class VisualPathAlignmentParaphrase:
    identifier: str
    text: str
    source_prompt_sha256: str


def _require_font(path: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return path


def visual_path_alignment_fonts(split: str) -> tuple[str, ...]:
    if split == "train":
        candidates = V38_TRAIN_FONTS
    elif split == "development":
        candidates = (V38_DEVELOPMENT_FONT, V38_HELD_FONT)
    elif split == "sealed":
        candidates = (V38_SEALED_FONT,)
    else:
        raise ValueError(f"unknown V38 split: {split}")
    return tuple(_require_font(path) for path in candidates)


def load_v38_paraphrases(
    path: str | Path,
    records: Sequence[VisualRasterRecord],
) -> dict[str, VisualPathAlignmentParaphrase]:
    by_identifier = {record.identifier: record for record in records}
    result: dict[str, VisualPathAlignmentParaphrase] = {}
    seen_text: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("identifier", ""))
            text = str(row.get("paraphrase", "")).strip()
            if not identifier or identifier not in by_identifier or not text:
                raise ValueError(f"invalid V38 paraphrase row {line_number}")
            if identifier in result or text in seen_text:
                raise ValueError("V38 paraphrase identifiers and text must be unique")
            if any(
                row.get(field) != "pass"
                for field in (
                    "instruction_judge",
                    "constraint_adjudicator",
                    "adversarial_confirmation",
                )
            ):
                raise ValueError(f"V38 paraphrase row {line_number} failed consensus")
            expected = hashlib.sha256(
                by_identifier[identifier].prompt.encode("utf-8")
            ).hexdigest()
            source_digest = str(row.get("source_prompt_sha256", ""))
            if source_digest != expected:
                raise ValueError(f"V38 paraphrase source changed at row {line_number}")
            result[identifier] = VisualPathAlignmentParaphrase(
                identifier=identifier,
                text=text,
                source_prompt_sha256=source_digest,
            )
            seen_text.add(text)
    if not result:
        raise ValueError("V38 paraphrase manifest is empty")
    return result


def visual_path_alignment_record_fits(
    record: VisualRasterRecord,
    *,
    split: str,
    render_config: VisualSemanticDistillationRenderConfig,
    paraphrase: str | None = None,
) -> bool:
    fonts = visual_path_alignment_fonts(split)
    if split == "train":
        font_size = render_config.maximum_font_size
        origin = render_config.maximum_origin
    else:
        font_size = render_config.evaluation_font_size
        origin = 0
    texts = [record.prompt, record.answer]
    if paraphrase is not None:
        texts.append(paraphrase)
    return all(
        visual_text_fits_v37(
            text,
            config=render_config,
            font_path=font_path,
            font_size=font_size,
            origin=origin,
        )
        for font_path in fonts
        for text in texts
    )


def select_v38_instruction_records(
    records: Sequence[VisualRasterRecord],
    *,
    split: str,
    render_config: VisualSemanticDistillationRenderConfig,
    paraphrases: Mapping[str, VisualPathAlignmentParaphrase] | None = None,
    include_all_records: bool = False,
) -> tuple[tuple[VisualRasterRecord, ...], tuple[str, ...]]:
    if split not in {"train", "development", "sealed"}:
        raise ValueError(f"unknown V38 split: {split}")
    partitioned = [
        record
        for record in records
        if include_all_records
        or visual_raster_partition(record.identifier, stream="instruction") == split
    ]
    rejected = tuple(
        record.identifier
        for record in partitioned
        if not visual_path_alignment_record_fits(
            record,
            split=split,
            render_config=render_config,
            paraphrase=(
                paraphrases[record.identifier].text
                if paraphrases is not None and record.identifier in paraphrases
                else None
            ),
        )
    )
    rejected_set = set(rejected)
    selected = tuple(
        record for record in partitioned if record.identifier not in rejected_set
    )
    if not selected:
        raise ValueError(f"V38 instruction split {split!r} is empty")
    return selected, rejected


def _render_train_view(
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


def render_visual_path_alignment_record(
    record: VisualRasterRecord,
    *,
    config: VisualSemanticDistillationRenderConfig,
    variant: int,
    paraphrase: VisualPathAlignmentParaphrase | None = None,
) -> dict[str, Any]:
    rng = random.Random(int(variant))
    fonts = list(visual_path_alignment_fonts("train"))
    rng.shuffle(fonts)
    if len(fonts) < 5 or len(set(fonts[:5])) != 5:
        raise RuntimeError("V38 requires five distinct training font paths")
    semantic_text = paraphrase.text if paraphrase is not None else record.prompt
    specs = (
        ("prompt_anchor", record.prompt, fonts[0], True, 0),
        ("prompt_view", record.prompt, fonts[1], False, 17),
        ("semantic_view", semantic_text, fonts[2], False, 31),
        ("answer_anchor", record.answer, fonts[3], True, 47),
        ("answer_view", record.answer, fonts[4], False, 61),
    )
    fields: dict[str, Any] = {}
    views: dict[str, Any] = {}
    for name, text, font_path, clean, offset in specs:
        pixels, mask, metadata = _render_train_view(
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
        "semantic_view_kind": "paraphrase" if paraphrase is not None else "exact",
        "distinct_font_paths": len({views[name]["font_path"] for name in views}),
        "views": views,
    }
    return fields


class VisualPathAlignmentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[VisualRasterRecord],
        *,
        render_config: VisualSemanticDistillationRenderConfig,
        seed: int,
        length: int | None = None,
        paraphrases: Mapping[str, VisualPathAlignmentParaphrase] | None = None,
    ) -> None:
        self.records, self.rejected_identifiers = select_v38_instruction_records(
            records,
            split="train",
            render_config=render_config,
            paraphrases=paraphrases,
            include_all_records=True,
        )
        self.render_config = render_config
        self.seed = int(seed)
        self.length = len(self.records) if length is None else int(length)
        self.paraphrases = dict(paraphrases or {})
        if self.length < 1:
            raise ValueError("V38 dataset length must be positive")

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
        return render_visual_path_alignment_record(
            record,
            config=self.render_config,
            variant=variant,
            paraphrase=self.paraphrases.get(record.identifier),
        )


def visual_path_alignment_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V38 batch")
    return {
        key: torch.stack([item[key] for item in batch]) for key in V38_PIXEL_KEYS
    } | {"metadata": [item.get("metadata", {}) for item in batch]}


def visual_path_alignment_pixel_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    result = {key: batch[key] for key in V38_PIXEL_KEYS}
    batch_size = result["prompt_anchor_pixels"].shape[0]
    strip_shape = (batch_size, 3, V37_PATCH_SIZE, V37_WIDTH)
    mask_shape = (batch_size, V37_PATCHES)
    for key, value in result.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V38 pixel value {key!r} must be floating")
        expected = mask_shape if key.endswith("_mask") else strip_shape
        if value.shape != expected:
            raise ValueError(f"V38 {key} has an invalid shape")
    return result


def visual_path_alignment_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V38_ARCHITECTURE,
        "pixel_keys": list(V38_PIXEL_KEYS),
        "deployable_keys": ["prompt_pixels", "prompt_mask"],
        "prompt_shape": [3, V37_PATCH_SIZE, V37_WIDTH],
        "mask_shape": [V37_PATCHES],
        "paired_paths": 5,
        "distinct_training_font_paths_per_item": 5,
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


__all__ = [
    "V38_DEVELOPMENT_FONT",
    "V38_HELD_FONT",
    "V38_PIXEL_KEYS",
    "V38_SEALED_FONT",
    "V38_TRAIN_FONTS",
    "VisualPathAlignmentDataset",
    "VisualPathAlignmentParaphrase",
    "load_v37_instruction_records",
    "load_v38_paraphrases",
    "render_visual_path_alignment_record",
    "select_v38_instruction_records",
    "visual_path_alignment_collate",
    "visual_path_alignment_data_boundary_receipt",
    "visual_path_alignment_fonts",
    "visual_path_alignment_pixel_batch",
    "visual_path_alignment_record_fits",
]

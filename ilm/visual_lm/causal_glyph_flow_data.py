from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .direct_visual_patch_data import (
    V33_SPLITS,
    V33_STUDENT_KEYS,
    DirectPatchRenderConfig,
    direct_patch_collate,
    direct_patch_partition,
    direct_patch_student_batch,
    render_direct_patch_instruction,
)
from .visual_semantic_raster_data import (
    VisualRasterRecord,
    VisualTextRecord,
    normalize_visible_text,
)


V35_STUDENT_KEYS = V33_STUDENT_KEYS
V35_STAGE_C_MIXTURE = (
    "instruction",
    "instruction",
    "instruction",
    "instruction",
    "instruction",
    "instruction",
    "copy",
    "public",
)


class CausalGlyphCopyDataset(Dataset[dict[str, Any]]):
    """Deterministic visual copy examples with source-inherited partitions."""

    def __init__(
        self,
        records: Sequence[VisualTextRecord],
        *,
        split: str,
        config: DirectPatchRenderConfig,
        length: int,
        seed: int = 20_263_500,
        minimum_characters: int = 2,
        maximum_characters: int = 16,
    ) -> None:
        if split not in V33_SPLITS:
            raise ValueError(f"unknown V35 split: {split!r}")
        if length < 1:
            raise ValueError("V35 copy dataset length must be positive")
        if not 2 <= minimum_characters <= maximum_characters <= 16:
            raise ValueError("V35 copy span must contain 2 to 16 characters")
        selected = tuple(
            record
            for record in records
            if direct_patch_partition(record.identifier, stream="public-domain")
            == split
            and len(normalize_visible_text(record.text)) >= minimum_characters
        )
        if not selected:
            raise ValueError(f"V35 copy split {split!r} is empty")
        self.records = selected
        self.split = split
        self.config = config
        self.length = int(length)
        self.seed = int(seed)
        self.minimum_characters = int(minimum_characters)
        self.maximum_characters = int(maximum_characters)

    def __len__(self) -> int:
        return self.length

    def _span(self, text: str, rng: random.Random) -> tuple[str, int, int]:
        normalized = normalize_visible_text(text)
        maximum = min(self.maximum_characters, len(normalized))
        for _ in range(256):
            count = rng.randint(self.minimum_characters, maximum)
            start = rng.randrange(len(normalized) - count + 1)
            span = normalize_visible_text(normalized[start : start + count])
            if len(span) >= self.minimum_characters:
                return span, start, count
        raise RuntimeError("V35 could not select a visible copy span")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        rng = random.Random(self.seed + index * 1_000_037)
        source = self.records[rng.randrange(len(self.records))]
        span, start, source_count = self._span(source.text, rng)
        record = VisualRasterRecord(
            identifier=f"copy:{source.identifier}:{start}:{source_count}",
            prompt=f"照写：{span}",
            answer=span,
            language=source.language,
            source=source.source,
            rights=source.rights,
        )
        sample = render_direct_patch_instruction(
            record,
            split=self.split,
            config=self.config,
            variant=rng.randrange(2**31),
        )
        sample["metadata"] = dict(sample["metadata"]) | {
            "stream": "copy",
            "source_identifier": source.identifier,
            "source_split": direct_patch_partition(
                source.identifier,
                stream="public-domain",
            ),
            "source_start": start,
            "source_characters": source_count,
            "copy_span": span,
        }
        return sample


class CausalGlyphStageCMixture(Dataset[dict[str, Any]]):
    """Exact deterministic 75% instruction, 12.5% copy, 12.5% replay."""

    def __init__(
        self,
        instruction: Dataset[dict[str, Any]],
        copy: Dataset[dict[str, Any]],
        public: Dataset[dict[str, Any]],
        *,
        length: int,
    ) -> None:
        datasets = {
            "instruction": instruction,
            "copy": copy,
            "public": public,
        }
        if length < 1 or any(len(dataset) < 1 for dataset in datasets.values()):
            raise ValueError("V35 Stage C mixture requires non-empty datasets")
        self.datasets = datasets
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def stream_for_index(index: int) -> str:
        if index < 0:
            raise IndexError(index)
        return V35_STAGE_C_MIXTURE[index % len(V35_STAGE_C_MIXTURE)]

    @staticmethod
    def stream_index(index: int) -> int:
        if index < 0:
            raise IndexError(index)
        cycle, position = divmod(index, len(V35_STAGE_C_MIXTURE))
        if position < 6:
            return cycle * 6 + position
        return cycle

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        stream = self.stream_for_index(index)
        dataset = self.datasets[stream]
        source_index = self.stream_index(index) % len(dataset)
        sample = dict(dataset[source_index])
        sample["metadata"] = dict(sample["metadata"]) | {
            "mixture_stream": stream,
            "mixture_index": index,
        }
        return sample

    def mixture_counts(self) -> dict[str, int]:
        return dict(
            Counter(self.stream_for_index(index) for index in range(self.length))
        )


def causal_glyph_flow_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return direct_patch_collate(samples)


def causal_glyph_flow_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = direct_patch_student_batch(batch)
    if tuple(student) != V35_STUDENT_KEYS:
        raise RuntimeError("V35 student batch has an invalid visual boundary")
    return student


def causal_glyph_flow_data_boundary_receipt(
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    student = causal_glyph_flow_student_batch(batch)
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

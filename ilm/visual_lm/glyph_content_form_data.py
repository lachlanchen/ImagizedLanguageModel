from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .continuous_glyph_codec_data import (
    V34_HISTORIC_SPLITS,
    HistoricGlyphRecord,
    historic_character_partition,
)


V40_STUDENT_KEYS = (
    "anchor_pixels",
    "positive_pixels",
    "anchor_style_pixels",
    "positive_style_pixels",
)


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class CrossEraContentFormDataset(Dataset[dict[str, Any]]):
    """Cross-era identity pairs with style references from other families."""

    def __init__(
        self,
        records: Sequence[HistoricGlyphRecord],
        pixels: torch.Tensor,
        *,
        split: str,
        length: int,
        seed: int = 20_264_000,
    ) -> None:
        if split not in V34_HISTORIC_SPLITS:
            raise ValueError(f"unknown V40 historical split: {split!r}")
        if length < 1:
            raise ValueError("V40 dataset length must be positive")
        if len(records) != len(pixels):
            raise ValueError("V40 historical records and pixels do not align")
        expected = (len(records), 1, 32, 32)
        if pixels.dtype != torch.uint8 or tuple(pixels.shape) != expected:
            raise ValueError("V40 historical raster cache has invalid geometry")
        if not bool(((pixels == 0) | (pixels == 1)).all()):
            raise ValueError("V40 historical raster cache must be binary")

        family_stages: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        stage_families: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record in enumerate(records):
            if historic_character_partition(record.character) != split:
                continue
            family_stages[record.character][record.stage].append(index)
            stage_families[record.stage][record.character].append(index)

        families = [
            character
            for character, stages in family_stages.items()
            if len(
                [
                    stage
                    for stage in stages
                    if len(stage_families[stage]) >= 2
                ]
            )
            >= 2
        ]
        if len(families) < 2:
            raise ValueError("V40 requires at least two cross-era character families")
        random.Random(seed).shuffle(families)
        self.records = tuple(records)
        self.pixels = pixels
        self.split = split
        self.length = int(length)
        self.seed = int(seed)
        self.families = tuple(families)
        selected_families = set(families)
        self.family_stages = {
            character: {
                stage: tuple(indices)
                for stage, indices in sorted(stages.items())
                if len(stage_families[stage]) >= 2
            }
            for character, stages in family_stages.items()
            if character in selected_families
        }
        self.stage_families = {
            stage: {
                character: tuple(indices)
                for character, indices in sorted(families_by_stage.items())
            }
            for stage, families_by_stage in stage_families.items()
            if len(families_by_stage) >= 2
        }

    def __len__(self) -> int:
        return self.length

    def _family_for_index(self, index: int) -> str:
        # Repeating one seeded permutation prevents duplicate families across a
        # cycle boundary while the index-specific seed still changes variants.
        return self.families[index % len(self.families)]

    def _style_reference(
        self,
        *,
        stage: str,
        excluded_character: str,
        rng: random.Random,
    ) -> tuple[int, str]:
        candidates = [
            character
            for character in self.stage_families[stage]
            if character != excluded_character
        ]
        reference_character = candidates[rng.randrange(len(candidates))]
        indices = self.stage_families[stage][reference_character]
        return indices[rng.randrange(len(indices))], reference_character

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        character = self._family_for_index(index)
        rng = random.Random(_stable_seed(self.seed, self.split, index, character))
        stages = sorted(self.family_stages[character])
        anchor_stage = stages[rng.randrange(len(stages))]
        positive_stages = [stage for stage in stages if stage != anchor_stage]
        positive_stage = positive_stages[rng.randrange(len(positive_stages))]
        anchor_indices = self.family_stages[character][anchor_stage]
        positive_indices = self.family_stages[character][positive_stage]
        anchor_index = anchor_indices[rng.randrange(len(anchor_indices))]
        positive_index = positive_indices[rng.randrange(len(positive_indices))]
        anchor_style_index, anchor_style_character = self._style_reference(
            stage=anchor_stage,
            excluded_character=character,
            rng=rng,
        )
        positive_style_index, positive_style_character = self._style_reference(
            stage=positive_stage,
            excluded_character=character,
            rng=rng,
        )
        return {
            "anchor_pixels": self.pixels[anchor_index].float(),
            "positive_pixels": self.pixels[positive_index].float(),
            "anchor_style_pixels": self.pixels[anchor_style_index].float(),
            "positive_style_pixels": self.pixels[positive_style_index].float(),
            "metadata": {
                "character": character,
                "split": self.split,
                "anchor_stage": anchor_stage,
                "positive_stage": positive_stage,
                "anchor_identifier": self.records[anchor_index].identifier,
                "positive_identifier": self.records[positive_index].identifier,
                "anchor_style_character": anchor_style_character,
                "positive_style_character": positive_style_character,
                "anchor_style_identifier": self.records[
                    anchor_style_index
                ].identifier,
                "positive_style_identifier": self.records[
                    positive_style_index
                ].identifier,
            },
        }


def glyph_content_form_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("V40 cannot collate an empty batch")
    return {
        **{
            key: torch.stack([sample[key] for sample in samples])
            for key in V40_STUDENT_KEYS
        },
        "metadata": [dict(sample["metadata"]) for sample in samples],
    }


def glyph_content_form_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    missing = [key for key in V40_STUDENT_KEYS if key not in batch]
    if missing:
        raise ValueError(f"V40 batch lacks visual tensors: {missing}")
    student = {key: batch[key] for key in V40_STUDENT_KEYS}
    if not all(isinstance(value, torch.Tensor) for value in student.values()):
        raise TypeError("V40 student inputs must all be tensors")
    return student


def glyph_content_form_stage_ids(
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, dict[str, int]]:
    if not metadata:
        raise ValueError("V40 stage supervision requires metadata")
    stages = sorted(
        {
            str(row[key])
            for row in metadata
            for key in ("anchor_stage", "positive_stage")
        }
    )
    stage_to_id = {stage: index for index, stage in enumerate(stages)}
    anchor = [stage_to_id[str(row["anchor_stage"])] for row in metadata]
    positive = [stage_to_id[str(row["positive_stage"])] for row in metadata]
    # Forms are concatenated as anchor, positive, anchor reference, positive reference.
    return torch.tensor(anchor + positive + anchor + positive), stage_to_id


def glyph_content_form_data_boundary_receipt(
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    student = glyph_content_form_student_batch(batch)
    metadata = batch.get("metadata")
    return {
        "student_keys": sorted(student),
        "metadata_excluded_from_model": "metadata" not in student,
        "all_student_values_are_tensors": all(
            isinstance(value, torch.Tensor) for value in student.values()
        ),
        "student_contains_strings": any(
            isinstance(value, str) for value in student.values()
        ),
        "host_metadata_contains_family_labels": bool(metadata),
        "family_labels_are_model_inputs": False,
        "stage_labels_are_model_inputs": False,
        "shapes": {key: list(value.shape) for key, value in student.items()},
    }


__all__ = [
    "CrossEraContentFormDataset",
    "V40_STUDENT_KEYS",
    "glyph_content_form_collate",
    "glyph_content_form_data_boundary_receipt",
    "glyph_content_form_stage_ids",
    "glyph_content_form_student_batch",
]

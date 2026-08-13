from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .conditional_visual_density_ratio_data import (
    ConditionalVisualCandidateBank,
    ConditionalVisualNaturalDataset,
    ConditionalVisualRenderConfig,
    build_v29_candidate_bank,
    build_v29_candidate_statistics,
    canonical_target_indices,
    conditional_visual_candidate_bank_receipt,
    conditional_visual_natural_collate,
    conditional_visual_render_config_payload,
)
from .conditional_visual_field_flow import V31_ARCHITECTURE
from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    V25_FONT_PARTITIONS,
    V25_TRAIN_FONTS,
    render_visual_cell_stream,
    script_variants,
    visual_cell_partition,
)


V31_CONTEXT_CELLS = 64
V31_SEQUENCE_CELLS = 65
V31_BANK_VIEWS = 2
V31_NATURAL_STUDENT_KEYS = (
    "first_context",
    "second_context",
    "first_target",
    "second_target",
)


class ConditionalVisualFlowNaturalDataset(Dataset):
    """Render a full visual sequence while keeping identity metadata host-side."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        allowed_targets: set[str] | frozenset[str],
        split: str,
        render_config: ConditionalVisualRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if split == "frozen":
            raise PermissionError("V31 training cannot open frozen writing")
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("unknown V31 data split")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record, mode=render_config.script_views
            ):
                if len(writing) >= V31_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected or not allowed_targets or length < 1:
            raise ValueError("V31 natural dataset is empty")
        self.records = tuple(selected)
        self.allowed_targets = frozenset(allowed_targets)
        self.split = split
        self.render_config = render_config
        self.canonical_config = ConditionalVisualRenderConfig(
            minimum_font_size=render_config.minimum_font_size,
            maximum_font_size=render_config.maximum_font_size,
            augment=False,
            script_views="original",
        )
        self.seed = int(seed)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 104_729)
        selected = None
        for _ in range(512):
            record, script_view, writing = rng.choice(self.records)
            start = rng.randint(0, len(writing) - V31_SEQUENCE_CELLS)
            target = writing[start + V31_CONTEXT_CELLS]
            if target in self.allowed_targets:
                selected = (record, script_view, writing, start, target)
                break
        if selected is None:
            raise RuntimeError("V31 could not sample an allowed natural target")
        record, script_view, writing, start, target = selected
        segment = writing[start : start + V31_SEQUENCE_CELLS]
        fonts = V25_FONT_PARTITIONS[self.split]
        first_index = rng.randrange(len(fonts))
        second_index = (first_index + 1 + rng.randrange(len(fonts) - 1)) % len(fonts)
        first = render_visual_cell_stream(
            segment,
            config=self.render_config,
            font_path=fonts[first_index],
            variant=rng.randrange(2**31),
        )
        second = render_visual_cell_stream(
            segment,
            config=self.render_config,
            font_path=fonts[second_index],
            variant=rng.randrange(2**31),
        )
        canonical = render_visual_cell_stream(
            target,
            config=self.canonical_config,
            font_path=V25_TRAIN_FONTS[0],
            variant=0,
        )[0]
        return {
            "first_context": first[:V31_CONTEXT_CELLS],
            "second_context": second[:V31_CONTEXT_CELLS],
            "first_target": first[V31_CONTEXT_CELLS],
            "second_target": second[V31_CONTEXT_CELLS],
            "canonical_target": canonical,
            "metadata": {
                "identifier": record.identifier,
                "script_view": script_view,
                "offset": start,
                "target": target,
                "first_font": fonts[first_index],
                "second_font": fonts[second_index],
            },
        }


def conditional_visual_flow_natural_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V31 natural batch")
    image_keys = (*V31_NATURAL_STUDENT_KEYS, "canonical_target")
    return {key: torch.stack([item[key] for item in batch]) for key in image_keys} | {
        "metadata": [item["metadata"] for item in batch]
    }


def build_v31_candidate_statistics(*args: Any, **kwargs: Any) -> Any:
    return build_v29_candidate_statistics(*args, **kwargs)


def build_v31_candidate_bank(
    *args: Any, **kwargs: Any
) -> ConditionalVisualCandidateBank:
    return build_v29_candidate_bank(*args, **kwargs)


def conditional_visual_flow_natural_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V31_NATURAL_STUDENT_KEYS}
    for name, value in student.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V31 natural value {name!r} is not an image tensor")
        expected = (
            (V31_CONTEXT_CELLS, 1, 32, 32) if name.endswith("context") else (1, 32, 32)
        )
        if tuple(value.shape[1:]) != expected:
            raise ValueError(f"V31 natural value {name!r} has invalid shape")
    return student


def conditional_visual_field_flow_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V31_ARCHITECTURE,
        "natural_context_shape": [V31_CONTEXT_CELLS, 1, 32, 32],
        "candidate_shape": [1, 32, 32],
        "generated_field_shape": [16, 192],
        "student_natural_keys": list(V31_NATURAL_STUDENT_KEYS),
        "input_is_continuous_image_stream": True,
        "output_is_candidate_independent_continuous_distribution": True,
        "canonical_target_pixels_are_host_only": True,
        "canonical_target_pixels_enter_student": False,
        "canonical_indices_enter_student": False,
        "pair_assignment_labels_are_positions": True,
        "pair_candidate_order_is_randomized": True,
        "pair_suffix_pixels_identical": True,
        "training_bank_is_host_only": True,
        "candidate_bank_deployed": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_external_language_model": False,
    }


__all__ = [
    "ConditionalVisualCandidateBank",
    "ConditionalVisualFlowNaturalDataset",
    "ConditionalVisualNaturalDataset",
    "ConditionalVisualRenderConfig",
    "V31_BANK_VIEWS",
    "V31_CONTEXT_CELLS",
    "V31_NATURAL_STUDENT_KEYS",
    "V31_SEQUENCE_CELLS",
    "build_v31_candidate_bank",
    "build_v31_candidate_statistics",
    "canonical_target_indices",
    "conditional_visual_candidate_bank_receipt",
    "conditional_visual_field_flow_data_boundary_receipt",
    "conditional_visual_flow_natural_collate",
    "conditional_visual_flow_natural_student_batch",
    "conditional_visual_natural_collate",
    "conditional_visual_render_config_payload",
]

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .factorized_visual_context_data import (
    FactorizedVisualSuffixPair,
    build_factorized_suffix_pairs,
)
from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    V25_FONT_PARTITIONS,
    V25_TRAIN_FONTS,
    render_visual_cell_stream,
    script_variants,
    visual_cell_partition,
)


V27_CONTEXT_CELLS = 64
V27_SEQUENCE_CELLS = 65
V27_NATURAL_IMAGE_KEYS = (
    "context",
    "target",
    "reference_context",
    "reference_target",
    "canonical_target",
)
V27_PAIR_IMAGE_KEYS = (
    "contexts",
    "candidates",
    "reference_contexts",
    "reference_candidates",
)
V27_PAIR_ASSIGNMENT_KEYS = (
    "assignment",
    "reference_assignment",
)

JointVisualSuffixPair = FactorizedVisualSuffixPair


@dataclass(frozen=True)
class JointVisualRenderConfig:
    cell_size: int = 32
    minimum_font_size: int = 24
    maximum_font_size: int = 28
    augment: bool = True
    script_views: str = "original+simplified"

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V27 fixes visual cells to 32x32")
        if not 8 <= self.minimum_font_size <= self.maximum_font_size <= 32:
            raise ValueError("V27 font sizes must fit inside a cell")
        if self.script_views not in {"original", "original+simplified"}:
            raise ValueError("unknown V27 script-view mode")


def build_joint_suffix_pairs(
    records: Sequence[VisualGrammarRecord],
    *,
    split: str,
    suffix_cells: int,
    count: int,
    seed: int,
    require_different_identifiers: bool,
    allowed_targets: set[str] | frozenset[str] | None = None,
    script_views_mode: str = "original+simplified",
) -> tuple[JointVisualSuffixPair, ...]:
    return build_factorized_suffix_pairs(
        records,
        split=split,
        suffix_cells=suffix_cells,
        count=count,
        seed=seed,
        require_different_identifiers=require_different_identifiers,
        allowed_targets=allowed_targets,
        script_views_mode=script_views_mode,
    )


class JointVisualNaturalDataset(Dataset):
    """Render cross-font next-cell examples plus a pixel identity witness."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        split: str,
        render_config: JointVisualRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if split == "frozen":
            raise PermissionError("V27 natural training cannot open frozen writing")
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("unknown V27 data split")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record, mode=render_config.script_views
            ):
                if len(writing) >= V27_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected or length < 1:
            raise ValueError("V27 natural dataset is empty")
        self.records = selected
        self.split = split
        self.render_config = render_config
        self.canonical_config = JointVisualRenderConfig(
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
        record, script_view, writing = rng.choice(self.records)
        start = rng.randint(0, len(writing) - V27_SEQUENCE_CELLS)
        segment = writing[start : start + V27_SEQUENCE_CELLS]
        fonts = V25_FONT_PARTITIONS[self.split]
        first_index = rng.randrange(len(fonts))
        second_index = (
            first_index + 1 + rng.randrange(len(fonts) - 1)
        ) % len(fonts)
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
            segment[-1],
            config=self.canonical_config,
            font_path=V25_TRAIN_FONTS[0],
            variant=0,
        )
        return {
            "context": first[:64],
            "target": first[64:65],
            "reference_context": second[:64],
            "reference_target": second[64:65],
            "canonical_target": canonical,
            "metadata": {
                "identifier": record.identifier,
                "script_view": script_view,
                "offset": start,
                "first_font": fonts[first_index],
                "second_font": fonts[second_index],
            },
        }


def _permutation_from_swap(swap: bool) -> tuple[list[int], torch.Tensor]:
    order = [1, 0] if swap else [0, 1]
    assignment = torch.empty(2, dtype=torch.long)
    for source_row, candidate_column in enumerate(order):
        assignment[candidate_column] = source_row
    return order, assignment


class JointVisualPairDataset(Dataset):
    """Render matched-suffix pairs with independently permuted candidates."""

    def __init__(
        self,
        pairs: Sequence[JointVisualSuffixPair],
        *,
        split: str,
        render_config: JointVisualRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if split == "frozen":
            raise PermissionError("V27 pair training cannot open frozen writing")
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("unknown V27 pair split")
        if not pairs or length < 1:
            raise ValueError("V27 pair dataset is empty")
        self.pairs = tuple(pairs)
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def _render(self, writing: str, *, font: str, variant: int) -> torch.Tensor:
        return render_visual_cell_stream(
            writing,
            config=self.render_config,
            font_path=font,
            variant=variant,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 104_729)
        pair = self.pairs[index % len(self.pairs)]
        fonts = V25_FONT_PARTITIONS[self.split]
        first_index = rng.randrange(len(fonts))
        second_index = (
            first_index + 1 + rng.randrange(len(fonts) - 1)
        ) % len(fonts)
        first_variant = rng.randrange(2**31)
        second_variant = rng.randrange(2**31)
        first_a = self._render(
            pair.context_a + pair.target_a,
            font=fonts[first_index],
            variant=first_variant,
        )
        first_b = self._render(
            pair.context_b + pair.target_b,
            font=fonts[first_index],
            variant=first_variant,
        )
        second_a = self._render(
            pair.context_a + pair.target_a,
            font=fonts[second_index],
            variant=second_variant,
        )
        second_b = self._render(
            pair.context_b + pair.target_b,
            font=fonts[second_index],
            variant=second_variant,
        )
        suffix = pair.suffix_cells
        if not torch.equal(first_a[64 - suffix : 64], first_b[64 - suffix : 64]):
            raise RuntimeError("V27 first shared suffix pixels are not equal")
        if not torch.equal(second_a[64 - suffix : 64], second_b[64 - suffix : 64]):
            raise RuntimeError("V27 second shared suffix pixels are not equal")

        first_order, first_assignment = _permutation_from_swap(
            bool(rng.getrandbits(1))
        )
        second_order, second_assignment = _permutation_from_swap(
            bool(rng.getrandbits(1))
        )
        first_targets = torch.stack((first_a[64], first_b[64]))
        second_targets = torch.stack((second_a[64], second_b[64]))
        return {
            "contexts": torch.stack((first_a[:64], first_b[:64])),
            "candidates": second_targets[first_order],
            "assignment": first_assignment,
            "reference_contexts": torch.stack((second_a[:64], second_b[:64])),
            "reference_candidates": first_targets[second_order],
            "reference_assignment": second_assignment,
            "metadata": {
                "identifier_a": pair.identifier_a,
                "identifier_b": pair.identifier_b,
                "script_view_a": pair.script_view_a,
                "script_view_b": pair.script_view_b,
                "target_a": pair.target_a,
                "target_b": pair.target_b,
                "suffix": pair.suffix,
                "suffix_cells": pair.suffix_cells,
                "first_font": fonts[first_index],
                "second_font": fonts[second_index],
            },
        }


class JointVisualPairAuditDataset(JointVisualPairDataset):
    def __init__(self, pairs: Sequence[JointVisualSuffixPair]) -> None:
        super().__init__(
            pairs,
            split="development",
            render_config=JointVisualRenderConfig(
                augment=False, script_views="original"
            ),
            seed=20260915,
            length=len(pairs),
        )


def joint_visual_natural_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V27 natural batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V27_NATURAL_IMAGE_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def joint_visual_pair_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V27 pair batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V27_PAIR_IMAGE_KEYS
    } | {
        key: torch.stack([item[key] for item in batch])
        for key in V27_PAIR_ASSIGNMENT_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def joint_visual_natural_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V27_NATURAL_IMAGE_KEYS}
    if set(student) != set(V27_NATURAL_IMAGE_KEYS):
        raise ValueError("V27 natural student batch has unregistered values")
    for name, value in student.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V27 natural value {name!r} is not an image tensor")
        if value.ndim != 5 or tuple(value.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V27 natural value {name!r} has invalid shape")
    return student


def joint_visual_pair_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    keys = (*V27_PAIR_IMAGE_KEYS, *V27_PAIR_ASSIGNMENT_KEYS)
    student = {key: batch[key] for key in keys}
    for name in V27_PAIR_IMAGE_KEYS:
        value = student[name]
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V27 pair value {name!r} is not an image tensor")
        expected_rank = 6 if "contexts" in name else 5
        if value.ndim != expected_rank or tuple(value.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V27 pair value {name!r} has invalid shape")
    for name in V27_PAIR_ASSIGNMENT_KEYS:
        value = student[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
            raise TypeError(f"V27 pair value {name!r} is not an int64 position")
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"V27 pair value {name!r} has invalid shape")
    return student


def joint_visual_render_config_payload(
    config: JointVisualRenderConfig,
) -> dict[str, Any]:
    return asdict(config)


def joint_visual_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": "joint-visual-compatibility-v27",
        "natural_context_shape": [64, 1, 32, 32],
        "natural_target_shape": [1, 1, 32, 32],
        "pair_context_shape": [2, 64, 1, 32, 32],
        "pair_candidate_shape": [2, 1, 32, 32],
        "canonical_identity_derived_from_exact_pixels": True,
        "pair_assignment_labels_are_positions": True,
        "pair_candidate_order_is_randomized": True,
        "pair_suffix_pixels_identical": True,
        "input_is_continuous_image_stream": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }

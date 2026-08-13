from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import VisualGrammarRecord
from .joint_visual_compatibility_data import (
    JointVisualPairAuditDataset,
    JointVisualPairDataset,
    JointVisualSuffixPair,
    build_joint_suffix_pairs,
    joint_visual_pair_collate,
    joint_visual_pair_student_batch,
)
from .visual_cell_data import (
    V25_FONT_PARTITIONS,
    V25_TRAIN_FONTS,
    render_visual_cell_stream,
    script_variants,
    visual_cell_partition,
)


V28_CONTEXT_CELLS = 64
V28_FUTURE_CELLS = 4
V28_SEQUENCE_CELLS = V28_CONTEXT_CELLS + V28_FUTURE_CELLS
V28_NATURAL_IMAGE_KEYS = ("first_view", "second_view", "canonical")
V28_NATURAL_STUDENT_KEYS = ("first_view", "second_view")


@dataclass(frozen=True)
class DenseVisualRenderConfig:
    cell_size: int = 32
    minimum_font_size: int = 24
    maximum_font_size: int = 28
    augment: bool = True
    script_views: str = "original+simplified"

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V28 fixes visual cells to 32x32")
        if not 8 <= self.minimum_font_size <= self.maximum_font_size <= 32:
            raise ValueError("V28 font sizes must fit inside a cell")
        if self.script_views not in {"original", "original+simplified"}:
            raise ValueError("unknown V28 script-view mode")


class DenseVisualNaturalDataset(Dataset):
    """Render two font views of a 68-cell visual language segment."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        split: str,
        render_config: DenseVisualRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if split == "frozen":
            raise PermissionError("V28 training cannot open frozen writing")
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("unknown V28 data split")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record, mode=render_config.script_views
            ):
                if len(writing) >= V28_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected or length < 1:
            raise ValueError("V28 natural dataset is empty")
        self.records = selected
        self.split = split
        self.render_config = render_config
        self.canonical_config = DenseVisualRenderConfig(
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
        start = rng.randint(0, len(writing) - V28_SEQUENCE_CELLS)
        segment = writing[start : start + V28_SEQUENCE_CELLS]
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
            segment,
            config=self.canonical_config,
            font_path=V25_TRAIN_FONTS[0],
            variant=0,
        )
        return {
            "first_view": first,
            "second_view": second,
            "canonical": canonical,
            "metadata": {
                "identifier": record.identifier,
                "script_view": script_view,
                "offset": start,
                "first_font": fonts[first_index],
                "second_font": fonts[second_index],
            },
        }


def dense_visual_natural_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V28 natural batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V28_NATURAL_IMAGE_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def dense_visual_natural_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V28_NATURAL_STUDENT_KEYS}
    for name, value in student.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V28 natural value {name!r} is not an image tensor")
        if value.ndim != 5 or tuple(value.shape[1:]) != (
            V28_SEQUENCE_CELLS,
            1,
            32,
            32,
        ):
            raise ValueError(f"V28 natural value {name!r} has invalid shape")
    return student


def canonical_pixel_groups(canonical: torch.Tensor) -> torch.Tensor:
    """Create temporary equality groups from pixels, without persistent IDs."""

    if not torch.is_floating_point(canonical):
        raise TypeError("V28 canonical witness must be a floating image tensor")
    if canonical.ndim < 4 or tuple(canonical.shape[-3:]) != (1, 32, 32):
        raise ValueError("V28 canonical witness must end in [1,32,32]")
    leading = canonical.shape[:-3]
    flat = canonical.reshape(-1, 32 * 32)
    quantized = (flat.clamp(0, 1) * 255.0).round().to(torch.uint8)
    _, inverse = torch.unique(quantized, dim=0, return_inverse=True)
    return inverse.reshape(*leading).long()


def stratified_causal_positions(
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    positions: list[torch.Tensor] = []
    for start in (0, 16, 32):
        positions.append(
            torch.randperm(16, generator=generator)[:4] + start
        )
    positions.append(torch.randperm(15, generator=generator)[:3] + 48)
    positions.append(torch.tensor([63], dtype=torch.long))
    selected = torch.cat(positions).sort().values
    if selected.shape != (16,) or int(selected[-1]) != 63:
        raise RuntimeError("V28 stratified position sampler violated its contract")
    return selected.to(device=device)


def causal_position_weights(positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim != 1 or positions.numel() != 16:
        raise ValueError("V28 requires exactly 16 selected positions")
    if positions.dtype != torch.long or int(positions.min()) < 0:
        raise ValueError("V28 selected positions must be nonnegative int64")
    if int(positions.max()) >= V28_CONTEXT_CELLS:
        raise ValueError("V28 selected positions exceed the causal context")
    value = positions.float() + 1.0
    weights = 0.25 + 0.75 * (value / V28_CONTEXT_CELLS).square()
    weights = torch.where(positions == 63, weights * 2.0, weights)
    return weights / weights.mean()


def dense_visual_render_config_payload(
    config: DenseVisualRenderConfig,
) -> dict[str, Any]:
    return asdict(config)


def dense_visual_data_boundary_receipt() -> dict[str, bool | str | list[int]]:
    return {
        "architecture": "dense-visual-future-energy-v28",
        "natural_stream_shape": [V28_SEQUENCE_CELLS, 1, 32, 32],
        "student_natural_keys": list(V28_NATURAL_STUDENT_KEYS),
        "input_is_continuous_image_stream": True,
        "canonical_identity_derived_from_exact_pixels": True,
        "canonical_groups_are_temporary_loss_only": True,
        "pair_assignment_labels_are_positions": True,
        "pair_candidate_order_is_randomized": True,
        "pair_suffix_pixels_identical": True,
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
    "DenseVisualNaturalDataset",
    "DenseVisualRenderConfig",
    "JointVisualPairAuditDataset",
    "JointVisualPairDataset",
    "JointVisualSuffixPair",
    "V28_CONTEXT_CELLS",
    "V28_FUTURE_CELLS",
    "V28_NATURAL_IMAGE_KEYS",
    "V28_NATURAL_STUDENT_KEYS",
    "V28_SEQUENCE_CELLS",
    "build_joint_suffix_pairs",
    "canonical_pixel_groups",
    "causal_position_weights",
    "dense_visual_data_boundary_receipt",
    "dense_visual_natural_collate",
    "dense_visual_natural_student_batch",
    "dense_visual_render_config_payload",
    "joint_visual_pair_collate",
    "joint_visual_pair_student_batch",
    "stratified_causal_positions",
]

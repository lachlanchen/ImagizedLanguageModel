from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    V25_FONT_PARTITIONS,
    V25_TRAIN_FONTS,
    render_visual_cell_stream,
    script_variants,
    visual_cell_partition,
)
from .visual_cell_eval_data import (
    VisualCharacterStatistics,
    build_visual_character_statistics,
)


V29_CONTEXT_CELLS = 64
V29_SEQUENCE_CELLS = 65
V29_BANK_VIEWS = 2
V29_NATURAL_IMAGE_KEYS = (
    "first_context",
    "second_context",
    "canonical_target",
)
V29_NATURAL_STUDENT_KEYS = ("first_context", "second_context")


@dataclass(frozen=True)
class ConditionalVisualRenderConfig:
    cell_size: int = 32
    minimum_font_size: int = 24
    maximum_font_size: int = 28
    augment: bool = True
    script_views: str = "original+simplified"

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V29 fixes visual cells to 32x32")
        if not 8 <= self.minimum_font_size <= self.maximum_font_size <= 32:
            raise ValueError("V29 font sizes must fit inside a cell")
        if self.script_views not in {"original", "original+simplified"}:
            raise ValueError("unknown V29 script-view mode")


@dataclass(frozen=True)
class ConditionalVisualCandidateBank:
    images: torch.Tensor
    canonical: torch.Tensor
    forms: tuple[str, ...]
    counts: tuple[int, ...]
    font_paths: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        identities = len(self.forms)
        if identities < 2 or len(set(self.forms)) != identities:
            raise ValueError("V29 candidate forms must be unique")
        if len(self.counts) != identities or any(value < 1 for value in self.counts):
            raise ValueError("V29 candidate counts must be positive and aligned")
        if tuple(self.images.shape) != (V29_BANK_VIEWS, identities, 1, 32, 32):
            raise ValueError("V29 candidate images have an invalid shape")
        if tuple(self.canonical.shape) != (identities, 1, 32, 32):
            raise ValueError("V29 canonical bank has an invalid shape")
        if not torch.is_floating_point(self.images) or not torch.is_floating_point(
            self.canonical
        ):
            raise TypeError("V29 candidate bank must contain floating images")
        if len(self.font_paths) != V29_BANK_VIEWS:
            raise ValueError("V29 requires exactly two candidate font views")

    @property
    def size(self) -> int:
        return len(self.forms)

    @property
    def index(self) -> dict[str, int]:
        return {form: index for index, form in enumerate(self.forms)}


def _quantized_rows(images: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(images):
        raise TypeError("V29 canonical matching requires floating images")
    if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
        raise ValueError("V29 canonical matching requires [...,1,32,32]")
    return (
        images.detach().cpu().reshape(-1, 32 * 32).clamp(0, 1).mul(255).round()
    ).to(torch.uint8)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def build_v29_candidate_statistics(
    records: Sequence[VisualGrammarRecord],
    *,
    bank_size: int = 1_024,
) -> VisualCharacterStatistics:
    """Select frequent training forms after canonical-pixel deduplication."""

    if bank_size < 2:
        raise ValueError("V29 candidate bank must contain at least two forms")
    pool_size = bank_size * 2
    pool = build_visual_character_statistics(records, bank_size=pool_size)
    config = ConditionalVisualRenderConfig(augment=False, script_views="original")
    rendered = render_visual_cell_stream(
        "".join(pool.characters),
        config=config,
        font_path=V25_TRAIN_FONTS[0],
        variant=0,
    )
    rows = _quantized_rows(rendered)
    seen: set[bytes] = set()
    selected_old_indices: list[int] = []
    for index, row in enumerate(rows):
        key = bytes(row.tolist())
        if key in seen:
            continue
        seen.add(key)
        selected_old_indices.append(index)
        if len(selected_old_indices) == bank_size:
            break
    if len(selected_old_indices) != bank_size:
        raise ValueError(
            f"V29 found {len(selected_old_indices)} unique canonical pixels, "
            f"needs {bank_size}"
        )

    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(selected_old_indices)
    }
    return VisualCharacterStatistics(
        characters=tuple(pool.characters[index] for index in selected_old_indices),
        counts=tuple(pool.counts[index] for index in selected_old_indices),
        bigram_rows={
            condition: tuple(
                (old_to_new[index], count)
                for index, count in row
                if index in old_to_new
            )
            for condition, row in pool.bigram_rows.items()
            if any(index in old_to_new for index, _ in row)
        },
        visible_character_count=pool.visible_character_count,
        han_character_count=pool.han_character_count,
    )


def build_v29_candidate_bank(
    statistics: VisualCharacterStatistics,
    *,
    seed: int,
) -> ConditionalVisualCandidateBank:
    config = ConditionalVisualRenderConfig(augment=False, script_views="original")
    writing = "".join(statistics.characters)
    views = torch.stack(
        [
            render_visual_cell_stream(
                writing,
                config=config,
                font_path=font,
                variant=seed + view,
            )
            for view, font in enumerate(V25_TRAIN_FONTS[:V29_BANK_VIEWS])
        ]
    )
    canonical = render_visual_cell_stream(
        writing,
        config=config,
        font_path=V25_TRAIN_FONTS[0],
        variant=0,
    )
    rows = _quantized_rows(canonical)
    if torch.unique(rows, dim=0).shape[0] != len(statistics.characters):
        raise ValueError("V29 candidate statistics contain duplicate canonical pixels")
    return ConditionalVisualCandidateBank(
        images=views,
        canonical=canonical,
        forms=statistics.characters,
        counts=statistics.counts,
        font_paths=tuple(V25_TRAIN_FONTS[:V29_BANK_VIEWS]),
        seed=int(seed),
    )


def canonical_target_indices(
    targets: torch.Tensor,
    canonical_bank: torch.Tensor,
) -> torch.Tensor:
    target_rows = _quantized_rows(targets)
    bank_rows = _quantized_rows(canonical_bank)
    matches = (target_rows[:, None] == bank_rows[None]).all(dim=-1)
    counts = matches.sum(dim=1)
    if not torch.equal(counts, torch.ones_like(counts)):
        raise ValueError("every V29 target must match exactly one canonical bank image")
    return matches.to(torch.int64).argmax(dim=1)


class ConditionalVisualNaturalDataset(Dataset):
    """Render target-filtered 64-cell contexts; labels remain host pixels."""

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
            raise PermissionError("V29 training cannot open frozen writing")
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("unknown V29 data split")
        if not allowed_targets or length < 1:
            raise ValueError("V29 natural target set and length must be nonempty")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record, mode=render_config.script_views
            ):
                if len(writing) >= V29_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected:
            raise ValueError("V29 natural dataset is empty")
        self.records = selected
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
            start = rng.randint(0, len(writing) - V29_SEQUENCE_CELLS)
            target = writing[start + V29_CONTEXT_CELLS]
            if target in self.allowed_targets:
                selected = (record, script_view, writing, start, target)
                break
        if selected is None:
            raise RuntimeError("V29 could not sample an allowed natural target")
        record, script_view, writing, start, target = selected
        context_writing = writing[start : start + V29_CONTEXT_CELLS]
        fonts = V25_FONT_PARTITIONS[self.split]
        first_index = rng.randrange(len(fonts))
        second_index = (
            first_index + 1 + rng.randrange(len(fonts) - 1)
        ) % len(fonts)
        first = render_visual_cell_stream(
            context_writing,
            config=self.render_config,
            font_path=fonts[first_index],
            variant=rng.randrange(2**31),
        )
        second = render_visual_cell_stream(
            context_writing,
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
            "first_context": first,
            "second_context": second,
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


def conditional_visual_natural_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V29 natural batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V29_NATURAL_IMAGE_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def conditional_visual_natural_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V29_NATURAL_STUDENT_KEYS}
    for name, value in student.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V29 natural value {name!r} is not an image tensor")
        if tuple(value.shape[1:]) != (V29_CONTEXT_CELLS, 1, 32, 32):
            raise ValueError(f"V29 natural value {name!r} has invalid shape")
    return student


def conditional_visual_render_config_payload(
    config: ConditionalVisualRenderConfig,
) -> dict[str, Any]:
    return asdict(config)


def conditional_visual_candidate_bank_receipt(
    bank: ConditionalVisualCandidateBank,
    *,
    include_host_forms: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "bank_size": bank.size,
        "views": V29_BANK_VIEWS,
        "font_paths": list(bank.font_paths),
        "seed": bank.seed,
        "images_sha256": _tensor_sha256(bank.images),
        "canonical_sha256": _tensor_sha256(bank.canonical),
        "ordered_canonical_row_sha256": [
            _tensor_sha256(image) for image in bank.canonical
        ],
        "student_receives_images_only": True,
        "model_state_contains_bank": False,
        "checkpoint_contains_bank": False,
        "inference_requires_bank": False,
    }
    if include_host_forms:
        output["host_forms"] = "".join(bank.forms)
        output["host_counts"] = list(bank.counts)
    return output


def conditional_visual_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": "conditional-visual-density-ratio-v29",
        "natural_context_shape": [V29_CONTEXT_CELLS, 1, 32, 32],
        "candidate_shape": [1, 32, 32],
        "student_natural_keys": list(V29_NATURAL_STUDENT_KEYS),
        "input_is_continuous_image_stream": True,
        "canonical_identity_derived_from_exact_pixels": True,
        "canonical_indices_are_temporary_loss_only": True,
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
    "ConditionalVisualNaturalDataset",
    "ConditionalVisualRenderConfig",
    "V29_BANK_VIEWS",
    "V29_CONTEXT_CELLS",
    "V29_NATURAL_IMAGE_KEYS",
    "V29_NATURAL_STUDENT_KEYS",
    "V29_SEQUENCE_CELLS",
    "build_v29_candidate_bank",
    "build_v29_candidate_statistics",
    "canonical_target_indices",
    "conditional_visual_candidate_bank_receipt",
    "conditional_visual_data_boundary_receipt",
    "conditional_visual_natural_collate",
    "conditional_visual_natural_student_batch",
    "conditional_visual_render_config_payload",
]

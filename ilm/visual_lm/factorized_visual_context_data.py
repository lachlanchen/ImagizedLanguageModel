from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    V25_DEVELOPMENT_FONTS,
    V25_FONT_PARTITIONS,
    script_variants,
    visual_cell_partition,
    render_visual_cell_stream,
)


V26_CONTEXT_CELLS = 64
V26_FUTURE_CELLS = 8
V26_SEQUENCE_CELLS = V26_CONTEXT_CELLS + V26_FUTURE_CELLS
V26_HORIZONS = (1, 2, 4, 8)
V26_NATURAL_STUDENT_KEYS = (
    "context",
    "future",
    "reference_context",
    "reference_future",
)
V26_PAIR_STUDENT_KEYS = (
    "context_a",
    "target_a",
    "reference_context_a",
    "reference_target_a",
    "context_b",
    "target_b",
    "reference_context_b",
    "reference_target_b",
)


@dataclass(frozen=True)
class FactorizedVisualRenderConfig:
    cell_size: int = 32
    minimum_font_size: int = 24
    maximum_font_size: int = 28
    augment: bool = True
    script_views: str = "original+simplified"

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V26 fixes visual cells to 32x32")
        if not 8 <= self.minimum_font_size <= self.maximum_font_size <= 32:
            raise ValueError("V26 font sizes must fit inside a cell")
        if self.script_views not in {"original", "original+simplified"}:
            raise ValueError("unknown V26 script-view mode")


@dataclass(frozen=True)
class FactorizedVisualWindow:
    identifier: str
    script_view: str
    context: str
    future: str

    def __post_init__(self) -> None:
        if len(self.context) != V26_CONTEXT_CELLS:
            raise ValueError("V26 context must contain 64 visible cells")
        if len(self.future) != V26_FUTURE_CELLS:
            raise ValueError("V26 future must contain eight visible cells")

    @property
    def target(self) -> str:
        return self.future[0]


@dataclass(frozen=True)
class FactorizedVisualSuffixPair:
    suffix_cells: int
    identifier_a: str
    script_view_a: str
    context_a: str
    target_a: str
    identifier_b: str
    script_view_b: str
    context_b: str
    target_b: str

    def __post_init__(self) -> None:
        if not 1 <= self.suffix_cells <= V26_CONTEXT_CELLS:
            raise ValueError("V26 pair suffix length is invalid")
        if len(self.context_a) != 64 or len(self.context_b) != 64:
            raise ValueError("V26 pair contexts must contain 64 cells")
        if self.target_a == self.target_b:
            raise ValueError("V26 pair targets must differ")
        if self.context_a[-self.suffix_cells :] != self.context_b[-self.suffix_cells :]:
            raise ValueError("V26 pair suffixes must match exactly")

    @property
    def suffix(self) -> str:
        return self.context_a[-self.suffix_cells :]


def _candidate_priority(seed: int, pair: FactorizedVisualSuffixPair) -> bytes:
    payload = "\0".join(
        (
            str(seed),
            pair.suffix,
            pair.identifier_a,
            pair.identifier_b,
            pair.target_a,
            pair.target_b,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _iter_endpoint_windows(
    records: Sequence[VisualGrammarRecord],
    *,
    split: str,
    script_views_mode: str,
    future_cells: int,
) -> Iterable[tuple[VisualGrammarRecord, str, str, int]]:
    for record in records:
        if visual_cell_partition(record.identifier) != split:
            continue
        for script_view, writing in script_variants(
            record, mode=script_views_mode
        ):
            for target_offset in range(64, len(writing) - future_cells + 1):
                yield record, script_view, writing, target_offset


def build_factorized_suffix_pairs(
    records: Sequence[VisualGrammarRecord],
    *,
    split: str,
    suffix_cells: int,
    count: int,
    seed: int,
    require_different_identifiers: bool,
    allowed_targets: set[str] | frozenset[str] | None = None,
    script_views_mode: str = "original+simplified",
) -> tuple[FactorizedVisualSuffixPair, ...]:
    """Build deterministic host-side pairs; no strings reach the student."""

    if split == "frozen":
        raise PermissionError("V26 pair construction cannot open frozen writing")
    if count < 1 or not 1 <= suffix_cells <= 16:
        raise ValueError("V26 pair count or suffix length is invalid")
    waiting: dict[str, list[tuple[str, str, str, str]]] = {}
    completed: dict[str, FactorizedVisualSuffixPair] = {}
    for record, script_view, writing, target_offset in _iter_endpoint_windows(
        records,
        split=split,
        script_views_mode=script_views_mode,
        future_cells=1,
    ):
        target = writing[target_offset]
        if allowed_targets is not None and target not in allowed_targets:
            continue
        context = writing[target_offset - 64 : target_offset]
        suffix = context[-suffix_cells:]
        if suffix in completed:
            continue
        candidate = (record.identifier, script_view, context, target)
        bucket = waiting.setdefault(suffix, [])
        matched = None
        for previous in bucket:
            different_record = previous[0] != record.identifier
            if previous[3] != target and (
                different_record or not require_different_identifiers
            ):
                matched = previous
                break
        if matched is not None:
            pair = FactorizedVisualSuffixPair(
                suffix_cells=suffix_cells,
                identifier_a=matched[0],
                script_view_a=matched[1],
                context_a=matched[2],
                target_a=matched[3],
                identifier_b=record.identifier,
                script_view_b=script_view,
                context_b=context,
                target_b=target,
            )
            completed[suffix] = pair
            del waiting[suffix]
            continue
        signature = (record.identifier, target)
        if signature not in {(item[0], item[3]) for item in bucket}:
            bucket.append(candidate)
            if len(bucket) > 8:
                del bucket[0]
    ordered = sorted(
        completed.values(), key=lambda pair: _candidate_priority(seed, pair)
    )
    if len(ordered) < count:
        raise ValueError(
            f"V26 found {len(ordered)} suffix-{suffix_cells} pairs, needs {count}"
        )
    return tuple(ordered[:count])


class FactorizedVisualNaturalDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        split: str,
        render_config: FactorizedVisualRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if split == "frozen":
            raise PermissionError("V26 natural training cannot open frozen writing")
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("unknown V26 data split")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record, mode=render_config.script_views
            ):
                if len(writing) >= V26_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected or length < 1:
            raise ValueError("V26 natural dataset is empty")
        self.records = selected
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index * 104_729)
        record, script_view, writing = rng.choice(self.records)
        start = rng.randint(0, len(writing) - V26_SEQUENCE_CELLS)
        segment = writing[start : start + V26_SEQUENCE_CELLS]
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
        return {
            "context": first[:64],
            "future": first[64:],
            "reference_context": second[:64],
            "reference_future": second[64:],
            "metadata": {
                "identifier": record.identifier,
                "script_view": script_view,
                "offset": start,
            },
        }


class FactorizedVisualPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[FactorizedVisualSuffixPair],
        *,
        split: str,
        render_config: FactorizedVisualRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if split == "frozen":
            raise PermissionError("V26 pair dataset cannot open frozen writing")
        if not pairs or length < 1:
            raise ValueError("V26 pair dataset is empty")
        self.pairs = tuple(pairs)
        self.split = split
        self.render_config = render_config
        self.seed = int(seed)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def _render(
        self,
        writing: str,
        *,
        font: str,
        variant: int,
    ) -> torch.Tensor:
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
        second_index = (first_index + 1 + rng.randrange(len(fonts) - 1)) % len(fonts)
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
            raise RuntimeError("V26 shared suffix pixels are not exactly equal")
        if not torch.equal(second_a[64 - suffix : 64], second_b[64 - suffix : 64]):
            raise RuntimeError("V26 reference suffix pixels are not exactly equal")
        return {
            "context_a": first_a[:64],
            "target_a": first_a[64:65],
            "reference_context_a": second_a[:64],
            "reference_target_a": second_a[64:65],
            "context_b": first_b[:64],
            "target_b": first_b[64:65],
            "reference_context_b": second_b[:64],
            "reference_target_b": second_b[64:65],
            "metadata": {
                "identifier_a": pair.identifier_a,
                "identifier_b": pair.identifier_b,
                "script_view_a": pair.script_view_a,
                "script_view_b": pair.script_view_b,
                "target_a": pair.target_a,
                "target_b": pair.target_b,
                "suffix": pair.suffix,
                "suffix_cells": pair.suffix_cells,
            },
        }


class FactorizedVisualAuditDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[FactorizedVisualWindow],
        character_index: Mapping[str, int],
    ) -> None:
        if not windows:
            raise ValueError("V26 audit windows are empty")
        self.windows = tuple(windows)
        self.character_index = dict(character_index)
        self.render_config = FactorizedVisualRenderConfig(
            augment=False, script_views="original"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        first_font = V25_DEVELOPMENT_FONTS[index % len(V25_DEVELOPMENT_FONTS)]
        second_font = V25_DEVELOPMENT_FONTS[(index + 1) % len(V25_DEVELOPMENT_FONTS)]
        context = render_visual_cell_stream(
            window.context,
            config=self.render_config,
            font_path=first_font,
            variant=index,
        )
        future = render_visual_cell_stream(
            window.future,
            config=self.render_config,
            font_path=first_font,
            variant=index,
        )
        reference_future = render_visual_cell_stream(
            window.future,
            config=self.render_config,
            font_path=second_font,
            variant=index + 1,
        )
        return {
            "context": context,
            "future": future,
            "reference_future": reference_future,
            "target_index": self.character_index[window.target],
            "target_character": window.target,
            "last_character": window.context[-1],
            "identifier": window.identifier,
            "script_view": window.script_view,
        }


class FactorizedVisualPairAuditDataset(FactorizedVisualPairDataset):
    def __init__(
        self,
        pairs: Sequence[FactorizedVisualSuffixPair],
        *,
        character_index: Mapping[str, int],
    ) -> None:
        super().__init__(
            pairs,
            split="development",
            render_config=FactorizedVisualRenderConfig(
                augment=False, script_views="original"
            ),
            seed=20260911,
            length=len(pairs),
        )
        self.character_index = dict(character_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["target_index_a"] = self.character_index[item["metadata"]["target_a"]]
        item["target_index_b"] = self.character_index[item["metadata"]["target_b"]]
        return item


def build_factorized_audit_windows(
    records: Sequence[VisualGrammarRecord],
    *,
    allowed_targets: set[str] | frozenset[str],
    count: int,
    seed: int = 20260911,
    script_views_mode: str = "original+simplified",
) -> tuple[FactorizedVisualWindow, ...]:
    if count < 1:
        raise ValueError("V26 audit count must be positive")
    rng = random.Random(seed)
    reservoir: list[FactorizedVisualWindow] = []
    eligible = 0
    for record, script_view, writing, target_offset in _iter_endpoint_windows(
        records,
        split="development",
        script_views_mode=script_views_mode,
        future_cells=V26_FUTURE_CELLS,
    ):
        if writing[target_offset] not in allowed_targets:
            continue
        window = FactorizedVisualWindow(
            identifier=record.identifier,
            script_view=script_view,
            context=writing[target_offset - 64 : target_offset],
            future=writing[target_offset : target_offset + 8],
        )
        eligible += 1
        if len(reservoir) < count:
            reservoir.append(window)
        else:
            replacement = rng.randrange(eligible)
            if replacement < count:
                reservoir[replacement] = window
    if len(reservoir) != count:
        raise ValueError(f"V26 found {len(reservoir)} of {count} audit windows")
    rng.shuffle(reservoir)
    return tuple(reservoir)


def _collate_images(
    batch: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V26 batch")
    return {key: torch.stack([item[key] for item in batch]) for key in keys} | {
        "metadata": [item["metadata"] for item in batch]
    }


def factorized_visual_natural_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _collate_images(batch, V26_NATURAL_STUDENT_KEYS)


def factorized_visual_pair_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = _collate_images(batch, V26_PAIR_STUDENT_KEYS)
    for key in ("target_index_a", "target_index_b"):
        if key in batch[0]:
            output[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    return output


def factorized_visual_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V26 audit batch")
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "future": torch.stack([item["future"] for item in batch]),
        "reference_future": torch.stack(
            [item["reference_future"] for item in batch]
        ),
        "target_index": torch.tensor(
            [item["target_index"] for item in batch], dtype=torch.long
        ),
        "target_character": [item["target_character"] for item in batch],
        "last_character": [item["last_character"] for item in batch],
        "identifier": [item["identifier"] for item in batch],
        "script_view": [item["script_view"] for item in batch],
    }


def _assert_student_batch(
    batch: Mapping[str, Any], expected_keys: Sequence[str]
) -> None:
    if set(batch) != set(expected_keys):
        raise ValueError("V26 student batch has unregistered values")
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V26 student value {key!r} is not a floating tensor")
        if value.ndim != 5 or tuple(value.shape[2:]) != (1, 32, 32):
            raise ValueError(f"V26 student value {key!r} is not an image stream")


def factorized_visual_natural_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V26_NATURAL_STUDENT_KEYS}
    _assert_student_batch(student, V26_NATURAL_STUDENT_KEYS)
    return student


def factorized_visual_pair_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V26_PAIR_STUDENT_KEYS}
    _assert_student_batch(student, V26_PAIR_STUDENT_KEYS)
    return student


def factorized_visual_render_config_payload(
    config: FactorizedVisualRenderConfig,
) -> dict[str, Any]:
    return asdict(config)


def factorized_visual_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": "factorized-visual-context-v26",
        "natural_sample_shapes": {
            "context": [64, 1, 32, 32],
            "future": [8, 1, 32, 32],
        },
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

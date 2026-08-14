from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .factorized_visual_context_data import FactorizedVisualSuffixPair
from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    V25_TRAIN_FONTS,
    VisualCellRenderConfig,
    render_visual_cell_stream,
    script_variants,
    visual_cell_partition,
)
from .visual_cell_eval_data import (
    VisualCellAuditWindow,
    VisualCharacterStatistics,
)


V42_ARCHITECTURE = "canonical-glyph-language-v42"
V42_CONTEXT_CELLS = 64
V42_SEQUENCE_CELLS = V42_CONTEXT_CELLS + 1
V42_CANONICAL_FONT = V25_TRAIN_FONTS[0]
V42_STUDENT_KEYS = ("context", "target")


@dataclass(frozen=True)
class CanonicalGlyphRenderConfig:
    cell_size: int = 32
    font_size: int = 26
    font_path: str = V42_CANONICAL_FONT
    script_views: str = "original+simplified"

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V42 fixes canonical glyph cells to 32x32")
        if not 8 <= self.font_size <= 32:
            raise ValueError("V42 font size must fit inside one cell")
        if self.font_path != V42_CANONICAL_FONT:
            raise ValueError("V42 fixes one canonical Noto Sans CJK font")
        if self.script_views not in {"original", "original+simplified"}:
            raise ValueError("unknown V42 script-view mode")

    def visual_cell_config(self) -> VisualCellRenderConfig:
        return VisualCellRenderConfig(
            cell_size=self.cell_size,
            sequence_cells=V42_SEQUENCE_CELLS,
            minimum_font_size=self.font_size,
            maximum_font_size=self.font_size,
            augment=False,
            script_views=self.script_views,
        )


class CanonicalGlyphLanguageDataset(Dataset[dict[str, Any]]):
    """Prepare canonical Chinese writing and expose images only to the student."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        split: str,
        render_config: CanonicalGlyphRenderConfig,
        seed: int,
        length: int,
        expose_evaluation_labels: bool = False,
    ) -> None:
        if split not in {"train", "development"}:
            raise ValueError("V42 dataset split must be train or development")
        if length < 1:
            raise ValueError("V42 dataset length must be positive")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record,
                mode=render_config.script_views,
            ):
                if len(writing) >= V42_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected:
            raise ValueError(f"V42 has no usable canonical streams for {split}")
        self.records = tuple(selected)
        self.split = split
        self.render_config = render_config
        self.cell_config = render_config.visual_cell_config()
        self.seed = int(seed)
        self.length = int(length)
        self.expose_evaluation_labels = bool(expose_evaluation_labels)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        rng = random.Random(
            self.seed + self.epoch * 10_000_019 + index * 104_729
        )
        record, script_view, writing = self.records[
            rng.randrange(len(self.records))
        ]
        start = rng.randrange(len(writing) - V42_SEQUENCE_CELLS + 1)
        segment = writing[start : start + V42_SEQUENCE_CELLS]
        cells = render_visual_cell_stream(
            segment,
            config=self.cell_config,
            font_path=self.render_config.font_path,
            variant=0,
        )
        metadata: dict[str, Any] = {
            "identifier": record.identifier,
            "source": record.source,
            "rights": record.rights,
            "script_view": script_view,
            "offset": start,
        }
        if self.expose_evaluation_labels:
            metadata["context_characters"] = segment[:-1]
            metadata["target_characters"] = segment[1:]
        return {
            "context": cells[:-1],
            "target": cells[1:],
            "metadata": metadata,
        }


class CanonicalGlyphAuditDataset(Dataset[dict[str, Any]]):
    """Render fixed development windows; labels remain evaluator-side."""

    def __init__(
        self,
        windows: Sequence[VisualCellAuditWindow],
        statistics: VisualCharacterStatistics,
        *,
        render_config: CanonicalGlyphRenderConfig,
    ) -> None:
        if not windows:
            raise ValueError("V42 audit requires development windows")
        self.windows = tuple(windows)
        self.statistics = statistics
        self.character_index = statistics.index
        self.render_config = render_config
        self.cell_config = render_config.visual_cell_config()

    def __len__(self) -> int:
        return len(self.windows)

    def _render(self, writing: str) -> torch.Tensor:
        return render_visual_cell_stream(
            writing,
            config=self.cell_config,
            font_path=self.render_config.font_path,
            variant=0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        return {
            "context": self._render(window.context),
            "continuation": self._render(window.continuation),
            "target_index": self.character_index[window.target],
            "target_character": window.target,
            "last_character": window.last_context,
            "identifier": window.identifier,
            "script_view": window.script_view,
        }


class CanonicalGlyphPairAuditDataset(Dataset[dict[str, Any]]):
    """Render exact-suffix counterfactual pairs in the same canonical font."""

    def __init__(
        self,
        pairs: Sequence[FactorizedVisualSuffixPair],
        *,
        render_config: CanonicalGlyphRenderConfig,
    ) -> None:
        if not pairs:
            raise ValueError("V42 pair audit requires counterfactual pairs")
        self.pairs = tuple(pairs)
        self.render_config = render_config
        self.cell_config = render_config.visual_cell_config()

    def __len__(self) -> int:
        return len(self.pairs)

    def _render(self, writing: str) -> torch.Tensor:
        return render_visual_cell_stream(
            writing,
            config=self.cell_config,
            font_path=self.render_config.font_path,
            variant=0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        first = self._render(pair.context_a + pair.target_a)
        second = self._render(pair.context_b + pair.target_b)
        suffix = pair.suffix_cells
        if not torch.equal(first[64 - suffix : 64], second[64 - suffix : 64]):
            raise RuntimeError("V42 pair suffix pixels are not exactly equal")
        return {
            "contexts": torch.stack((first[:64], second[:64])),
            "candidates": torch.stack((first[64], second[64])),
            "assignment": torch.arange(2, dtype=torch.long),
            "metadata": {
                "identifier_a": pair.identifier_a,
                "identifier_b": pair.identifier_b,
                "target_a": pair.target_a,
                "target_b": pair.target_b,
                "suffix": pair.suffix,
                "suffix_cells": suffix,
            },
        }


def canonical_glyph_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V42 batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V42_STUDENT_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def canonical_glyph_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V42 audit batch")
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "continuation": torch.stack([item["continuation"] for item in batch]),
        "target_index": torch.tensor(
            [item["target_index"] for item in batch], dtype=torch.long
        ),
        "target_character": [item["target_character"] for item in batch],
        "last_character": [item["last_character"] for item in batch],
        "identifier": [item["identifier"] for item in batch],
        "script_view": [item["script_view"] for item in batch],
    }


def canonical_glyph_pair_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V42 pair audit batch")
    return {
        "contexts": torch.stack([item["contexts"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch]),
        "assignment": torch.stack([item["assignment"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def canonical_glyph_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V42_STUDENT_KEYS}
    if set(student) != set(V42_STUDENT_KEYS):
        raise ValueError("V42 student batch has unregistered values")
    for name, value in student.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"V42 student value {name!r} is not a float image")
        if value.ndim != 5 or tuple(value.shape[2:]) != (1, 32, 32):
            raise ValueError(f"V42 student value {name!r} must be [B,T,1,32,32]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"V42 student value {name!r} is not finite")
    return student


def render_canonical_character_bank(
    statistics: VisualCharacterStatistics,
    *,
    render_config: CanonicalGlyphRenderConfig,
) -> torch.Tensor:
    return render_visual_cell_stream(
        "".join(statistics.characters),
        config=render_config.visual_cell_config(),
        font_path=render_config.font_path,
        variant=0,
    )


def canonical_glyph_render_config_payload(
    config: CanonicalGlyphRenderConfig,
) -> dict[str, Any]:
    return asdict(config)


def canonical_glyph_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V42_ARCHITECTURE,
        "student_keys": list(V42_STUDENT_KEYS),
        "context_shape": [V42_CONTEXT_CELLS, 1, 32, 32],
        "target_shape": [V42_CONTEXT_CELLS, 1, 32, 32],
        "one_canonical_font": True,
        "input_is_continuous_image_stream": True,
        "target_is_continuous_image_stream": True,
        "metadata_excluded_from_student": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .canonical_glyph_language_data import CanonicalGlyphRenderConfig
from .factorized_visual_context_data import FactorizedVisualSuffixPair
from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    iter_split_writing,
    render_visual_cell_stream,
    script_variants,
    visual_cell_partition,
)
from .visual_cell_eval_data import (
    VisualCellAuditWindow,
    VisualCharacterStatistics,
)


V48_ARCHITECTURE = "visual-future-block-language-v48"
V48_CONTEXT_CELLS = 64
V48_FUTURE_HORIZONS = 4
V48_SEQUENCE_CELLS = V48_CONTEXT_CELLS + V48_FUTURE_HORIZONS
V48_STUDENT_KEYS = ("context", "future_pixels")


class VisualFutureBlockLanguageDataset(Dataset[dict[str, Any]]):
    """Render 68-cell Chinese streams and expose only image tensors."""

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
            raise ValueError("V48 dataset split must be train or development")
        if length < 1:
            raise ValueError("V48 dataset length must be positive")
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_view, writing in script_variants(
                record,
                mode=render_config.script_views,
            ):
                if len(writing) >= V48_SEQUENCE_CELLS:
                    selected.append((record, script_view, writing))
        if not selected:
            raise ValueError(f"V48 has no usable visual streams for {split}")
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
        start = rng.randrange(len(writing) - V48_SEQUENCE_CELLS + 1)
        segment = writing[start : start + V48_SEQUENCE_CELLS]
        cells = render_visual_cell_stream(
            segment,
            config=self.cell_config,
            font_path=self.render_config.font_path,
            variant=0,
        )
        future_pixels = torch.stack(
            [
                cells[horizon : horizon + V48_CONTEXT_CELLS]
                for horizon in range(1, V48_FUTURE_HORIZONS + 1)
            ],
            dim=1,
        )
        metadata: dict[str, Any] = {
            "identifier": record.identifier,
            "source": record.source,
            "rights": record.rights,
            "script_view": script_view,
            "offset": start,
        }
        if self.expose_evaluation_labels:
            metadata["segment"] = segment
        return {
            "context": cells[:V48_CONTEXT_CELLS],
            "future_pixels": future_pixels,
            "metadata": metadata,
        }


def visual_future_block_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V48 batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V48_STUDENT_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def visual_future_block_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V48_STUDENT_KEYS}
    if set(student) != set(V48_STUDENT_KEYS):
        raise ValueError("V48 student batch has unregistered values")
    context = student["context"]
    future = student["future_pixels"]
    if not isinstance(context, torch.Tensor) or not context.is_floating_point():
        raise TypeError("V48 context is not a floating image tensor")
    if not isinstance(future, torch.Tensor) or not future.is_floating_point():
        raise TypeError("V48 future block is not a floating image tensor")
    if context.ndim != 5 or tuple(context.shape[1:]) != (64, 1, 32, 32):
        raise ValueError("V48 context must be [B,64,1,32,32]")
    if future.ndim != 6 or tuple(future.shape[1:]) != (
        64,
        4,
        1,
        32,
        32,
    ):
        raise ValueError("V48 future block must be [B,64,4,1,32,32]")
    if not bool(torch.isfinite(context).all()) or not bool(
        torch.isfinite(future).all()
    ):
        raise ValueError("V48 student images must be finite")
    return student


def build_four_future_audit_windows_v48(
    records: Sequence[VisualGrammarRecord],
    statistics: VisualCharacterStatistics,
    *,
    count: int,
    seed: int,
    script_views_mode: str = "original+simplified",
) -> tuple[tuple[VisualCellAuditWindow, ...], int]:
    """Reservoir-sample development windows with four bank-visible futures."""

    if count < 1:
        raise ValueError("V48 four-future audit count must be positive")
    bank = set(statistics.characters)
    rng = random.Random(seed)
    reservoir: list[VisualCellAuditWindow] = []
    eligible = 0
    for record, script_view, writing in iter_split_writing(
        records,
        split="development",
        script_views_mode=script_views_mode,
    ):
        for target_offset in range(
            V48_CONTEXT_CELLS,
            len(writing) - V48_FUTURE_HORIZONS + 1,
        ):
            future = writing[
                target_offset : target_offset + V48_FUTURE_HORIZONS
            ]
            if len(future) != V48_FUTURE_HORIZONS or any(
                character not in bank for character in future
            ):
                continue
            window = VisualCellAuditWindow(
                identifier=record.identifier,
                script_view=script_view,
                context=writing[
                    target_offset - V48_CONTEXT_CELLS : target_offset
                ],
                continuation=future,
            )
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(window)
                continue
            replacement = rng.randrange(eligible)
            if replacement < count:
                reservoir[replacement] = window
    if len(reservoir) != count:
        raise ValueError(
            f"V48 found {len(reservoir)} of {count} four-future windows"
        )
    rng.shuffle(reservoir)
    return tuple(reservoir), eligible


class VisualFutureBlockAuditDataset(Dataset[dict[str, Any]]):
    """Render fixed four-future windows with labels kept evaluator-side."""

    def __init__(
        self,
        windows: Sequence[VisualCellAuditWindow],
        statistics: VisualCharacterStatistics,
        *,
        render_config: CanonicalGlyphRenderConfig,
    ) -> None:
        if not windows:
            raise ValueError("V48 future audit requires windows")
        if any(len(window.continuation) != V48_FUTURE_HORIZONS for window in windows):
            raise ValueError("V48 future audit windows must contain four targets")
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
            "future_pixels": self._render(window.continuation),
            "target_indices": torch.tensor(
                [
                    self.character_index[character]
                    for character in window.continuation
                ],
                dtype=torch.long,
            ),
            "last_character": window.last_context,
            "target_characters": window.continuation,
            "identifier": window.identifier,
            "script_view": window.script_view,
        }


def visual_future_block_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V48 future audit batch")
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "future_pixels": torch.stack(
            [item["future_pixels"] for item in batch]
        ),
        "target_indices": torch.stack(
            [item["target_indices"] for item in batch]
        ),
        "last_character": [item["last_character"] for item in batch],
        "target_characters": [item["target_characters"] for item in batch],
        "identifier": [item["identifier"] for item in batch],
        "script_view": [item["script_view"] for item in batch],
    }


def visual_pair_digest_v48(
    pairs: Sequence[FactorizedVisualSuffixPair],
) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        for value in (
            pair.identifier_a,
            pair.script_view_a,
            pair.context_a,
            pair.target_a,
            pair.identifier_b,
            pair.script_view_b,
            pair.context_b,
            pair.target_b,
        ):
            digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def visual_future_block_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V48_ARCHITECTURE,
        "student_keys": list(V48_STUDENT_KEYS),
        "context_shape": [64, 1, 32, 32],
        "future_shape": [64, 4, 1, 32, 32],
        "one_canonical_font": True,
        "input_is_continuous_image_stream": True,
        "target_is_continuous_image_block": True,
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


__all__ = [
    "V48_ARCHITECTURE",
    "V48_CONTEXT_CELLS",
    "V48_FUTURE_HORIZONS",
    "V48_SEQUENCE_CELLS",
    "V48_STUDENT_KEYS",
    "VisualFutureBlockAuditDataset",
    "VisualFutureBlockLanguageDataset",
    "build_four_future_audit_windows_v48",
    "visual_future_block_audit_collate",
    "visual_future_block_collate",
    "visual_future_block_data_boundary_receipt",
    "visual_future_block_student_batch",
    "visual_pair_digest_v48",
]

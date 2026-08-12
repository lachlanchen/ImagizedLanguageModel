from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import (
    V25_DEVELOPMENT_FONTS,
    VisualCellRenderConfig,
    iter_split_writing,
    render_visual_cell_stream,
)


def is_han_character(character: str) -> bool:
    if len(character) != 1:
        return False
    codepoint = ord(character)
    return any(
        lower <= codepoint <= upper
        for lower, upper in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x323AF),
        )
    )


@dataclass(frozen=True)
class VisualCharacterStatistics:
    characters: tuple[str, ...]
    counts: tuple[int, ...]
    bigram_rows: dict[str, tuple[int, ...]]
    visible_character_count: int
    han_character_count: int

    def __post_init__(self) -> None:
        if not self.characters or len(self.characters) != len(self.counts):
            raise ValueError("visual character bank and counts must align")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("visual character bank must contain unique forms")
        width = len(self.characters)
        if any(len(row) != width for row in self.bigram_rows.values()):
            raise ValueError("bigram rows must align with the visual bank")

    @property
    def index(self) -> dict[str, int]:
        return {character: index for index, character in enumerate(self.characters)}


def build_visual_character_statistics(
    records: Sequence[VisualGrammarRecord],
    *,
    bank_size: int = 1_024,
    script_views_mode: str = "original+simplified",
) -> VisualCharacterStatistics:
    """Build host-only visual-bank labels and symbolic controls from training text."""

    if bank_size < 2:
        raise ValueError("visual audit bank must contain at least two forms")
    counts: Counter[str] = Counter()
    visible_count = 0
    for _, _, writing in iter_split_writing(
        records,
        split="train",
        script_views_mode=script_views_mode,
    ):
        visible_count += len(writing)
        counts.update(character for character in writing if is_han_character(character))
    ordered = sorted(counts, key=lambda character: (-counts[character], ord(character)))
    characters = tuple(ordered[:bank_size])
    if len(characters) < bank_size:
        raise ValueError(
            f"training corpus contains only {len(characters)} supported Han forms"
        )
    index = {character: position for position, character in enumerate(characters)}
    bigrams: dict[str, list[int]] = defaultdict(lambda: [0] * len(characters))
    for _, _, writing in iter_split_writing(
        records,
        split="train",
        script_views_mode=script_views_mode,
    ):
        for previous, target in zip(writing, writing[1:]):
            target_index = index.get(target)
            if target_index is not None:
                bigrams[previous][target_index] += 1
    return VisualCharacterStatistics(
        characters=characters,
        counts=tuple(counts[character] for character in characters),
        bigram_rows={key: tuple(value) for key, value in sorted(bigrams.items())},
        visible_character_count=visible_count,
        han_character_count=sum(counts.values()),
    )


@dataclass(frozen=True)
class VisualCellAuditWindow:
    identifier: str
    script_view: str
    context: str
    continuation: str

    def __post_init__(self) -> None:
        if len(self.context) != 64:
            raise ValueError("V25 audit context must contain 64 visible cells")
        if len(self.continuation) < 1:
            raise ValueError("V25 audit continuation cannot be empty")

    @property
    def target(self) -> str:
        return self.continuation[0]

    @property
    def last_context(self) -> str:
        return self.context[-1]


def build_visual_cell_audit_windows(
    records: Sequence[VisualGrammarRecord],
    statistics: VisualCharacterStatistics,
    *,
    count: int,
    continuation_cells: int = 16,
    seed: int = 20260831,
    script_views_mode: str = "original+simplified",
) -> tuple[VisualCellAuditWindow, ...]:
    """Deterministically reservoir-sample development windows without opening frozen."""

    if count < 1 or continuation_cells < 1:
        raise ValueError("audit count and continuation length must be positive")
    bank = set(statistics.characters)
    rng = random.Random(seed)
    reservoir: list[VisualCellAuditWindow] = []
    eligible = 0
    for record, script_view, writing in iter_split_writing(
        records,
        split="development",
        script_views_mode=script_views_mode,
    ):
        last_start = len(writing) - continuation_cells
        for target_offset in range(64, last_start + 1):
            if writing[target_offset] not in bank:
                continue
            window = VisualCellAuditWindow(
                identifier=record.identifier,
                script_view=script_view,
                context=writing[target_offset - 64 : target_offset],
                continuation=writing[
                    target_offset : target_offset + continuation_cells
                ],
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
            f"development split yielded {len(reservoir)} of {count} audit windows"
        )
    rng.shuffle(reservoir)
    return tuple(reservoir)


class VisualCellAuditDataset(Dataset):
    """Render development windows; labels remain evaluator-side metadata."""

    def __init__(
        self,
        windows: Sequence[VisualCellAuditWindow],
        statistics: VisualCharacterStatistics,
    ) -> None:
        if not windows:
            raise ValueError("visual-cell audit dataset cannot be empty")
        self.windows = tuple(windows)
        self.statistics = statistics
        self.character_index = statistics.index
        self.render_config = VisualCellRenderConfig(
            augment=False,
            script_views="original",
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        first_font = V25_DEVELOPMENT_FONTS[index % len(V25_DEVELOPMENT_FONTS)]
        second_font = V25_DEVELOPMENT_FONTS[
            (index + 1) % len(V25_DEVELOPMENT_FONTS)
        ]
        context = render_visual_cell_stream(
            window.context,
            config=self.render_config,
            font_path=first_font,
            variant=index,
        )
        continuation = render_visual_cell_stream(
            window.continuation,
            config=self.render_config,
            font_path=first_font,
            variant=index,
        )
        reference = render_visual_cell_stream(
            window.continuation,
            config=self.render_config,
            font_path=second_font,
            variant=index + 1,
        )
        return {
            "context": context,
            "continuation": continuation,
            "reference_continuation": reference,
            "target_index": self.character_index[window.target],
            "target_character": window.target,
            "last_character": window.last_context,
            "identifier": window.identifier,
            "script_view": window.script_view,
        }


def visual_cell_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty visual-cell audit batch")
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "continuation": torch.stack([item["continuation"] for item in batch]),
        "reference_continuation": torch.stack(
            [item["reference_continuation"] for item in batch]
        ),
        "target_index": torch.tensor(
            [item["target_index"] for item in batch], dtype=torch.long
        ),
        "target_character": [item["target_character"] for item in batch],
        "last_character": [item["last_character"] for item in batch],
        "identifier": [item["identifier"] for item in batch],
        "script_view": [item["script_view"] for item in batch],
    }


def render_visual_character_bank(
    statistics: VisualCharacterStatistics,
) -> torch.Tensor:
    config = VisualCellRenderConfig(augment=False, script_views="original")
    views = [
        render_visual_cell_stream(
            "".join(statistics.characters),
            config=config,
            font_path=font,
            variant=view,
        )
        for view, font in enumerate(V25_DEVELOPMENT_FONTS)
    ]
    return torch.stack(views, dim=1)


def visual_character_statistics_receipt(
    statistics: VisualCharacterStatistics,
) -> dict[str, Any]:
    return {
        "bank_size": len(statistics.characters),
        "bank_characters": "".join(statistics.characters),
        "bank_counts": list(statistics.counts),
        "bigram_conditioning_forms": len(statistics.bigram_rows),
        "visible_character_count": statistics.visible_character_count,
        "han_character_count": statistics.han_character_count,
        "student_receives_bank": False,
        "student_receives_labels": False,
        "evaluator_only": True,
    }

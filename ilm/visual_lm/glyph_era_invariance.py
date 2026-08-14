from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .continuous_glyph_codec_data import (
    HistoricGlyphRecord,
    historic_character_partition,
)


@dataclass(frozen=True)
class CrossEraGlyphPair:
    character: str
    anchor_index: int
    positive_index: int
    anchor_stage: str
    positive_stage: str

    def __post_init__(self) -> None:
        if not self.character:
            raise ValueError("cross-era pair requires a character family")
        if min(self.anchor_index, self.positive_index) < 0:
            raise ValueError("cross-era pair indices must be nonnegative")
        if self.anchor_index == self.positive_index:
            raise ValueError("cross-era pair must use distinct glyphs")
        if self.anchor_stage == self.positive_stage:
            raise ValueError("cross-era pair must cross stage labels")


def select_cross_era_glyph_pairs(
    records: Sequence[HistoricGlyphRecord],
    *,
    split: str,
    seed: int,
    maximum_families: int = 0,
) -> tuple[CrossEraGlyphPair, ...]:
    if split not in {"train", "development", "sealed"}:
        raise ValueError("cross-era audit split is invalid")
    if maximum_families < 0:
        raise ValueError("maximum families cannot be negative")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if historic_character_partition(record.character) == split:
            grouped[record.character].append(index)
    families = sorted(
        character
        for character, indices in grouped.items()
        if len({records[index].stage for index in indices}) >= 2
    )
    rng = random.Random(int(seed))
    rng.shuffle(families)
    if maximum_families:
        families = families[:maximum_families]
    pairs: list[CrossEraGlyphPair] = []
    for character in families:
        indices = grouped[character]
        anchor_index = indices[rng.randrange(len(indices))]
        alternatives = [
            index
            for index in indices
            if records[index].stage != records[anchor_index].stage
        ]
        positive_index = alternatives[rng.randrange(len(alternatives))]
        pairs.append(
            CrossEraGlyphPair(
                character=character,
                anchor_index=anchor_index,
                positive_index=positive_index,
                anchor_stage=records[anchor_index].stage,
                positive_stage=records[positive_index].stage,
            )
        )
    if len(pairs) < 2:
        raise ValueError("cross-era audit requires at least two families")
    return tuple(pairs)


def cross_era_pair_sha256(pairs: Sequence[CrossEraGlyphPair]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        value = (
            f"{pair.character}\0{pair.anchor_index}\0{pair.positive_index}\0"
            f"{pair.anchor_stage}\0{pair.positive_stage}\n"
        )
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def cross_era_retrieval_metrics(
    anchor_states: torch.Tensor,
    positive_states: torch.Tensor,
) -> dict[str, float | int]:
    if anchor_states.ndim != 2 or positive_states.shape != anchor_states.shape:
        raise ValueError("cross-era states must be aligned matrices")
    if len(anchor_states) < 2 or not torch.is_floating_point(anchor_states):
        raise ValueError("cross-era audit requires floating states from two families")
    if not torch.is_floating_point(positive_states):
        raise TypeError("cross-era positive states must be floating point")
    if not bool(
        torch.isfinite(anchor_states).all() and torch.isfinite(positive_states).all()
    ):
        raise FloatingPointError("cross-era states must be finite")
    anchors = F.normalize(anchor_states.float(), dim=-1)
    positives = F.normalize(positive_states.float(), dim=-1)
    similarities = anchors @ positives.T
    labels = torch.arange(len(anchors), device=similarities.device)
    paired = similarities.diagonal()
    ranks = (similarities > paired.unsqueeze(1)).sum(dim=1) + 1
    argmax = similarities.argmax(dim=1)
    cyclic = (anchors * positives.roll(1, dims=0)).sum(dim=-1)
    return {
        "families": len(anchors),
        "candidates": len(positives),
        "chance_top1": 1.0 / len(positives),
        "argmax_top1": float((argmax == labels).float().mean()),
        "rank_top1": float((ranks == 1).float().mean()),
        "top5": float((ranks <= min(5, len(positives))).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
        "paired_cosine_mean": float(paired.mean()),
        "paired_cosine_median": float(paired.median()),
        "cyclic_cosine_mean": float(cyclic.mean()),
        "paired_beats_cyclic": float((paired > cyclic).float().mean()),
    }


__all__ = [
    "CrossEraGlyphPair",
    "cross_era_pair_sha256",
    "cross_era_retrieval_metrics",
    "select_cross_era_glyph_pairs",
]

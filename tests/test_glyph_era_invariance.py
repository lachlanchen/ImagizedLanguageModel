from __future__ import annotations

import torch

from ilm.visual_lm.continuous_glyph_codec_data import (
    HistoricGlyphRecord,
    historic_character_partition,
)
from ilm.visual_lm.glyph_era_invariance import (
    cross_era_pair_sha256,
    cross_era_retrieval_metrics,
    select_cross_era_glyph_pairs,
)


def record(character: str, stage: str, label: str) -> HistoricGlyphRecord:
    return HistoricGlyphRecord(
        character=character,
        stage=stage,
        label=label,
        local_path=f"data/{label}.svg",
    )


def test_cross_era_pair_selection_is_deterministic_and_crosses_stages() -> None:
    records: list[HistoricGlyphRecord] = []
    candidate = 0x4E00
    while len({item.character for item in records}) < 8:
        character = chr(candidate)
        candidate += 1
        records.extend(
            (
                record(character, "oracle", f"J{candidate}"),
                record(character, "bronze", f"B{candidate}"),
            )
        )
    split = next(
        value
        for value in ("train", "development", "sealed")
        if sum(
            historic_character_partition(character) == value
            for character in {item.character for item in records}
        )
        >= 2
    )

    first = select_cross_era_glyph_pairs(records, split=split, seed=40)
    second = select_cross_era_glyph_pairs(records, split=split, seed=40)

    assert first == second
    assert cross_era_pair_sha256(first) == cross_era_pair_sha256(second)
    assert all(pair.anchor_stage != pair.positive_stage for pair in first)
    assert all(
        records[pair.anchor_index].character
        == records[pair.positive_index].character
        == pair.character
        for pair in first
    )


def test_cross_era_retrieval_recovers_aligned_continuous_states() -> None:
    states = torch.eye(8)
    metrics = cross_era_retrieval_metrics(states, states + 0.001)

    assert metrics["families"] == 8
    assert metrics["argmax_top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["paired_cosine_mean"] > 0.999

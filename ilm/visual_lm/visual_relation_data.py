from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import RetinalRenderConfig, VisualGrammarRecord
from .saccade_data import render_glyph_fovea
from .visual_binding_data import (
    CANONICAL_TARGET_VARIANT,
    LABEL_PAIRS,
    OPERATIONS,
    binding_partition_receipt,
    build_binding_character_bank,
    noncanonical_variant,
    split_binding_characters,
)


PARTITION_SALT = "visual-relation-circuit-v23"
HELD_OUT_COMBINATIONS = frozenset({("异", "天地"), ("同", "左右")})


def build_relation_character_bank(
    records: Sequence[VisualGrammarRecord],
    *,
    bank_size: int = 1_024,
) -> list[str]:
    return build_binding_character_bank(records, bank_size=bank_size)


def split_relation_characters(
    characters: Sequence[str],
    *,
    salt: str = PARTITION_SALT,
) -> dict[str, list[str]]:
    return split_binding_characters(characters, salt=salt)


def relation_partition_receipt(
    partitions: dict[str, Sequence[str]],
    *,
    salt: str = PARTITION_SALT,
) -> dict[str, Any]:
    return binding_partition_receipt(partitions, salt=salt)


@dataclass(frozen=True)
class VisualRelationEpisodeConfig:
    fovea_size: int = 32
    source_font_size: int = 25
    source_minimum_font_size: int = 21
    target_font_size: int = 25
    canonical_target_variant: int = CANONICAL_TARGET_VARIANT
    development_heldout_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("relation fovea size must be a multiple of eight")
        if self.source_minimum_font_size > self.source_font_size:
            raise ValueError("source minimum font size exceeds source font size")
        if not 0.0 <= self.development_heldout_fraction <= 1.0:
            raise ValueError("development heldout fraction must be in [0, 1]")

    def source_render_config(self) -> RetinalRenderConfig:
        return RetinalRenderConfig(
            font_size=self.source_font_size,
            minimum_font_size=self.source_minimum_font_size,
            augment=True,
        )

    def target_render_config(self) -> RetinalRenderConfig:
        return RetinalRenderConfig(
            font_size=self.target_font_size,
            minimum_font_size=self.target_font_size,
            augment=False,
        )


def _combination_key(operation: str, label_pair: tuple[str, str]) -> tuple[str, str]:
    return operation, "".join(label_pair)


def _allowed_combinations(*, held_out: bool) -> tuple[tuple[str, tuple[str, str]], ...]:
    return tuple(
        (operation, pair)
        for operation in OPERATIONS
        for pair in LABEL_PAIRS
        if (_combination_key(operation, pair) in HELD_OUT_COMBINATIONS)
        == held_out
    )


class VisualRelationEpisodeDataset(Dataset):
    """Render image-only V23 relation episodes and causal counterfactuals."""

    def __init__(
        self,
        characters: Sequence[str],
        *,
        split: str,
        length: int,
        config: VisualRelationEpisodeConfig | None = None,
        seed: int = 20260827,
    ) -> None:
        if split not in {"train", "development"}:
            raise ValueError("relation dataset split must be train or development")
        if len(characters) < 2:
            raise ValueError("relation dataset needs at least two identities")
        if length < 1:
            raise ValueError("relation dataset length must be positive")
        self.characters = tuple(characters)
        self.split = split
        self.length = int(length)
        self.config = config or VisualRelationEpisodeConfig()
        self.seed = int(seed)
        self.epoch = 0
        self.source_config = self.config.source_render_config()
        self.target_config = self.config.target_render_config()
        self.seen_combinations = _allowed_combinations(held_out=False)
        self.heldout_combinations = _allowed_combinations(held_out=True)
        self._target_cache: dict[str, torch.Tensor] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _render_source(self, character: str, variant: int) -> torch.Tensor:
        return render_glyph_fovea(
            character,
            render_config=self.source_config,
            fovea_size=self.config.fovea_size,
            variant=variant,
        )

    def _render_target(self, character: str) -> torch.Tensor:
        cached = self._target_cache.get(character)
        if cached is None:
            cached = render_glyph_fovea(
                character,
                render_config=self.target_config,
                fovea_size=self.config.fovea_size,
                variant=self.config.canonical_target_variant,
            )
            self._target_cache[character] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(
            self.seed + self.epoch * 10_000_019 + index * 104_729
        )
        heldout_combination = (
            self.split == "development"
            and rng.random() < self.config.development_heldout_fraction
        )
        combinations = (
            self.heldout_combinations
            if heldout_combination
            else self.seen_combinations
        )
        operation, canonical_pair = rng.choice(combinations)
        labels = canonical_pair if rng.random() < 0.5 else canonical_pair[::-1]
        glyphs = tuple(rng.sample(self.characters, 2))
        query_index = rng.randrange(2)
        target_index = query_index if operation == "同" else 1 - query_index
        other_operation = "异" if operation == "同" else "同"

        variants = [
            noncanonical_variant(rng, self.config.canonical_target_variant)
            for _ in range(6)
        ]
        prompt = torch.stack(
            (
                self._render_source(labels[0], variants[0]),
                self._render_source(glyphs[0], variants[1]),
                self._render_source(labels[1], variants[2]),
                self._render_source(glyphs[1], variants[3]),
                self._render_source(operation, variants[4]),
                self._render_source(labels[query_index], variants[5]),
            )
        )

        query_counterfactual_prompt = prompt.clone()
        query_counterfactual_prompt[-1] = self._render_source(
            labels[1 - query_index], variants[5]
        )
        operation_counterfactual_prompt = prompt.clone()
        operation_counterfactual_prompt[4] = self._render_source(
            other_operation, variants[4]
        )
        pair_swapped_prompt = prompt[(2, 3, 0, 1, 4, 5), ...].clone()

        canonical_targets = tuple(self._render_target(glyph) for glyph in glyphs)
        other_target_index = 1 - target_index
        return {
            "prompt": prompt,
            "target": canonical_targets[target_index],
            "query_counterfactual_prompt": query_counterfactual_prompt,
            "query_counterfactual_target": canonical_targets[other_target_index],
            "operation_counterfactual_prompt": operation_counterfactual_prompt,
            "operation_counterfactual_target": canonical_targets[other_target_index],
            "pair_swapped_prompt": pair_swapped_prompt,
            "pair_swapped_target": canonical_targets[target_index],
            "oracle_reference": prompt[1 + target_index * 2],
            "counterfactual_oracle_reference": prompt[
                1 + other_target_index * 2
            ],
            "distractor_target": canonical_targets[other_target_index],
            "metadata": {
                "operation": operation,
                "labels": labels,
                "glyphs": glyphs,
                "query_index": query_index,
                "target_index": target_index,
                "target_character": glyphs[target_index],
                "counterfactual_target_character": glyphs[other_target_index],
                "heldout_combination": heldout_combination,
            },
        }


def visual_relation_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty visual relation batch")
    tensor_keys = (
        "prompt",
        "target",
        "query_counterfactual_prompt",
        "query_counterfactual_target",
        "operation_counterfactual_prompt",
        "operation_counterfactual_target",
        "pair_swapped_prompt",
        "pair_swapped_target",
        "oracle_reference",
        "counterfactual_oracle_reference",
        "distractor_target",
    )
    output = {
        key: torch.stack([item[key] for item in batch]) for key in tensor_keys
    }
    output["metadata"] = [item["metadata"] for item in batch]
    return output

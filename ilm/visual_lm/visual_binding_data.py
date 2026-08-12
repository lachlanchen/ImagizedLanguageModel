from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import (
    RETINAL_CJK_AVAILABLE_FONTS,
    RetinalRenderConfig,
    VisualGrammarRecord,
    retinal_character_supported,
)
from .saccade_data import render_glyph_fovea


PARTITION_SALT = "visual-binding-stream-v22"
LABEL_PAIRS = (("甲", "乙"), ("天", "地"), ("上", "下"), ("左", "右"))
OPERATIONS = ("同", "异")
HELD_OUT_COMBINATIONS = frozenset({("同", "天地"), ("异", "左右")})
MARKER_CHARACTERS = frozenset(
    character
    for pair in LABEL_PAIRS
    for character in pair
) | frozenset(OPERATIONS)
CANONICAL_TARGET_VARIANT = 1


def is_han_character(character: str) -> bool:
    if len(character) != 1:
        return False
    value = ord(character)
    return any(
        lower <= value <= upper
        for lower, upper in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x323AF),
        )
    )


def build_binding_character_bank(
    records: Sequence[VisualGrammarRecord],
    *,
    bank_size: int = 1_024,
) -> list[str]:
    if bank_size < 2:
        raise ValueError("visual binding bank needs at least two identities")
    counts = Counter(
        character
        for record in records
        for character in record.text
        if is_han_character(character)
        and character not in MARKER_CHARACTERS
        and retinal_character_supported(character)
    )
    characters = [character for character, _ in counts.most_common(bank_size)]
    if len(characters) != bank_size:
        raise ValueError(
            f"requested {bank_size} binding identities, corpus supplied "
            f"{len(characters)}"
        )
    return characters


def _partition_fraction(character: str, salt: str) -> float:
    digest = hashlib.sha256(
        salt.encode("utf-8") + b"\0" + character.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_binding_characters(
    characters: Sequence[str],
    *,
    salt: str = PARTITION_SALT,
) -> dict[str, list[str]]:
    output = {"train": [], "development": [], "frozen": []}
    for character in characters:
        fraction = _partition_fraction(character, salt)
        if fraction < 0.80:
            output["train"].append(character)
        elif fraction < 0.90:
            output["development"].append(character)
        else:
            output["frozen"].append(character)
    if len(output["train"]) < 2:
        raise ValueError("binding partition has fewer than two training identities")
    if len(output["development"]) < 2:
        raise ValueError("binding partition has fewer than two development identities")
    if len(output["frozen"]) < 2:
        raise ValueError("binding partition has fewer than two frozen identities")
    return output


def _identifier_sha256(characters: Sequence[str]) -> str:
    payload = "\n".join(sorted(characters)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def binding_partition_receipt(
    partitions: dict[str, Sequence[str]],
    *,
    salt: str = PARTITION_SALT,
) -> dict[str, Any]:
    return {
        "algorithm": "sha256(salt + NUL + character) first 64 bits",
        "salt": salt,
        "train_fraction": 0.80,
        "development_fraction": 0.10,
        "frozen_fraction": 0.10,
        "train_identities": len(partitions["train"]),
        "development_identities": len(partitions["development"]),
        "frozen_identities": len(partitions["frozen"]),
        "development_identifiers_sha256": _identifier_sha256(
            partitions["development"]
        ),
        "frozen_identifiers_sha256": _identifier_sha256(partitions["frozen"]),
        "frozen_images_instantiated": False,
    }


@dataclass(frozen=True)
class VisualBindingEpisodeConfig:
    fovea_size: int = 32
    source_font_size: int = 25
    source_minimum_font_size: int = 21
    target_font_size: int = 25
    canonical_target_variant: int = CANONICAL_TARGET_VARIANT
    development_heldout_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("binding fovea size must be a multiple of eight")
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


def _allowed_combinations(*, held_out: bool) -> list[tuple[str, tuple[str, str]]]:
    combinations = [
        (operation, label_pair)
        for operation in OPERATIONS
        for label_pair in LABEL_PAIRS
    ]
    return [
        combination
        for combination in combinations
        if (_combination_key(*combination) in HELD_OUT_COMBINATIONS) == held_out
    ]


def _noncanonical_variant(rng: random.Random, canonical_variant: int) -> int:
    variant = rng.randrange(2**31)
    font_count = len(RETINAL_CJK_AVAILABLE_FONTS)
    if font_count > 1:
        while variant % font_count == canonical_variant % font_count:
            variant += 1
    return variant


class VisualBindingEpisodeDataset(Dataset):
    """Render paired image-only visual binding prompts.

    Text and character identities are restricted to this offline renderer.
    Model-facing tensors contain only continuous image values.
    """

    def __init__(
        self,
        characters: Sequence[str],
        *,
        split: str,
        length: int,
        config: VisualBindingEpisodeConfig | None = None,
        seed: int = 20260823,
    ) -> None:
        if split not in {"train", "development"}:
            raise ValueError("binding dataset split must be train or development")
        if len(characters) < 2:
            raise ValueError("binding dataset needs at least two identities")
        if length < 1:
            raise ValueError("binding dataset length must be positive")
        self.characters = tuple(characters)
        self.split = split
        self.length = int(length)
        self.config = config or VisualBindingEpisodeConfig()
        self.seed = int(seed)
        self.epoch = 0
        self.source_config = self.config.source_render_config()
        self.target_config = self.config.target_render_config()
        self.seen_combinations = tuple(_allowed_combinations(held_out=False))
        self.heldout_combinations = tuple(_allowed_combinations(held_out=True))

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
        return render_glyph_fovea(
            character,
            render_config=self.target_config,
            fovea_size=self.config.fovea_size,
            variant=self.config.canonical_target_variant,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(
            self.seed + self.epoch * 10_000_019 + index * 104_729
        )
        if self.split == "train":
            heldout_combination = False
        else:
            heldout_combination = (
                rng.random() < self.config.development_heldout_fraction
            )
        combinations = (
            self.heldout_combinations
            if heldout_combination
            else self.seen_combinations
        )
        operation, canonical_pair = rng.choice(combinations)
        labels = canonical_pair if rng.random() < 0.5 else canonical_pair[::-1]
        glyph_a, glyph_b = rng.sample(self.characters, 2)
        glyphs = (glyph_a, glyph_b)
        query_index = rng.randrange(2)
        other_index = 1 - query_index
        target_index = query_index if operation == "同" else other_index
        counterfactual_query_index = other_index
        counterfactual_target_index = (
            counterfactual_query_index
            if operation == "同"
            else query_index
        )

        variants = [
            _noncanonical_variant(rng, self.config.canonical_target_variant)
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
        counterfactual_prompt = prompt.clone()
        counterfactual_prompt[-1] = self._render_source(
            labels[counterfactual_query_index],
            variants[5],
        )

        target = self._render_target(glyphs[target_index])
        counterfactual_target = self._render_target(
            glyphs[counterfactual_target_index]
        )
        return {
            "prompt": prompt,
            "target": target,
            "counterfactual_prompt": counterfactual_prompt,
            "counterfactual_target": counterfactual_target,
            "oracle_reference": prompt[1 + target_index * 2],
            "counterfactual_oracle_reference": prompt[
                1 + counterfactual_target_index * 2
            ],
            "distractor_target": self._render_target(glyphs[1 - target_index]),
            "counterfactual_distractor_target": self._render_target(
                glyphs[1 - counterfactual_target_index]
            ),
            "metadata": {
                "operation": operation,
                "labels": labels,
                "glyphs": glyphs,
                "target_character": glyphs[target_index],
                "counterfactual_target_character": glyphs[
                    counterfactual_target_index
                ],
                "target_prompt_index": 1 + target_index * 2,
                "counterfactual_target_prompt_index": (
                    1 + counterfactual_target_index * 2
                ),
                "heldout_combination": heldout_combination,
            },
        }


def visual_binding_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty visual binding batch")
    tensor_keys = (
        "prompt",
        "target",
        "counterfactual_prompt",
        "counterfactual_target",
        "oracle_reference",
        "counterfactual_oracle_reference",
        "distractor_target",
        "counterfactual_distractor_target",
    )
    output = {
        key: torch.stack([item[key] for item in batch])
        for key in tensor_keys
    }
    output["metadata"] = [item["metadata"] for item in batch]
    return output

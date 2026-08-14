from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from .canonical_glyph_flow_v43_data import (
    CanonicalGlyphPairTrainingDataset,
    canonical_glyph_pair_student_batch,
    canonical_glyph_pair_training_collate,
)
from .canonical_glyph_language_data import CanonicalGlyphRenderConfig
from .factorized_visual_context_data import FactorizedVisualSuffixPair


V47_PAIR_COUNT = 80_000
V47_PAIR_SUFFIX_CELLS = 4
V47_PAIR_SEED = 20264702
V47_PAIR_SEQUENCE_SHA256 = (
    "2f573c4c79deb9e2bf97c2b0af588a438c7a43280bd632c05c0ae477ec6918eb"
)


def pair_sequence_receipt_v47(
    pairs: Sequence[FactorizedVisualSuffixPair],
) -> dict[str, Any]:
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
    return {
        "count": len(pairs),
        "unique_suffixes": len({pair.suffix for pair in pairs}),
        "suffix_cells": V47_PAIR_SUFFIX_CELLS,
        "seed": V47_PAIR_SEED,
        "split": "train",
        "sha256": digest.hexdigest(),
        "candidate_columns_permuted": True,
        "consumed_without_replacement": True,
    }


def validate_frozen_pair_sequence_v47(
    pairs: Sequence[FactorizedVisualSuffixPair],
    *,
    strict: bool,
) -> dict[str, Any]:
    receipt = pair_sequence_receipt_v47(pairs)
    if receipt["count"] != receipt["unique_suffixes"]:
        raise ValueError("V47 pair sequence repeats a four-glyph suffix")
    if any(pair.suffix_cells != V47_PAIR_SUFFIX_CELLS for pair in pairs):
        raise ValueError("V47 requires four-cell pair suffixes")
    if strict and (
        receipt["count"] != V47_PAIR_COUNT
        or receipt["sha256"] != V47_PAIR_SEQUENCE_SHA256
    ):
        raise ValueError("V47 pair sequence differs from the frozen protocol")
    return receipt


class CodecSphericalPairTrainingDatasetV47(CanonicalGlyphPairTrainingDataset):
    """V47 pair renderer that forbids cycling or replacement."""

    def __init__(
        self,
        pairs: Sequence[FactorizedVisualSuffixPair],
        *,
        render_config: CanonicalGlyphRenderConfig,
        seed: int,
    ) -> None:
        if len(pairs) < 1:
            raise ValueError("V47 pair sequence cannot be empty")
        super().__init__(
            pairs,
            render_config=render_config,
            seed=seed,
            length=len(pairs),
        )


def codec_spherical_glyph_language_v47_data_boundary_receipt() -> dict[str, Any]:
    return {
        "pair_student_keys": ["contexts", "candidates", "assignment"],
        "pair_suffix_cells": V47_PAIR_SUFFIX_CELLS,
        "candidate_columns_permuted": True,
        "pairs_consumed_without_replacement": True,
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
    "CodecSphericalPairTrainingDatasetV47",
    "V47_PAIR_COUNT",
    "V47_PAIR_SEED",
    "V47_PAIR_SEQUENCE_SHA256",
    "V47_PAIR_SUFFIX_CELLS",
    "canonical_glyph_pair_student_batch",
    "canonical_glyph_pair_training_collate",
    "codec_spherical_glyph_language_v47_data_boundary_receipt",
    "pair_sequence_receipt_v47",
    "validate_frozen_pair_sequence_v47",
]

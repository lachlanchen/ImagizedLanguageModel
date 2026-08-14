from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .canonical_glyph_language_data import CanonicalGlyphRenderConfig
from .factorized_visual_context_data import FactorizedVisualSuffixPair
from .visual_cell_data import render_visual_cell_stream


V43_ARCHITECTURE = "canonical-glyph-flow-v43"
V43_PAIR_SUFFIX_CELLS = 4
V43_PAIR_STUDENT_KEYS = ("contexts", "candidates", "assignment")


class CanonicalGlyphPairTrainingDataset(Dataset[dict[str, Any]]):
    """Render train-only same-suffix pairs and hide all symbolic metadata."""

    def __init__(
        self,
        pairs: Sequence[FactorizedVisualSuffixPair],
        *,
        render_config: CanonicalGlyphRenderConfig,
        seed: int,
        length: int,
    ) -> None:
        if not pairs or length < 1:
            raise ValueError("V43 pair training data cannot be empty")
        if any(pair.suffix_cells != V43_PAIR_SUFFIX_CELLS for pair in pairs):
            raise ValueError("V43 fixes every training pair to a four-cell suffix")
        self.pairs = tuple(pairs)
        self.render_config = render_config
        self.cell_config = render_config.visual_cell_config()
        self.seed = int(seed)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def _render(self, writing: str) -> torch.Tensor:
        return render_visual_cell_stream(
            writing,
            config=self.cell_config,
            font_path=self.render_config.font_path,
            variant=0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        pair = self.pairs[index % len(self.pairs)]
        first = self._render(pair.context_a + pair.target_a)
        second = self._render(pair.context_b + pair.target_b)
        suffix = pair.suffix_cells
        if not torch.equal(first[64 - suffix : 64], second[64 - suffix : 64]):
            raise RuntimeError("V43 paired suffix pixels must be exactly equal")
        candidates = torch.stack((first[64], second[64]))
        rng = random.Random(self.seed + index * 104_729)
        if rng.randrange(2):
            permutation = torch.tensor((1, 0), dtype=torch.long)
            assignment = torch.tensor((1, 0), dtype=torch.long)
        else:
            permutation = torch.tensor((0, 1), dtype=torch.long)
            assignment = torch.tensor((0, 1), dtype=torch.long)
        return {
            "contexts": torch.stack((first[:64], second[:64])),
            "candidates": candidates[permutation],
            "assignment": assignment,
            "metadata": {
                "identifier_a": pair.identifier_a,
                "identifier_b": pair.identifier_b,
                "script_view_a": pair.script_view_a,
                "script_view_b": pair.script_view_b,
                "target_a": pair.target_a,
                "target_b": pair.target_b,
                "suffix": pair.suffix,
            },
        }


def canonical_glyph_pair_training_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V43 pair batch")
    return {
        "contexts": torch.stack([item["contexts"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch]),
        "assignment": torch.stack([item["assignment"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def canonical_glyph_pair_student_batch(
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V43_PAIR_STUDENT_KEYS}
    contexts = student["contexts"]
    candidates = student["candidates"]
    assignment = student["assignment"]
    if (
        not isinstance(contexts, torch.Tensor)
        or not contexts.is_floating_point()
        or contexts.ndim != 6
        or tuple(contexts.shape[1:]) != (2, 64, 1, 32, 32)
    ):
        raise ValueError("V43 pair contexts must be float [B,2,64,1,32,32]")
    if (
        not isinstance(candidates, torch.Tensor)
        or not candidates.is_floating_point()
        or candidates.ndim != 5
        or tuple(candidates.shape[1:]) != (2, 1, 32, 32)
    ):
        raise ValueError("V43 pair candidates must be float [B,2,1,32,32]")
    if (
        not isinstance(assignment, torch.Tensor)
        or assignment.dtype != torch.long
        or assignment.shape != contexts.shape[:2]
    ):
        raise ValueError("V43 pair assignment must be long [B,2]")
    if not bool(torch.isfinite(contexts).all() and torch.isfinite(candidates).all()):
        raise ValueError("V43 pair images must be finite")
    if not bool(((contexts >= 0.0) & (contexts <= 1.0)).all()):
        raise ValueError("V43 pair contexts must lie in [0,1]")
    if not bool(((candidates >= 0.0) & (candidates <= 1.0)).all()):
        raise ValueError("V43 pair candidates must lie in [0,1]")
    expected = torch.tensor((0, 1), dtype=torch.long, device=assignment.device)
    if not bool(assignment.sort(dim=1).values.eq(expected).all()):
        raise ValueError("V43 each pair assignment must be a permutation")
    return student


def canonical_glyph_flow_v43_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V43_ARCHITECTURE,
        "pair_student_keys": list(V43_PAIR_STUDENT_KEYS),
        "pair_suffix_cells": V43_PAIR_SUFFIX_CELLS,
        "candidate_columns_permuted": True,
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

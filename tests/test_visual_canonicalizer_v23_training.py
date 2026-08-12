from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from ilm.visual_lm.visual_relation_circuit import VisualCanonicalizer
from ilm.visual_lm.visual_relation_data import PARTITION_SALT
from scripts.train_visual_canonicalizer_v23 import (
    ARCHITECTURE,
    EXPECTED_PARAMETERS,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    _canonicalizer_pair,
    _require_fixed_arguments,
    canonicalizer_boundary_is_clean,
    canonicalizer_selection_gate_report,
    canonicalizer_selection_rank,
)


def fixed_args(*, smoke: bool = False) -> Namespace:
    return Namespace(
        partition_salt=PARTITION_SALT,
        smoke=smoke,
        **FIXED_MODEL_ARGUMENTS,
        **FIXED_LOSS_ARGUMENTS,
        **FIXED_OPTIMIZATION_ARGUMENTS,
        **FIXED_EVIDENCE_ARGUMENTS,
    )


def passing_metrics() -> dict[str, float]:
    return {
        "pixel_f1": 0.80,
        "identity_top1": 0.90,
        "target_cosine": 0.90,
        "raw_source_pixel_f1": 0.60,
        "source_shuffled_pixel_f1": 0.40,
        "source_shuffled_identity_top1": 0.10,
        "ink_fraction": 0.20,
        "identity_bank_identities": 109.0,
        "student_boundary_clean": 1.0,
        "frozen_images_instantiated": 0.0,
        "step": 400.0,
    }


def test_evidence_arguments_are_fixed_and_smoke_is_bounded() -> None:
    _require_fixed_arguments(fixed_args())
    changed = fixed_args()
    changed.batch_size = 32
    with pytest.raises(ValueError, match="batch-size"):
        _require_fixed_arguments(changed)

    smoke = fixed_args(smoke=True)
    smoke.maximum_steps = 20
    smoke.batch_size = 2
    _require_fixed_arguments(smoke)
    smoke.maximum_steps = 21
    with pytest.raises(ValueError, match="1--20"):
        _require_fixed_arguments(smoke)


def test_selection_gates_use_strict_preregistered_thresholds() -> None:
    metrics = passing_metrics()
    assert all(canonicalizer_selection_gate_report(metrics).values())

    metrics["pixel_f1"] = 0.72
    assert canonicalizer_selection_gate_report(metrics)["pixel_f1"] is False
    metrics = passing_metrics()
    metrics["raw_source_pixel_f1"] = 0.68
    assert (
        canonicalizer_selection_gate_report(metrics)[
            "raw_source_pixel_f1_gain"
        ]
        is False
    )


def test_selection_rank_prefers_quality_then_earlier_step() -> None:
    first = passing_metrics()
    later = dict(first, step=600.0)
    assert canonicalizer_selection_rank(first) > canonicalizer_selection_rank(later)
    later["pixel_f1"] = 0.81
    assert canonicalizer_selection_rank(later) > canonicalizer_selection_rank(first)


def test_boundary_rejects_symbolic_inputs() -> None:
    receipt = VisualCanonicalizer().boundary_receipt()
    assert receipt["architecture"] == ARCHITECTURE
    assert canonicalizer_boundary_is_clean(receipt)
    symbolic = dict(receipt)
    symbolic["uses_unicode_ids"] = True
    assert not canonicalizer_boundary_is_clean(symbolic)


def test_canonicalizer_pair_uses_both_visible_sources_in_matching_order() -> None:
    batch = {
        "oracle_reference": torch.full((2, 1, 32, 32), 1.0),
        "counterfactual_oracle_reference": torch.full((2, 1, 32, 32), 2.0),
        "target": torch.full((2, 1, 32, 32), 3.0),
        "query_counterfactual_target": torch.full((2, 1, 32, 32), 4.0),
        "metadata": [
            {"target_character": "甲", "counterfactual_target_character": "乙"},
            {"target_character": "丙", "counterfactual_target_character": "丁"},
        ],
    }
    source, target, characters = _canonicalizer_pair(batch)
    assert tuple(source.shape) == (4, 1, 32, 32)
    assert source[:, 0, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert target[:, 0, 0, 0].tolist() == [3.0, 3.0, 4.0, 4.0]
    assert characters == ["甲", "丙", "乙", "丁"]


def test_canonicalizer_parameter_budget_is_exact() -> None:
    model = VisualCanonicalizer()
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETERS
    )

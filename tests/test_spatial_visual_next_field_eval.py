from __future__ import annotations

import torch

from ilm.visual_lm.spatial_visual_next_field import (
    V30_GLOBAL_ROUTE,
    V30_SPATIAL_ROUTE,
    SpatialVisualNextFieldConfig,
    SpatialVisualNextFieldModel,
)
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from scripts.eval_spatial_visual_next_field_v30 import (
    SCORE_NAMES,
    evaluate_natural_language,
    evaluate_suffix_pairs,
    final_checkpoint_is_clean,
    student_boundary_is_clean,
    v30_gate_report,
)


def _model(route: str = V30_SPATIAL_ROUTE) -> SpatialVisualNextFieldModel:
    return SpatialVisualNextFieldModel(
        SpatialVisualNextFieldConfig(
            visual_dim=72,
            semantic_dim=72,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=24,
            semantic_hidden_dim=96,
            field_channels=72,
            decoder_hidden_channels=72,
            decoder_dropout=0.0,
            score_chunk_size=2,
            route_mode=route,
        )
    ).eval()


def _checkpoint(model: SpatialVisualNextFieldModel) -> dict:
    return {
        "architecture": "spatial-visual-next-field-v30",
        "route_mode": model.config.route_mode,
        "model": model.state_dict(),
        "optimizer": None,
        "rng_state": None,
        "resumable": False,
        "deployed_state_includes_training_candidate_images": False,
        "deployed_state_includes_training_form_labels": False,
        "candidate_bank_receipt": {
            "images_in_checkpoint": False,
            "forms_in_checkpoint": False,
            "inference_requires_bank": False,
        },
    }


def _pair_batch(batch: int = 2) -> dict[str, object]:
    suffix = torch.rand(batch, 4, 1, 32, 32)
    contexts = torch.cat(
        (
            torch.rand(batch, 2, 60, 1, 32, 32),
            suffix[:, None].expand(-1, 2, -1, -1, -1, -1),
        ),
        dim=2,
    )
    reference_suffix = torch.rand(batch, 4, 1, 32, 32)
    reference_contexts = torch.cat(
        (
            torch.rand(batch, 2, 60, 1, 32, 32),
            reference_suffix[:, None].expand(-1, 2, -1, -1, -1, -1),
        ),
        dim=2,
    )
    return {
        "contexts": contexts,
        "candidates": torch.rand(batch, 2, 1, 32, 32),
        "assignment": torch.tensor([[0, 1], [1, 0]]),
        "reference_contexts": reference_contexts,
        "reference_candidates": torch.rand(batch, 2, 1, 32, 32),
        "reference_assignment": torch.tensor([[1, 0], [0, 1]]),
        "metadata": [{"suffix_cells": 4} for _ in range(batch)],
    }


def test_v30_natural_audit_reports_spatial_intervention() -> None:
    model = _model()
    statistics = VisualCharacterStatistics(
        characters=("中", "文", "天"),
        counts=(9, 7, 5),
        bigram_rows={"中": ((1, 3),)},
        visible_character_count=30,
        han_character_count=21,
    )
    batch = {
        "context": torch.rand(2, 64, 1, 32, 32),
        "target_index": torch.tensor([0, 2]),
        "candidate_view": torch.tensor([1, 0]),
        "context_text": ["中" * 64, "文" * 64],
    }
    metrics = evaluate_natural_language(
        model,
        [batch],
        statistics,
        {},
        torch.rand(3, 2, 1, 32, 32),
        device=torch.device("cpu"),
        precision="fp32",
    )
    for name in SCORE_NAMES:
        assert 0.0 <= metrics[f"{name}_top1"] <= 1.0
        assert torch.isfinite(torch.tensor(metrics[f"{name}_target_log_probability"]))
    assert metrics["all_scores_finite"] == 1.0
    assert len(metrics["rendered_batch_sha256"]) == 64


def test_v30_pair_audit_checks_both_permutations_and_exact_suffix() -> None:
    metrics = evaluate_suffix_pairs(
        _model(),
        [_pair_batch()],
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert metrics["suffix_pixel_equality"] == 1.0
    assert metrics["suffix_score_row_max_error"] == 0.0
    for name in SCORE_NAMES:
        assert metrics[f"candidate_permutation_{name}_max_score_error"] < 1e-5
        assert metrics[f"candidate_permutation_{name}_accuracy_agreement"] == 1.0


def test_v30_global_route_is_spatial_permutation_invariant_in_audit() -> None:
    metrics = evaluate_suffix_pairs(
        _model(V30_GLOBAL_ROUTE),
        [_pair_batch()],
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert metrics["spatial_permutation_max_score_error"] == 0.0


def test_v30_checkpoint_and_student_boundary_are_clean() -> None:
    model = _model()
    checkpoint = _checkpoint(model)
    assert final_checkpoint_is_clean(checkpoint)
    assert student_boundary_is_clean(model, checkpoint)
    checkpoint["rng_state"] = {"torch": torch.zeros(1)}
    assert not final_checkpoint_is_clean(checkpoint)


def _gate_route(*, spatial: bool) -> dict:
    natural = {
        "candidate_cross_font_identity_top1": 0.96,
        "spatial_permutation_max_score_error": 2.0 if spatial else 0.0,
        "full_top1": 0.30 if spatial else 0.10,
        "suffix4_top1": 0.10,
        "shuffled_top1": 0.10,
        "unigram_top1": 0.05,
        "bigram_top1": 0.10,
        "full_target_log_probability": -2.0 if spatial else -2.2,
        "shuffled_target_log_probability": -2.2,
        "spatial_permuted_target_log_probability": -2.2,
        "bigram_target_log_probability": -3.0,
    }
    suffix = {
        "suffix_pixel_equality": 1.0,
        "suffix_score_row_max_error": 0.0,
        "spatial_permutation_max_score_error": 2.0 if spatial else 0.0,
        "full_arm_accuracy": 0.80 if spatial else 0.60,
        "full_both_correct_rate": 0.60 if spatial else 0.40,
        "full_minus_shuffled_arm_accuracy": 0.20,
        "full_minus_shuffled_mean_margin": 0.20,
        "full_minus_spatial_permuted_arm_accuracy": 0.20,
    }
    for name in SCORE_NAMES:
        suffix[f"candidate_permutation_{name}_max_score_error"] = 0.0
        suffix[f"candidate_permutation_{name}_accuracy_agreement"] = 1.0
    return {
        "natural": natural,
        "suffix4": suffix,
        "integrity": {
            "model_state_finite": True,
            "training_metrics_finite": True,
            "scores_finite": True,
            "student_boundary_clean": True,
            "final_checkpoint_clean": True,
            "candidate_independent_output_shape_clean": True,
            "parameter_cap_clean": True,
            "memory_cap_clean": True,
        },
    }


def test_v30_gate_report_requires_spatial_control_and_matched_gates() -> None:
    matched = {
        "initialized_parameter_states_exact": True,
        "final_parameter_counts_exact": True,
        "source_and_data_receipts_exact": True,
        "audit_windows_and_pixels_exact": True,
        "both_arms_completed_8000_finite_updates": True,
    }
    groups = v30_gate_report(
        _gate_route(spatial=True),
        _gate_route(spatial=False),
        matched,
        frozen_images_instantiated=False,
    )
    assert all(all(group.values()) for group in groups)
    matched["initialized_parameter_states_exact"] = False
    groups = v30_gate_report(
        _gate_route(spatial=True),
        _gate_route(spatial=False),
        matched,
        frozen_images_instantiated=False,
    )
    assert groups[2]["initialized_parameter_states_exact"] is False

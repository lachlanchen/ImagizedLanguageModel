from __future__ import annotations

import torch

from ilm.visual_lm.conditional_visual_field_flow import (
    V31_GLOBAL_ROUTE,
    V31_SPATIAL_ROUTE,
    ConditionalVisualFieldFlowConfig,
    ConditionalVisualFieldFlowModel,
    conditional_visual_field_flow_boundary_receipt,
)
from ilm.visual_lm.conditional_visual_field_flow_data import (
    conditional_visual_field_flow_data_boundary_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from scripts.eval_conditional_visual_field_flow_v31 import (
    PATH_SCORE_NAMES,
    SAMPLE_SCORE_NAMES,
    choose_device,
    evaluate_natural_language,
    evaluate_suffix_pairs,
    final_checkpoint_is_clean,
    student_boundary_is_clean,
    v31_gate_report,
)


def _model(route: str = V31_SPATIAL_ROUTE) -> ConditionalVisualFieldFlowModel:
    return ConditionalVisualFieldFlowModel(
        ConditionalVisualFieldFlowConfig(
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
            velocity_hidden_channels=72,
            velocity_blocks=1,
            velocity_dropout=0.0,
            time_embedding_dim=32,
            score_chunk_size=2,
            route_mode=route,
        )
    ).eval()


def _checkpoint(model: ConditionalVisualFieldFlowModel) -> dict:
    return {
        "architecture": "conditional-visual-field-flow-v31",
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
        "model_boundary": conditional_visual_field_flow_boundary_receipt(model.config),
        "data_boundary": conditional_visual_field_flow_data_boundary_receipt(),
    }


def _pair_batch(batch: int = 1) -> dict[str, object]:
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
        "assignment": torch.tensor([[0, 1]]).expand(batch, -1).clone(),
        "reference_contexts": reference_contexts,
        "reference_candidates": torch.rand(batch, 2, 1, 32, 32),
        "reference_assignment": torch.tensor([[1, 0]]).expand(batch, -1).clone(),
        "metadata": [{"suffix_cells": 4} for _ in range(batch)],
    }


def test_v31_natural_audit_reports_path_and_autonomous_samples() -> None:
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
    for name in PATH_SCORE_NAMES:
        assert 0.0 <= metrics[f"path_{name}_top1"] <= 1.0
    for name in SAMPLE_SCORE_NAMES:
        assert 0.0 <= metrics[f"sample_{name}_top1"] <= 1.0
    assert metrics["all_scores_and_samples_finite"] == 1.0
    assert metrics["candidate_independent_sample_shape_clean"] == 1.0
    assert len(metrics["sample_examples"]) == 2


def test_v31_pair_audit_checks_suffix_permutation_and_samples() -> None:
    metrics = evaluate_suffix_pairs(
        _model(),
        [_pair_batch()],
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert metrics["suffix_pixel_equality"] == 1.0
    assert metrics["suffix_path_score_row_max_error"] == 0.0
    for name in PATH_SCORE_NAMES:
        assert metrics[f"candidate_permutation_path_{name}_max_score_error"] < 1e-5
        assert metrics[f"candidate_permutation_path_{name}_accuracy_agreement"] == 1.0
    for name in SAMPLE_SCORE_NAMES:
        assert metrics[f"candidate_permutation_sample_{name}_max_score_error"] < 1e-5


def test_v31_global_route_is_exactly_spatially_invariant() -> None:
    metrics = evaluate_suffix_pairs(
        _model(V31_GLOBAL_ROUTE),
        [_pair_batch()],
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert metrics["path_spatial_permutation_max_score_error"] == 0.0
    assert metrics["sample_spatial_permutation_max_score_error"] == 0.0


def test_v31_checkpoint_and_student_boundary_are_clean() -> None:
    model = _model()
    checkpoint = _checkpoint(model)
    assert final_checkpoint_is_clean(checkpoint)
    assert student_boundary_is_clean(model, checkpoint)
    checkpoint["rng_state"] = {"torch": torch.zeros(1)}
    assert not final_checkpoint_is_clean(checkpoint)


def test_v31_evaluator_explicit_cuda_device_has_an_index() -> None:
    assert choose_device("cuda") == torch.device("cuda:0")
    assert choose_device("cuda:1") == torch.device("cuda:1")


def _gate_route(*, spatial: bool) -> dict:
    natural = {
        "candidate_cross_font_identity_top1": 0.96,
        "path_spatial_permutation_max_score_error": 2.0 if spatial else 0.0,
        "sample_spatial_permutation_max_score_error": 2.0 if spatial else 0.0,
        "path_full_top1": 0.30 if spatial else 0.10,
        "path_suffix4_top1": 0.10,
        "path_shuffled_top1": 0.10,
        "unigram_top1": 0.05,
        "bigram_top1": 0.10,
        "path_full_target_log_probability": -2.0 if spatial else -2.2,
        "path_shuffled_target_log_probability": -2.2,
        "path_spatial_permuted_target_log_probability": -2.2,
        "bigram_target_log_probability": -3.0,
        "sample_full_top1": 0.10,
        "sample_shuffled_top1": 0.05,
        "sample_mean_pairwise_cosine_distance": 0.2,
        "same_noise_full_shuffled_sample_displacement": 0.05,
    }
    suffix = {
        "suffix_pixel_equality": 1.0,
        "suffix_path_score_row_max_error": 0.0,
        "path_spatial_permutation_max_score_error": 2.0 if spatial else 0.0,
        "sample_spatial_permutation_max_score_error": 2.0 if spatial else 0.0,
        "path_full_arm_accuracy": 0.80 if spatial else 0.60,
        "path_full_both_correct_rate": 0.60 if spatial else 0.40,
        "path_full_minus_shuffled_arm_accuracy": 0.20,
        "path_full_minus_shuffled_mean_margin": 0.20,
        "path_full_minus_spatial_permuted_arm_accuracy": 0.20,
        "sample_full_arm_accuracy": 0.70,
    }
    for name in PATH_SCORE_NAMES:
        suffix[f"candidate_permutation_path_{name}_max_score_error"] = 0.0
        suffix[f"candidate_permutation_path_{name}_accuracy_agreement"] = 1.0
    return {
        "natural": natural,
        "suffix4": suffix,
        "integrity": {
            "model_state_finite": True,
            "training_metrics_finite": True,
            "natural_metrics_finite": True,
            "pair_metrics_finite": True,
            "natural_scores_and_samples_finite": True,
            "pair_scores_and_samples_finite": True,
            "student_boundary_clean": True,
            "final_checkpoint_clean": True,
            "candidate_independent_sample_shape_clean": True,
            "candidate_bank_absent": True,
            "total_parameters": 18_000_000,
            "trainable_parameters": 17_000_000,
            "peak_allocated_vram_gib": 2.0,
            "frozen_images_instantiated": False,
        },
    }


def test_v31_gate_report_requires_all_matched_controls() -> None:
    matched = {
        "initialized_parameter_states_exact": True,
        "final_parameter_counts_exact": True,
        "source_and_data_receipts_exact": True,
        "audit_windows_and_pixels_exact": True,
        "both_arms_completed_10000_finite_updates": True,
    }
    groups = v31_gate_report(
        _gate_route(spatial=True),
        _gate_route(spatial=False),
        matched,
        frozen_images_instantiated=False,
    )
    assert all(all(group.values()) for group in groups)
    matched["initialized_parameter_states_exact"] = False
    groups = v31_gate_report(
        _gate_route(spatial=True),
        _gate_route(spatial=False),
        matched,
        frozen_images_instantiated=False,
    )
    assert groups[2]["initialized_states_exact"] is False

from __future__ import annotations

import torch

from ilm.visual_lm.conditional_visual_density_ratio import (
    ConditionalVisualDensityRatioConfig,
    ConditionalVisualDensityRatioModel,
)
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from scripts.eval_conditional_visual_density_ratio_v29 import (
    evaluate_natural_language,
    evaluate_suffix_pairs,
    v29_gate_report,
)


def _model() -> ConditionalVisualDensityRatioModel:
    return ConditionalVisualDensityRatioModel(
        ConditionalVisualDensityRatioConfig(
            visual_dim=64,
            semantic_dim=64,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=8,
            semantic_hidden_dim=96,
            evidence_layers=1,
            evidence_heads=4,
            evidence_mlp_ratio=1.0,
            evidence_dropout=0.0,
            relation_hidden_dim=64,
            score_chunk_size=2,
        )
    ).eval()


def _checkpoint(model: ConditionalVisualDensityRatioModel) -> dict:
    return {
        "architecture": "conditional-visual-density-ratio-v29",
        "model": model.state_dict(),
        "deployed_state_includes_training_candidate_images": False,
        "peak_allocated_vram_gib": 0.0,
    }


def test_v29_natural_audit_reports_full_suffix_and_increment_scores() -> None:
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
        _checkpoint(model),
        device=torch.device("cpu"),
        precision="fp32",
    )
    for name in (
        "full",
        "suffix4",
        "shuffled",
        "increment",
        "shuffled_increment",
    ):
        assert 0.0 <= metrics[f"{name}_top1"] <= 1.0
        assert torch.isfinite(torch.tensor(metrics[f"{name}_target_log_probability"]))
    assert metrics["student_boundary_clean"] == 1.0
    assert metrics["training_bank_absent_from_checkpoint"] == 1.0


def test_v29_pair_audit_enforces_exact_suffix_and_permutation_controls() -> None:
    model = _model()
    suffix = torch.rand(2, 4, 1, 32, 32)
    contexts = torch.cat(
        (
            torch.rand(2, 2, 60, 1, 32, 32),
            suffix[:, None].expand(-1, 2, -1, -1, -1, -1),
        ),
        dim=2,
    )
    reference_suffix = torch.rand(2, 4, 1, 32, 32)
    reference_contexts = torch.cat(
        (
            torch.rand(2, 2, 60, 1, 32, 32),
            reference_suffix[:, None].expand(-1, 2, -1, -1, -1, -1),
        ),
        dim=2,
    )
    batch = {
        "contexts": contexts,
        "candidates": torch.rand(2, 2, 1, 32, 32),
        "assignment": torch.tensor([[0, 1], [1, 0]]),
        "reference_contexts": reference_contexts,
        "reference_candidates": torch.rand(2, 2, 1, 32, 32),
        "reference_assignment": torch.tensor([[1, 0], [0, 1]]),
        "metadata": [{"suffix_cells": 4}, {"suffix_cells": 4}],
    }
    metrics = evaluate_suffix_pairs(
        model,
        [batch],
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert metrics["suffix_pixel_equality"] == 1.0
    assert metrics["suffix_score_row_equality"] == 1.0
    assert metrics["suffix4_arm_accuracy"] == 0.5
    for name in ("full", "suffix4", "increment"):
        assert metrics[f"candidate_permutation_{name}_max_score_error"] < 1e-5
        assert metrics[f"candidate_permutation_{name}_accuracy_agreement"] == 1.0


def test_v29_gate_report_requires_every_mechanism_and_language_gate() -> None:
    natural = {
        "frozen_semantic_cross_font_identity_top1": 0.96,
        "student_boundary_clean": 1.0,
        "training_bank_absent_from_checkpoint": 1.0,
        "peak_allocated_vram_gib": 1.0,
        "full_top1": 0.20,
        "unigram_top1": 0.01,
        "bigram_top1": 0.13,
        "full_target_log_probability": -4.0,
        "bigram_target_log_probability": -5.0,
        "suffix4_target_log_probability": -4.2,
        "shuffled_target_log_probability": -4.2,
    }
    suffix = {
        "increment_arm_accuracy": 0.80,
        "increment_both_correct_rate": 0.60,
        "shuffled_increment_arm_accuracy": 0.55,
        "increment_mean_margin": 0.30,
        "shuffled_increment_mean_margin": 0.10,
        "full_arm_accuracy": 0.80,
        "full_both_correct_rate": 0.60,
        "suffix4_arm_accuracy": 0.50,
        "suffix_pixel_equality": 1.0,
        "suffix_score_row_equality": 1.0,
        "raw_retina_two_candidate_identity_accuracy": 0.999,
    }
    for name in ("full", "suffix4", "increment"):
        suffix[f"candidate_permutation_{name}_max_score_error"] = 0.0
        suffix[f"candidate_permutation_{name}_accuracy_agreement"] = 1.0
    mechanism, language = v29_gate_report(
        natural, suffix, frozen_images_instantiated=False
    )
    assert all(mechanism.values())
    assert all(language.values())
    suffix["increment_arm_accuracy"] = 0.5
    mechanism, _ = v29_gate_report(
        natural, suffix, frozen_images_instantiated=False
    )
    assert mechanism["increment_pair_arm_accuracy"] is False
    assert mechanism["increment_gain_over_shuffled"] is False

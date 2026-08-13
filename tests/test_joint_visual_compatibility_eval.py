from __future__ import annotations

import torch

from ilm.visual_lm.joint_visual_compatibility import (
    JointVisualCompatibilityConfig,
    JointVisualCompatibilityModel,
)
from scripts.eval_joint_visual_compatibility_v27 import (
    _assignment_statistics,
    _shuffle_pair_prefix,
    student_boundary_is_clean,
    v27_gate_report,
)


def _passing_natural() -> dict[str, float]:
    return {
        "full_top1": 0.20,
        "suffix4_top1": 0.10,
        "shuffled_top1": 0.10,
        "unigram_top1": 0.10,
        "bigram_top1": 0.15,
        "full_target_log_probability": -2.0,
        "suffix4_target_log_probability": -2.1,
        "shuffled_target_log_probability": -2.1,
        "bigram_target_log_probability": -2.1,
        "learned_candidate_cross_font_identity_top1": 1.0,
        "student_boundary_clean": 1.0,
        "peak_allocated_vram_gib": 1.0,
    }


def _passing_pairs() -> dict[str, float]:
    return {
        "full_arm_accuracy": 0.80,
        "full_both_correct_rate": 0.60,
        "full_mean_margin": 0.10,
        "last_arm_accuracy": 0.50,
        "suffix4_arm_accuracy": 0.50,
        "shuffled_arm_accuracy": 0.60,
        "shuffled_mean_margin": 0.05,
        "suffix_pixel_equality": 1.0,
        "candidate_permutation_max_score_error": 0.0,
        "candidate_permutation_accuracy_agreement": 1.0,
        "raw_retina_cross_font_identity_accuracy": 1.0,
    }


def test_v27_gates_distinguish_mechanism_from_language() -> None:
    mechanism, language = v27_gate_report(_passing_natural(), _passing_pairs())
    assert all(mechanism.values())
    assert all(language.values())

    below_bigram = _passing_natural()
    below_bigram["bigram_top1"] = 0.25
    mechanism, language = v27_gate_report(below_bigram, _passing_pairs())
    assert all(mechanism.values())
    assert language["full_top1_gain_over_bigram"] is False


def test_v27_gate_thresholds_are_strict_and_controls_are_exact() -> None:
    pairs = _passing_pairs()
    pairs["full_arm_accuracy"] = 0.65
    pairs["last_arm_accuracy"] = 0.500002
    mechanism, _ = v27_gate_report(_passing_natural(), pairs)
    assert mechanism["full_pair_arm_accuracy"] is False
    assert mechanism["last_control_at_chance"] is False


def test_tie_aware_assignment_statistics_give_identical_rows_chance() -> None:
    logits = torch.tensor([[[2.0, 1.0], [2.0, 1.0]]])
    labels = torch.tensor([[0, 1]], dtype=torch.long)
    metrics = _assignment_statistics(logits, labels)
    assert float(metrics["accuracy_sum"] / metrics["arms"]) == 0.5
    assert float(metrics["strict_accuracy_sum"] / metrics["arms"]) == 0.5
    assert float(metrics["both_correct_sum"]) == 0.0


def test_pair_prefix_shuffle_preserves_suffix_and_uses_one_pair_permutation() -> None:
    contexts = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(
        2, 2, 8, 1, 1, 1
    ).expand(-1, -1, -1, 1, 32, 32)
    shuffled = _shuffle_pair_prefix(
        contexts, first_index=0, preserved_suffix=4
    )
    assert torch.equal(shuffled[:, :, -4:], contexts[:, :, -4:])
    difference_before = contexts[:, 1, :4] - contexts[:, 0, :4]
    difference_after = shuffled[:, 1, :4] - shuffled[:, 0, :4]
    assert torch.equal(difference_after, difference_before)


def test_recursive_v27_boundary_audit_is_clean() -> None:
    model = JointVisualCompatibilityModel(
        JointVisualCompatibilityConfig(
            visual_dim=64,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            candidate_hidden_dim=64,
        )
    )
    assert student_boundary_is_clean(model)

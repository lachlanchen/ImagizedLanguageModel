from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from scripts import eval_dense_visual_future_energy_v28 as evaluation
from scripts.eval_dense_visual_future_energy_v28 import v28_gate_report


def _passing_natural() -> dict[str, float]:
    return {
        "full_top1": 0.25,
        "unigram_top1": 0.02,
        "bigram_top1": 0.12,
        "full_target_log_probability": -4.0,
        "bigram_target_log_probability": -5.0,
        "suffix4_target_log_probability": -4.5,
        "shuffled_target_log_probability": -4.6,
        "ema_semantic_cross_font_identity_top1": 0.97,
        "raw_retina_cross_font_identity_top1": 0.92,
        "student_boundary_clean": 1.0,
        "peak_allocated_vram_gib": 4.0,
    }


def _passing_pairs() -> dict[str, float]:
    return {
        "full_arm_accuracy": 0.80,
        "full_both_correct_rate": 0.60,
        "suffix4_arm_accuracy": 0.50,
        "shuffled_arm_accuracy": 0.60,
        "full_mean_margin": 0.20,
        "shuffled_mean_margin": 0.10,
        "last_arm_accuracy": 0.50,
        "suffix_pixel_equality": 1.0,
        "candidate_permutation_max_score_error": 0.0,
        "candidate_permutation_accuracy_agreement": 1.0,
    }


def test_v28_gate_report_selects_only_complete_mechanism_and_language() -> None:
    mechanism, language = v28_gate_report(
        _passing_natural(),
        _passing_pairs(),
        frozen_images_instantiated=False,
    )
    assert len(mechanism) == 14
    assert len(language) == 6
    assert all(mechanism.values())
    assert all(language.values())


def test_v28_gate_report_rejects_shuffled_equivalence_and_bigram_failure() -> None:
    natural = _passing_natural()
    pairs = _passing_pairs()
    natural["full_top1"] = 0.125
    natural["full_target_log_probability"] = -4.58
    pairs["shuffled_arm_accuracy"] = 0.78
    pairs["shuffled_mean_margin"] = 0.195
    mechanism, language = v28_gate_report(
        natural,
        pairs,
        frozen_images_instantiated=False,
    )
    assert mechanism["full_gain_over_shuffled"] is False
    assert mechanism["full_margin_gain_over_shuffled"] is False
    assert language["full_top1_gain_over_bigram"] is False
    assert language["full_log_probability_gain_over_shuffled"] is False


def test_v28_identity_gate_compares_the_same_full_bank_scope() -> None:
    natural = _passing_natural()
    natural["ema_semantic_cross_font_identity_top1"] = 0.94
    natural["raw_retina_cross_font_identity_top1"] = 0.93
    mechanism, _ = v28_gate_report(
        natural,
        _passing_pairs(),
        frozen_images_instantiated=False,
    )
    assert mechanism["ema_semantic_cross_font_identity"] is False
    assert mechanism["semantic_improves_same_scope_identity"] is False


def test_v28_audit_bank_renderer_requires_full_statistics(monkeypatch) -> None:
    statistics = VisualCharacterStatistics(
        characters=("天", "地"),
        counts=(9, 7),
        bigram_rows={"天": ((1, 7),), "地": ((0, 9),)},
        visible_character_count=16,
        han_character_count=16,
    )
    expected = torch.zeros(2, 2, 1, 32, 32)
    received: list[VisualCharacterStatistics] = []

    def fake_renderer(value: VisualCharacterStatistics) -> torch.Tensor:
        received.append(value)
        return expected

    monkeypatch.setattr(evaluation, "render_visual_character_bank", fake_renderer)
    assert evaluation._render_audit_bank(statistics) is expected
    assert received == [statistics]
    with pytest.raises(TypeError, match="full statistics"):
        evaluation._render_audit_bank(statistics.characters)  # type: ignore[arg-type]

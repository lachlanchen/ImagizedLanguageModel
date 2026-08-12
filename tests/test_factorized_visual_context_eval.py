from __future__ import annotations

from scripts.eval_factorized_visual_context_v26 import v26_gate_report


def _passing_natural() -> dict[str, float]:
    return {
        "full_top1": 0.20,
        "last_top1": 0.10,
        "unigram_top1": 0.10,
        "bigram_top1": 0.15,
        "full_target_log_probability": -2.0,
        "last_target_log_probability": -2.2,
        "suffix_4_target_log_probability": -2.1,
        "shuffled_prefix_target_log_probability": -2.1,
        "bigram_target_log_probability": -2.1,
        "retina_bank_oracle_top1": 1.0,
        "student_boundary_clean": 1.0,
        "peak_allocated_vram_gib": 1.0,
    }


def _passing_pairs() -> dict[str, float]:
    return {
        "pair_ranking_accuracy": 0.8,
        "swapped_residual_target_accuracy": 0.8,
        "suffix_pixel_equality": 1.0,
    }


def test_v26_gates_distinguish_mechanism_from_language() -> None:
    mechanism, language = v26_gate_report(_passing_natural(), _passing_pairs())
    assert all(mechanism.values())
    assert all(language.values())

    below_bigram = _passing_natural()
    below_bigram["bigram_top1"] = 0.25
    mechanism, language = v26_gate_report(below_bigram, _passing_pairs())
    assert all(mechanism.values())
    assert language["full_top1_gain_over_symbolic_bigram"] is False


def test_v26_thresholds_are_strict() -> None:
    natural = _passing_natural()
    natural["last_top1"] = natural["full_top1"] - 0.02
    pairs = _passing_pairs()
    pairs["pair_ranking_accuracy"] = 0.65
    mechanism, _ = v26_gate_report(natural, pairs)
    assert mechanism["full_top1_gain_over_last"] is False
    assert mechanism["suffix4_pair_ranking_accuracy"] is False

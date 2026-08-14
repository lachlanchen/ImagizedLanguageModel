from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_path_alignment_evaluation import (
    semantic_transition_metrics,
    v38_path_alignment_gate,
)


def _states(count: int = 40, dimension: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(38)
    return F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)


def test_transition_metrics_measure_full_answer_displacement() -> None:
    prompt = _states()
    answer = torch.roll(prompt, shifts=1, dims=-1)

    metrics = semantic_transition_metrics(prompt, answer, prompt, answer)

    assert metrics["transition_direction_cosine"] > 0.999
    assert metrics["prompt_answer_cosine"] < 0.5
    assert abs(metrics["delta_norm_ratio"] - 1.0) < 1e-5


def _qualified_report() -> dict:
    return {
        "weight_route": "all-parameter-ema",
        "finite": True,
        "integrity": {
            "finite_targets": True,
            "finite_model": True,
            "finite_optimizer": True,
            "finite_ema": True,
            "protocol_hash": True,
            "source_hashes": True,
            "data_hashes": True,
            "initialization_hash": True,
            "bge_hashes": True,
            "strict_mapping": True,
            "boundary": True,
            "student_inference_before_banks": True,
            "holdout_exclusion": True,
            "total_parameters": 90_753_281,
        },
        "training": {"global_update": 8_000},
        "resources": {"peak_vram_bytes": 10 * 1024**3},
        "correct": {
            "prompt_state": {
                "correct_cosine": 0.55,
                "top1": 0.60,
                "top5": 0.85,
            },
            "answer_reading": {"correct_cosine": 0.50, "top1": 0.50},
            "answer_state": {
                "top1": 0.40,
                "top5": 0.70,
                "mrr": 0.50,
                "correct_cosine": 0.35,
                "cyclic_margin": 0.20,
                "cyclic_pair_win": 0.90,
                "state_effective_rank": 40.0,
                "target_effective_rank": 76.0,
                "length_mae": 2.0,
            },
        },
        "controls": {
            "shuffled": {"correct_cosine": 0.15},
            "blank": {"correct_cosine": 0.10},
        },
        "counterfactual": {"assignment_rate": 0.95},
        "font": {
            "prompt_state_cosine": 0.80,
            "answer_state_cosine": 0.80,
            "held_answer_state": {"top5": 0.55},
        },
        "paraphrase": {
            "top5": 0.75,
            "original_prompt_cosine": 0.75,
            "original_answer_cosine": 0.75,
        },
        "transition": {
            "transition_direction_cosine": 0.30,
            "prompt_answer_cosine": 0.80,
        },
    }


def test_v38_gate_is_strictly_conjunctive_and_rejects_identity() -> None:
    report = _qualified_report()

    gate = v38_path_alignment_gate(report)

    assert gate["passed"] is True
    assert gate["decision"] == "visual-path-alignment-qualified"
    report["transition"]["prompt_answer_cosine"] = 0.99
    failed = v38_path_alignment_gate(report)
    assert failed["passed"] is False
    assert failed["conditions"]["answer_map_not_identity"] is False


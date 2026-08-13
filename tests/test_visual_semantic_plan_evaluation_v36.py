from __future__ import annotations

import torch

from ilm.visual_lm.visual_semantic_plan_evaluation import (
    nearest_length_counterfactual_pairs,
    v36_semantic_plan_gate,
    visual_plan_control_metrics,
    visual_plan_counterfactual_assignment,
    visual_plan_retrieval_metrics,
)


def test_v36_retrieval_and_control_metrics_detect_binding() -> None:
    targets = torch.eye(8)
    correct = targets.clone()
    shuffled = torch.roll(targets, shifts=1, dims=0)
    lengths = torch.arange(8, dtype=torch.float32)
    metrics = visual_plan_retrieval_metrics(
        correct,
        targets,
        predicted_lengths=lengths,
        target_lengths=lengths,
    )
    control = visual_plan_control_metrics(
        shuffled,
        correct,
        targets,
        predicted_lengths=lengths + 1,
        target_lengths=lengths,
    )
    assert metrics["top1"] == 1.0
    assert metrics["cyclic_pair_win"] == 1.0
    assert metrics["length_mae"] == 0.0
    assert control["top1"] == 0.0
    assert control["length_mae"] == 1.0


def test_v36_counterfactual_assignment_uses_nearest_lengths() -> None:
    targets = torch.eye(6)
    lengths = torch.tensor([1.0, 2.0, 8.0, 9.0, 15.0, 16.0])
    pairs = nearest_length_counterfactual_pairs(lengths, targets)
    result = visual_plan_counterfactual_assignment(targets, targets, pairs)
    assert [(pair.first, pair.second) for pair in pairs] == [(0, 1), (2, 3), (4, 5)]
    assert result["assignment_rate"] == 1.0
    assert result["assignment_margin"] > 0.0


def _passing_report() -> dict:
    return {
        "integrity": {
            "source_hashes": True,
            "external_hashes": True,
            "data_hashes": True,
            "strict_mapping": True,
            "boundary": True,
            "total_parameters": 93_000_000,
        },
        "training": {"global_update": 6_000},
        "resources": {"peak_vram_bytes": 5 * 1024**3},
        "correct": {
            "top1": 0.20,
            "top5": 0.40,
            "mrr": 0.25,
            "correct_cosine": 0.70,
            "cyclic_margin": 0.10,
            "cyclic_pair_win": 0.80,
            "length_mae": 2.0,
        },
        "controls": {
            "shuffled": {"correct_cosine": 0.60},
            "blank": {"correct_cosine": 0.50},
        },
        "counterfactual": {"assignment_rate": 0.80},
        "font": {"prompt_plan_cosine": 0.90, "answer_teacher_cosine": 0.90},
        "paraphrase": {"top5": 0.30, "original_plan_cosine": 0.80},
        "baselines": {"untrained_head": {"top1": 0.02}},
    }


def test_v36_gate_is_conjunctive() -> None:
    report = _passing_report()
    passed = v36_semantic_plan_gate(report)
    assert passed["decision"] == "semantic-plan-qualified"
    report["controls"]["blank"]["correct_cosine"] = 0.69
    failed = v36_semantic_plan_gate(report)
    assert failed["decision"] == "not-qualified"
    assert failed["conditions"]["blank_drop"] is False

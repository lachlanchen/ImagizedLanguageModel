from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_semantic_distillation_evaluation import (
    counterfactual_assignment,
    indexed_semantic_retrieval_metrics,
    nearest_length_counterfactual_pairs,
    semantic_control_metrics,
    semantic_retrieval_metrics,
    v37_semantic_distillation_gate,
)


def _states(count: int = 40, dimension: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(37)
    return F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)


def test_semantic_retrieval_and_controls_measure_causal_drop() -> None:
    targets = _states()
    correct = F.normalize(targets + 0.01 * _states(), dim=-1)
    shuffled = torch.roll(correct, shifts=1, dims=0)
    metrics = semantic_retrieval_metrics(
        correct,
        targets,
        predicted_lengths=torch.arange(40).float(),
        target_lengths=torch.arange(40).float(),
    )
    control = semantic_control_metrics(shuffled, correct, targets)

    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["length_mae"] == 0.0
    assert metrics["state_effective_rank"] > 10
    assert control["top1"] == 0.0
    assert control["correct_cosine"] < metrics["correct_cosine"]


def test_indexed_retrieval_accepts_subset_labels() -> None:
    candidates = _states()
    labels = torch.tensor([3, 7, 11, 19], dtype=torch.long)
    states = candidates[labels]
    metrics = indexed_semantic_retrieval_metrics(states, candidates, labels)
    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_nearest_length_counterfactual_assignment() -> None:
    targets = _states(count=12)
    lengths = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]).float()
    labels = [f"answer-{index}" for index in range(12)]
    pairs = nearest_length_counterfactual_pairs(lengths, targets, labels=labels)
    metrics = counterfactual_assignment(targets, targets, pairs)
    assert len(pairs) == 6
    assert metrics["assignment_rate"] == 1.0
    assert metrics["assignment_margin"] > 0


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
            "pixel_linguist_hashes": True,
            "bge_hashes": True,
            "strict_mapping": True,
            "target_sanity": True,
            "boundary": True,
            "total_parameters": 89_768_706,
        },
        "training": {"global_update": 8_000},
        "resources": {"peak_vram_bytes": 10 * 1024**3},
        "correct": {
            "prompt_state": {
                "correct_cosine": 0.75,
                "top1": 0.30,
                "top5": 0.65,
            },
            "answer_plan": {
                "top1": 0.35,
                "top5": 0.65,
                "mrr": 0.45,
                "correct_cosine": 0.40,
                "cyclic_margin": 0.25,
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
        "counterfactual": {"assignment_rate": 0.90},
        "font": {"prompt_plan_cosine": 0.90},
        "paraphrase": {"top5": 0.55, "original_plan_cosine": 0.80},
        "baselines": {"untrained_head": {"top1": 0.01}},
    }


def test_gate_is_strictly_conjunctive() -> None:
    report = _qualified_report()
    gate = v37_semantic_distillation_gate(report)
    assert gate["passed"] is True
    assert gate["decision"] == "semantic-distillation-qualified"

    report["controls"]["blank"]["correct_cosine"] = 0.20
    failed = v37_semantic_distillation_gate(report)
    assert failed["passed"] is False
    assert failed["decision"] == "not-qualified"
    assert failed["conditions"]["blank_drop"] is False

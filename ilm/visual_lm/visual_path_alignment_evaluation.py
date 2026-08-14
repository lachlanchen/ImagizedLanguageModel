from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .visual_semantic_distillation_evaluation import (
    CounterfactualPair,
    counterfactual_assignment,
    indexed_semantic_retrieval_metrics,
    mean_semantic_cosine,
    nearest_length_counterfactual_pairs,
    semantic_control_metrics,
    semantic_retrieval_metrics,
)


V38_GATE_THRESHOLDS: dict[str, float] = {
    "prompt_cosine": 0.50,
    "prompt_top1": 0.55,
    "prompt_top5": 0.80,
    "answer_read_cosine": 0.45,
    "answer_read_top1": 0.45,
    "answer_top1": 0.35,
    "answer_top5": 0.65,
    "answer_mrr": 0.45,
    "answer_cosine": 0.30,
    "cyclic_margin": 0.15,
    "cyclic_pair_win": 0.85,
    "shuffled_drop": 0.15,
    "blank_drop": 0.20,
    "counterfactual_assignment": 0.90,
    "held_prompt_cosine": 0.75,
    "held_answer_cosine": 0.75,
    "held_answer_top5": 0.50,
    "paraphrase_top5": 0.70,
    "paraphrase_prompt_cosine": 0.70,
    "paraphrase_answer_cosine": 0.70,
    "transition_direction_cosine": 0.25,
    "maximum_prompt_answer_cosine": 0.95,
    "answer_rank_absolute": 32.0,
    "answer_rank_relative": 0.40,
    "maximum_length_mae": 3.0,
}


def _normalized_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"V38 {name} must be a floating tensor")
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"V38 {name} must be [N,D] with N >= 2")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"V38 {name} contains non-finite values")
    if not bool((value.float().norm(dim=-1) > 1e-8).all()):
        raise ValueError(f"V38 {name} contains a zero vector")
    return F.normalize(value.float(), dim=-1)


def semantic_transition_metrics(
    prompt_states: torch.Tensor,
    answer_states: torch.Tensor,
    prompt_targets: torch.Tensor,
    answer_targets: torch.Tensor,
) -> dict[str, float | int]:
    prompt = _normalized_matrix("prompt states", prompt_states)
    answer = _normalized_matrix("answer states", answer_states)
    prompt_target = _normalized_matrix("prompt targets", prompt_targets)
    answer_target = _normalized_matrix("answer targets", answer_targets)
    if not (
        prompt.shape == answer.shape == prompt_target.shape == answer_target.shape
    ):
        raise ValueError("V38 transition matrices must align")
    student_delta = answer - prompt
    target_delta = answer_target - prompt_target
    student_norm = student_delta.norm(dim=-1)
    target_norm = target_delta.norm(dim=-1)
    valid = (student_norm > 1e-8) & (target_norm > 1e-8)
    if not bool(valid.any()):
        raise ValueError("V38 transition deltas are all zero")
    direction = F.cosine_similarity(
        student_delta[valid],
        target_delta[valid],
        dim=-1,
    )
    return {
        "samples": len(prompt),
        "valid_delta_samples": int(valid.sum()),
        "prompt_answer_cosine": float(
            F.cosine_similarity(prompt, answer, dim=-1).mean()
        ),
        "target_prompt_answer_cosine": float(
            F.cosine_similarity(prompt_target, answer_target, dim=-1).mean()
        ),
        "transition_direction_cosine": float(direction.mean()),
        "student_delta_norm": float(student_norm.mean()),
        "target_delta_norm": float(target_norm.mean()),
        "delta_norm_ratio": float(
            student_norm.mean() / target_norm.mean().clamp_min(1e-8)
        ),
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite_tree(item) for item in value)
    return True


def v38_path_alignment_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    integrity = report["integrity"]
    training = report["training"]
    resources = report["resources"]
    prompt = report["correct"]["prompt_state"]
    answer_read = report["correct"]["answer_reading"]
    answer = report["correct"]["answer_state"]
    controls = report["controls"]
    counterfactual = report["counterfactual"]
    font = report["font"]
    paraphrase = report["paraphrase"]
    transition = report["transition"]
    thresholds = V38_GATE_THRESHOLDS
    target_rank = float(answer["target_effective_rank"])
    answer_rank = float(answer["state_effective_rank"])
    conditions = {
        "ema_primary_route": report.get("weight_route") == "all-parameter-ema",
        "finite_report": (
            _finite_tree(report)
            and bool(report.get("finite", False))
            and bool(integrity.get("finite_targets", False))
            and bool(integrity.get("finite_model", False))
            and bool(integrity.get("finite_optimizer", False))
            and bool(integrity.get("finite_ema", False))
        ),
        "protocol_hash": bool(integrity["protocol_hash"]),
        "source_hashes": bool(integrity["source_hashes"]),
        "data_hashes": bool(integrity["data_hashes"]),
        "initialization_hash": bool(integrity["initialization_hash"]),
        "bge_hashes": bool(integrity["bge_hashes"]),
        "strict_mapping": bool(integrity["strict_mapping"]),
        "boundary": bool(integrity["boundary"]),
        "student_inference_before_banks": bool(
            integrity["student_inference_before_banks"]
        ),
        "holdout_exclusion": bool(integrity["holdout_exclusion"]),
        "parameter_cap": int(integrity["total_parameters"]) < 100_000_000,
        "updates": int(training["global_update"]) == 8_000,
        "peak_vram": int(resources["peak_vram_bytes"]) < 20 * 1024**3,
        "prompt_cosine": float(prompt["correct_cosine"])
        >= thresholds["prompt_cosine"],
        "prompt_top1": float(prompt["top1"]) >= thresholds["prompt_top1"],
        "prompt_top5": float(prompt["top5"]) >= thresholds["prompt_top5"],
        "answer_read_cosine": float(answer_read["correct_cosine"])
        >= thresholds["answer_read_cosine"],
        "answer_read_top1": float(answer_read["top1"])
        >= thresholds["answer_read_top1"],
        "answer_top1": float(answer["top1"]) >= thresholds["answer_top1"],
        "answer_top5": float(answer["top5"]) >= thresholds["answer_top5"],
        "answer_mrr": float(answer["mrr"]) >= thresholds["answer_mrr"],
        "answer_cosine": float(answer["correct_cosine"])
        >= thresholds["answer_cosine"],
        "cyclic_margin": float(answer["cyclic_margin"])
        >= thresholds["cyclic_margin"],
        "cyclic_pair_win": float(answer["cyclic_pair_win"])
        >= thresholds["cyclic_pair_win"],
        "shuffled_drop": float(answer["correct_cosine"])
        - float(controls["shuffled"]["correct_cosine"])
        >= thresholds["shuffled_drop"],
        "blank_drop": float(answer["correct_cosine"])
        - float(controls["blank"]["correct_cosine"])
        >= thresholds["blank_drop"],
        "counterfactual_assignment": float(counterfactual["assignment_rate"])
        >= thresholds["counterfactual_assignment"],
        "held_prompt_cosine": float(font["prompt_state_cosine"])
        >= thresholds["held_prompt_cosine"],
        "held_answer_cosine": float(font["answer_state_cosine"])
        >= thresholds["held_answer_cosine"],
        "held_answer_top5": float(font["held_answer_state"]["top5"])
        >= thresholds["held_answer_top5"],
        "paraphrase_top5": float(paraphrase["top5"])
        >= thresholds["paraphrase_top5"],
        "paraphrase_prompt_cosine": float(paraphrase["original_prompt_cosine"])
        >= thresholds["paraphrase_prompt_cosine"],
        "paraphrase_answer_cosine": float(paraphrase["original_answer_cosine"])
        >= thresholds["paraphrase_answer_cosine"],
        "transition_direction": float(transition["transition_direction_cosine"])
        >= thresholds["transition_direction_cosine"],
        "answer_map_not_identity": float(transition["prompt_answer_cosine"])
        <= thresholds["maximum_prompt_answer_cosine"],
        "answer_rank_absolute": answer_rank >= thresholds["answer_rank_absolute"],
        "answer_rank_relative": answer_rank
        >= thresholds["answer_rank_relative"] * target_rank,
        "length_mae": float(answer["length_mae"])
        <= thresholds["maximum_length_mae"],
    }
    passed = all(conditions.values())
    return {
        "thresholds": dict(thresholds),
        "conditions": conditions,
        "passed": passed,
        "decision": "visual-path-alignment-qualified" if passed else "not-qualified",
        "passed_conditions": sum(bool(value) for value in conditions.values()),
        "total_conditions": len(conditions),
    }


__all__ = [
    "CounterfactualPair",
    "V38_GATE_THRESHOLDS",
    "counterfactual_assignment",
    "indexed_semantic_retrieval_metrics",
    "mean_semantic_cosine",
    "nearest_length_counterfactual_pairs",
    "semantic_control_metrics",
    "semantic_retrieval_metrics",
    "semantic_transition_metrics",
    "v38_path_alignment_gate",
]

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .visual_semantic_distillation_training import centered_effective_rank


@dataclass(frozen=True)
class CounterfactualPair:
    first: int
    second: int


def _normalized_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"V37 {name} must be a floating tensor")
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"V37 {name} must be [N,D] with N >= 2")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"V37 {name} contains non-finite values")
    norms = value.float().norm(dim=-1)
    if not bool((norms > 1e-8).all()):
        raise ValueError(f"V37 {name} contains a zero vector")
    return F.normalize(value.float(), dim=-1)


def semantic_retrieval_metrics(
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    predicted_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, float | int]:
    states = _normalized_matrix("states", states)
    targets = _normalized_matrix("targets", targets)
    if states.shape != targets.shape:
        raise ValueError("V37 semantic states and targets must align")
    count = len(states)
    similarities = states @ targets.T
    ranking = similarities.argsort(dim=1, descending=True)
    correct = torch.arange(count, device=states.device)
    positions = (ranking == correct[:, None]).nonzero(as_tuple=False)[:, 1]
    cyclic = torch.roll(correct, shifts=1)
    correct_cosine = similarities.diag()
    cyclic_cosine = similarities[correct, cyclic]
    result: dict[str, float | int] = {
        "samples": count,
        "top1": float((positions == 0).float().mean()),
        "top5": float((positions < min(5, count)).float().mean()),
        "mrr": float((1.0 / (positions.float() + 1.0)).mean()),
        "correct_cosine": float(correct_cosine.mean()),
        "cyclic_cosine": float(cyclic_cosine.mean()),
        "cyclic_margin": float((correct_cosine - cyclic_cosine).mean()),
        "cyclic_pair_win": float((correct_cosine > cyclic_cosine).float().mean()),
        "state_variance": float(states.var(dim=0, unbiased=False).mean()),
        "state_effective_rank": centered_effective_rank(states),
        "target_effective_rank": centered_effective_rank(targets),
    }
    if predicted_lengths is not None or target_lengths is not None:
        if predicted_lengths is None or target_lengths is None:
            raise ValueError("V37 length metrics require prediction and target")
        if predicted_lengths.shape != (count,) or target_lengths.shape != (count,):
            raise ValueError("V37 visual lengths must be [N]")
        if not bool(
            torch.isfinite(predicted_lengths).all()
            and torch.isfinite(target_lengths).all()
        ):
            raise ValueError("V37 visual lengths contain non-finite values")
        result["length_mae"] = float(
            (predicted_lengths.float() - target_lengths.float()).abs().mean()
        )
    return result


def indexed_semantic_retrieval_metrics(
    states: torch.Tensor,
    candidates: torch.Tensor,
    correct_indices: torch.Tensor,
) -> dict[str, float | int]:
    states = _normalized_matrix("indexed states", states)
    candidates = _normalized_matrix("indexed candidates", candidates)
    if states.shape[1] != candidates.shape[1]:
        raise ValueError("V37 indexed state and candidate widths differ")
    if correct_indices.shape != (len(states),) or correct_indices.dtype != torch.long:
        raise ValueError("V37 indexed retrieval labels must be long [N]")
    if not bool(((correct_indices >= 0) & (correct_indices < len(candidates))).all()):
        raise ValueError("V37 indexed retrieval label is outside candidates")
    similarities = states @ candidates.T
    ranking = similarities.argsort(dim=1, descending=True)
    positions = (ranking == correct_indices[:, None]).nonzero(as_tuple=False)[:, 1]
    correct_cosine = similarities[torch.arange(len(states)), correct_indices]
    return {
        "samples": len(states),
        "candidates": len(candidates),
        "top1": float((positions == 0).float().mean()),
        "top5": float((positions < min(5, len(candidates))).float().mean()),
        "mrr": float((1.0 / (positions.float() + 1.0)).mean()),
        "correct_cosine": float(correct_cosine.mean()),
        "state_effective_rank": centered_effective_rank(states),
        "candidate_effective_rank": centered_effective_rank(candidates),
    }


def semantic_control_metrics(
    control_states: torch.Tensor,
    correct_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    predicted_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, float | int]:
    control = _normalized_matrix("control states", control_states)
    correct = _normalized_matrix("correct states", correct_states)
    if control.shape != correct.shape:
        raise ValueError("V37 control and correct states must align")
    result = semantic_retrieval_metrics(
        control,
        targets,
        predicted_lengths=predicted_lengths,
        target_lengths=target_lengths,
    )
    result["cosine_to_correct"] = float(
        F.cosine_similarity(control, correct, dim=-1).mean()
    )
    result["mean_conditioned_delta"] = float(
        (control - correct).square().sum(dim=-1).sqrt().mean()
    )
    return result


def mean_semantic_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = _normalized_matrix("first states", first)
    second = _normalized_matrix("second states", second)
    if first.shape != second.shape:
        raise ValueError("V37 semantic cosine inputs must align")
    return float(F.cosine_similarity(first, second, dim=-1).mean())


def nearest_length_counterfactual_pairs(
    target_lengths: torch.Tensor,
    targets: torch.Tensor,
    *,
    maximum_target_cosine: float = 0.98,
    labels: Sequence[str] | None = None,
) -> tuple[CounterfactualPair, ...]:
    if target_lengths.ndim != 1 or len(target_lengths) < 2:
        raise ValueError("V37 counterfactual lengths must be [N] with N >= 2")
    normalized = _normalized_matrix("counterfactual targets", targets)
    if len(normalized) != len(target_lengths):
        raise ValueError("V37 counterfactual targets and lengths do not align")
    if labels is not None and len(labels) != len(target_lengths):
        raise ValueError("V37 counterfactual labels and lengths do not align")
    remaining = set(range(len(target_lengths)))
    pairs: list[CounterfactualPair] = []
    while len(remaining) >= 2:
        first = min(remaining)
        remaining.remove(first)
        candidates = sorted(
            remaining,
            key=lambda index: (
                abs(float(target_lengths[first] - target_lengths[index])),
                index,
            ),
        )
        distinct = (
            candidates
            if labels is None
            else [index for index in candidates if labels[index] != labels[first]]
        )
        if not distinct:
            raise ValueError("V37 cannot form a different-answer counterfactual pair")
        second = next(
            (
                index
                for index in distinct
                if float(normalized[first] @ normalized[index]) < maximum_target_cosine
            ),
            distinct[0],
        )
        remaining.remove(second)
        pairs.append(CounterfactualPair(first=first, second=second))
    return tuple(pairs)


def _bootstrap_binary_interval(
    values: torch.Tensor,
    *,
    seed: int,
    replicates: int = 2_000,
) -> tuple[float, float] | None:
    if values.numel() < 2:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        len(values),
        (replicates, len(values)),
        generator=generator,
    )
    samples = values.float().cpu()[indices].mean(dim=1)
    return float(torch.quantile(samples, 0.025)), float(torch.quantile(samples, 0.975))


def counterfactual_assignment(
    plans: torch.Tensor,
    targets: torch.Tensor,
    pairs: Sequence[CounterfactualPair],
    *,
    bootstrap_seed: int = 20_263_701,
) -> dict[str, Any]:
    plans = _normalized_matrix("counterfactual plans", plans)
    targets = _normalized_matrix("counterfactual targets", targets)
    if plans.shape != targets.shape or not pairs:
        raise ValueError("V37 counterfactual plans, targets, or pairs are invalid")
    outcomes: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for pair in pairs:
        if not 0 <= pair.first < len(plans) or not 0 <= pair.second < len(plans):
            raise IndexError("V37 counterfactual pair is outside the plan set")
        direct = (
            plans[pair.first] @ targets[pair.first]
            + plans[pair.second] @ targets[pair.second]
        )
        crossed = (
            plans[pair.first] @ targets[pair.second]
            + plans[pair.second] @ targets[pair.first]
        )
        margin = direct - crossed
        outcomes.append((margin > 0).float())
        margins.append(margin)
    outcome_tensor = torch.stack(outcomes)
    margin_tensor = torch.stack(margins)
    interval = _bootstrap_binary_interval(outcome_tensor, seed=bootstrap_seed)
    return {
        "pairs": len(pairs),
        "assignment_rate": float(outcome_tensor.mean()),
        "assignment_ci95": list(interval) if interval is not None else None,
        "assignment_margin": float(margin_tensor.mean()),
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


def v37_semantic_distillation_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    integrity = report["integrity"]
    training = report["training"]
    resources = report["resources"]
    reading = report["correct"]["prompt_state"]
    planning = report["correct"]["answer_plan"]
    controls = report["controls"]
    counterfactual = report["counterfactual"]
    font = report["font"]
    paraphrase = report["paraphrase"]
    untrained_top1 = float(report["baselines"]["untrained_head"]["top1"])
    target_rank = float(planning["target_effective_rank"])
    plan_rank = float(planning["state_effective_rank"])
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
        "pixel_linguist_hashes": bool(integrity["pixel_linguist_hashes"]),
        "bge_hashes": bool(integrity["bge_hashes"]),
        "strict_mapping": bool(integrity["strict_mapping"]),
        "target_sanity": bool(integrity["target_sanity"]),
        "boundary": bool(integrity["boundary"]),
        "parameter_cap": int(integrity["total_parameters"]) < 100_000_000,
        "updates": int(training["global_update"]) == 8_000,
        "peak_vram": int(resources["peak_vram_bytes"]) < 20 * 1024**3,
        "prompt_cosine": float(reading["correct_cosine"]) >= 0.70,
        "prompt_top1": float(reading["top1"]) >= 0.25,
        "prompt_top5": float(reading["top5"]) >= 0.60,
        "answer_top1": float(planning["top1"]) >= 0.30,
        "answer_top5": float(planning["top5"]) >= 0.60,
        "answer_mrr": float(planning["mrr"]) >= 0.40,
        "answer_over_untrained_ratio": float(planning["top1"]) >= 10.0 * untrained_top1,
        "answer_over_untrained_absolute": float(planning["top1"])
        >= untrained_top1 + 0.20,
        "answer_cosine": float(planning["correct_cosine"]) >= 0.35,
        "cyclic_margin": float(planning["cyclic_margin"]) >= 0.20,
        "cyclic_pair_win": float(planning["cyclic_pair_win"]) >= 0.85,
        "shuffled_drop": float(planning["correct_cosine"])
        - float(controls["shuffled"]["correct_cosine"])
        >= 0.20,
        "blank_drop": float(planning["correct_cosine"])
        - float(controls["blank"]["correct_cosine"])
        >= 0.25,
        "counterfactual_assignment": float(counterfactual["assignment_rate"]) >= 0.85,
        "held_font": float(font["prompt_plan_cosine"]) >= 0.85,
        "paraphrase_top5": float(paraphrase["top5"]) >= 0.50,
        "paraphrase_cosine": float(paraphrase["original_plan_cosine"]) >= 0.75,
        "plan_rank_absolute": plan_rank >= 32.0,
        "plan_rank_relative": plan_rank >= 0.40 * target_rank,
        "length_mae": float(planning["length_mae"]) <= 3.0,
    }
    passed = all(conditions.values())
    return {
        "conditions": conditions,
        "passed": passed,
        "decision": "semantic-distillation-qualified" if passed else "not-qualified",
        "passed_conditions": sum(bool(value) for value in conditions.values()),
        "total_conditions": len(conditions),
    }


__all__ = [
    "CounterfactualPair",
    "counterfactual_assignment",
    "indexed_semantic_retrieval_metrics",
    "mean_semantic_cosine",
    "nearest_length_counterfactual_pairs",
    "semantic_control_metrics",
    "semantic_retrieval_metrics",
    "v37_semantic_distillation_gate",
]

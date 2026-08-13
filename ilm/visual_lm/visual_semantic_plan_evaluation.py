from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CounterfactualPair:
    first: int
    second: int


def _validate_plan_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"V36 {name} must be a floating tensor")
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"V36 {name} must be [N,D] with N >= 2")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"V36 {name} contains non-finite values")
    return F.normalize(value.float(), dim=-1)


def visual_plan_retrieval_metrics(
    plans: torch.Tensor,
    targets: torch.Tensor,
    *,
    predicted_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, float | int]:
    plans = _validate_plan_matrix("plans", plans)
    targets = _validate_plan_matrix("targets", targets)
    if plans.shape != targets.shape:
        raise ValueError("V36 plans and answer targets must have matching shapes")
    count = plans.shape[0]
    similarities = plans @ targets.T
    ranking = similarities.argsort(dim=1, descending=True)
    correct = torch.arange(count, device=plans.device)
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
        "plan_variance": float(plans.var(dim=0, unbiased=False).mean()),
    }
    if predicted_lengths is not None or target_lengths is not None:
        if predicted_lengths is None or target_lengths is None:
            raise ValueError("V36 length metrics require prediction and target")
        if predicted_lengths.shape != (count,) or target_lengths.shape != (count,):
            raise ValueError("V36 visual lengths must be [N]")
        result["length_mae"] = float(
            (predicted_lengths.float() - target_lengths.float()).abs().mean()
        )
    return result


def visual_plan_control_metrics(
    control_plans: torch.Tensor,
    correct_plans: torch.Tensor,
    targets: torch.Tensor,
    *,
    predicted_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, float | int]:
    control = _validate_plan_matrix("control plans", control_plans)
    correct = _validate_plan_matrix("correct plans", correct_plans)
    if control.shape != correct.shape:
        raise ValueError("V36 control and correct plans must align")
    result = visual_plan_retrieval_metrics(
        control,
        targets,
        predicted_lengths=predicted_lengths,
        target_lengths=target_lengths,
    )
    result["plan_cosine_to_correct"] = float(
        F.cosine_similarity(control, correct, dim=-1).mean()
    )
    result["mean_pixel_conditioned_plan_delta"] = float(
        (control - correct).square().sum(dim=-1).sqrt().mean()
    )
    return result


def nearest_length_counterfactual_pairs(
    target_lengths: torch.Tensor,
    targets: torch.Tensor,
    *,
    maximum_target_cosine: float = 0.98,
) -> tuple[CounterfactualPair, ...]:
    if target_lengths.ndim != 1 or len(target_lengths) < 2:
        raise ValueError("V36 counterfactual lengths must be [N] with N >= 2")
    normalized = _validate_plan_matrix("counterfactual targets", targets)
    if len(normalized) != len(target_lengths):
        raise ValueError("V36 counterfactual targets and lengths do not align")
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
        second = next(
            (
                index
                for index in candidates
                if float(normalized[first] @ normalized[index]) < maximum_target_cosine
            ),
            candidates[0],
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
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        len(values),
        (replicates, len(values)),
        generator=generator,
    )
    samples = values.float().cpu()[indices].mean(dim=1)
    lower = torch.quantile(samples, 0.025)
    upper = torch.quantile(samples, 0.975)
    return float(lower), float(upper)


def visual_plan_counterfactual_assignment(
    plans: torch.Tensor,
    targets: torch.Tensor,
    pairs: Sequence[CounterfactualPair],
    *,
    bootstrap_seed: int = 20_263_601,
) -> dict[str, Any]:
    plans = _validate_plan_matrix("counterfactual plans", plans)
    targets = _validate_plan_matrix("counterfactual targets", targets)
    if plans.shape != targets.shape:
        raise ValueError("V36 counterfactual plans and targets must align")
    if not pairs:
        raise ValueError("V36 counterfactual assignment requires pairs")
    outcomes: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for pair in pairs:
        if not 0 <= pair.first < len(plans) or not 0 <= pair.second < len(plans):
            raise IndexError("V36 counterfactual pair is outside the plan set")
        direct = plans[pair.first] @ targets[pair.first] + plans[pair.second] @ targets[
            pair.second
        ]
        crossed = plans[pair.first] @ targets[pair.second] + plans[pair.second] @ targets[
            pair.first
        ]
        margin = direct - crossed
        margins.append(margin)
        outcomes.append((margin > 0).float())
    outcome_tensor = torch.stack(outcomes)
    margin_tensor = torch.stack(margins)
    interval = _bootstrap_binary_interval(
        outcome_tensor,
        seed=bootstrap_seed,
    )
    return {
        "pairs": len(pairs),
        "assignment_rate": float(outcome_tensor.mean()),
        "assignment_ci95": list(interval) if interval is not None else None,
        "assignment_margin": float(margin_tensor.mean()),
    }


def mean_plan_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = _validate_plan_matrix("first plans", first)
    second = _validate_plan_matrix("second plans", second)
    if first.shape != second.shape:
        raise ValueError("V36 plan cosine inputs must align")
    return float(F.cosine_similarity(first, second, dim=-1).mean())


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


def v36_semantic_plan_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    integrity = report["integrity"]
    correct = report["correct"]
    controls = report["controls"]
    counterfactual = report["counterfactual"]
    font = report["font"]
    paraphrase = report["paraphrase"]
    baselines = report["baselines"]
    resources = report["resources"]
    training = report["training"]
    untrained_top1 = float(baselines["untrained_head"]["top1"])

    gates = {
        "finite_report": _finite_tree(report),
        "source_hashes": bool(integrity["source_hashes"]),
        "external_hashes": bool(integrity["external_hashes"]),
        "data_hashes": bool(integrity["data_hashes"]),
        "strict_mapping": bool(integrity["strict_mapping"]),
        "boundary": bool(integrity["boundary"]),
        "parameter_cap": int(integrity["total_parameters"]) < 100_000_000,
        "updates": int(training["global_update"]) == 6_000,
        "peak_vram": int(resources["peak_vram_bytes"]) < 20 * 1024**3,
        "top1_absolute": float(correct["top1"]) >= 0.08,
        "top1_over_untrained": float(correct["top1"])
        >= 4.0 * untrained_top1,
        "top5": float(correct["top5"]) >= 0.25,
        "mrr": float(correct["mrr"]) >= 0.15,
        "cyclic_margin": float(correct["cyclic_margin"]) >= 0.05,
        "cyclic_pair_win": float(correct["cyclic_pair_win"]) >= 0.70,
        "shuffled_drop": float(correct["correct_cosine"])
        - float(controls["shuffled"]["correct_cosine"])
        >= 0.03,
        "blank_drop": float(correct["correct_cosine"])
        - float(controls["blank"]["correct_cosine"])
        >= 0.05,
        "counterfactual_assignment": float(counterfactual["assignment_rate"])
        >= 0.70,
        "held_font_prompt": float(font["prompt_plan_cosine"]) >= 0.85,
        "paraphrase_top5": float(paraphrase["top5"]) >= 0.20,
        "paraphrase_plan_cosine": float(paraphrase["original_plan_cosine"])
        >= 0.75,
        "answer_teacher_cross_font": float(font["answer_teacher_cosine"])
        >= 0.80,
        "length_mae": float(correct["length_mae"]) <= 4.0,
    }
    passed = all(gates.values())
    return {
        "conditions": gates,
        "passed": passed,
        "decision": "semantic-plan-qualified" if passed else "not-qualified",
        "passed_conditions": sum(bool(value) for value in gates.values()),
        "total_conditions": len(gates),
    }

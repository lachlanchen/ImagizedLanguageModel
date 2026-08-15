from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import iter_split_writing
from .visual_cell_eval_data import (
    VisualCellAuditWindow,
    VisualCharacterStatistics,
)


PREDICTIVE_STATE_CONTEXT_LENGTHS = (1, 2, 4, 8, 16, 32, 64)
PREDICTIVE_STATE_SHUFFLED_LENGTHS = (8, 16, 32, 64)
DIRECT_ACTUATOR_THRESHOLDS = tuple(
    round(-0.50 + 0.025 * index, 3) for index in range(51)
)


def build_partition_audit_windows(
    records: Sequence[VisualGrammarRecord],
    statistics: VisualCharacterStatistics,
    *,
    split: str,
    count: int,
    seed: int,
    script_views_mode: str = "original+simplified",
) -> tuple[VisualCellAuditWindow, ...]:
    """Reservoir-sample fixed train or development next-cell windows."""

    if split not in {"train", "development"}:
        raise ValueError("diagnostic split must be train or development")
    if count < 1:
        raise ValueError("diagnostic window count must be positive")
    bank = set(statistics.characters)
    rng = random.Random(seed)
    reservoir: list[VisualCellAuditWindow] = []
    eligible = 0
    for record, script_view, writing in iter_split_writing(
        records,
        split=split,
        script_views_mode=script_views_mode,
    ):
        for target_offset in range(64, len(writing)):
            if writing[target_offset] not in bank:
                continue
            window = VisualCellAuditWindow(
                identifier=record.identifier,
                script_view=script_view,
                context=writing[target_offset - 64 : target_offset],
                continuation=writing[target_offset],
            )
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(window)
                continue
            replacement = rng.randrange(eligible)
            if replacement < count:
                reservoir[replacement] = window
    if len(reservoir) != count:
        raise ValueError(
            f"{split} split yielded {len(reservoir)} of {count} diagnostic windows"
        )
    rng.shuffle(reservoir)
    return tuple(reservoir)


def audit_window_digest(windows: Sequence[VisualCellAuditWindow]) -> str:
    """Hash the ordered evaluator-side window identity without ambiguity."""

    digest = hashlib.sha256()
    for window in windows:
        payload = json.dumps(
            [
                window.identifier,
                window.script_view,
                window.context,
                window.continuation,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def shuffle_prefix_preserving_suffix(
    context: torch.Tensor,
    *,
    first_index: int,
    seed: int,
    suffix_cells: int = 4,
) -> torch.Tensor:
    """Deterministically permute visible history while preserving its suffix."""

    if context.ndim != 5 or tuple(context.shape[-3:]) != (1, 32, 32):
        raise ValueError("diagnostic context must be [B,T,1,32,32]")
    if not 0 < suffix_cells < context.shape[1]:
        raise ValueError("shuffle suffix must be shorter than the context")
    prefix_cells = context.shape[1] - suffix_cells
    permutations: list[torch.Tensor] = []
    for offset in range(context.shape[0]):
        generator = torch.Generator().manual_seed(
            int(seed)
            + context.shape[1] * 1_000_003
            + (first_index + offset) * 104_729
        )
        permutations.append(torch.randperm(prefix_cells, generator=generator))
    permutation = torch.stack(permutations).to(context.device)
    gather = permutation[:, :, None, None, None].expand(
        -1, -1, *context.shape[2:]
    )
    shuffled = context.clone()
    shuffled[:, :prefix_cells] = context[:, :prefix_cells].gather(1, gather)
    return shuffled


def _autocast(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("diagnostic precision must be fp32, fp16, or bf16")
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _empty_metric_accumulator() -> dict[str, Any]:
    return {
        "examples": 0,
        "correct_top1": 0.0,
        "correct_top5": 0.0,
        "target_log_probability_sum": 0.0,
        "target_cosine_sum": 0.0,
        "entropy_sum": 0.0,
        "target_rank_sum": 0.0,
        "reciprocal_rank_sum": 0.0,
        "predictions": Counter(),
    }


def _score_batch(
    accumulator: dict[str, Any],
    logits: torch.Tensor,
    targets: torch.Tensor,
    anchors: torch.Tensor,
    target_fields: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("diagnostic logits and targets do not align")
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probabilities = logits.log_softmax(dim=1)
    target_log_probability = log_probabilities.gather(1, targets[:, None])[:, 0]
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    target_score = logits.gather(1, targets[:, None])[:, 0]
    rank = 1 + (logits > target_score[:, None]).sum(dim=1)
    other = logits.clone()
    other.scatter_(1, targets[:, None], -torch.inf)
    margin = target_score - other.max(dim=1).values

    examples = int(targets.shape[0])
    accumulator["examples"] += examples
    accumulator["correct_top1"] += float((top[:, 0] == targets).sum())
    accumulator["correct_top5"] += float(
        (top == targets[:, None]).any(dim=1).sum()
    )
    accumulator["target_log_probability_sum"] += float(
        target_log_probability.sum()
    )
    accumulator["target_cosine_sum"] += float(
        (anchors.float() * target_fields.float()).sum(dim=-1).sum()
    )
    accumulator["entropy_sum"] += float(entropy.sum())
    accumulator["target_rank_sum"] += float(rank.sum())
    accumulator["reciprocal_rank_sum"] += float((1.0 / rank.float()).sum())
    accumulator["predictions"].update(top[:, 0].detach().cpu().tolist())
    return rank, margin


def _finish_metrics(accumulator: Mapping[str, Any]) -> dict[str, float]:
    examples = int(accumulator["examples"])
    if examples < 1:
        raise ValueError("diagnostic metric accumulator is empty")
    predictions: Counter[int] = accumulator["predictions"]
    most_common = predictions.most_common(1)[0][1]
    return {
        "examples": float(examples),
        "top1": float(accumulator["correct_top1"]) / examples,
        "top5": float(accumulator["correct_top5"]) / examples,
        "target_log_probability": float(
            accumulator["target_log_probability_sum"]
        )
        / examples,
        "target_cosine": float(accumulator["target_cosine_sum"]) / examples,
        "prediction_entropy": float(accumulator["entropy_sum"]) / examples,
        "target_rank_mean": float(accumulator["target_rank_sum"]) / examples,
        "mean_reciprocal_rank": float(accumulator["reciprocal_rank_sum"])
        / examples,
        "distinct_top1_predictions": float(len(predictions)),
        "most_common_top1_fraction": float(most_common) / examples,
    }


def _empty_intervention_accumulator() -> dict[str, float]:
    return {
        "examples": 0.0,
        "anchor_cosine_sum": 0.0,
        "rank_improved": 0.0,
        "rank_worsened": 0.0,
        "rank_unchanged": 0.0,
        "margin_change_sum": 0.0,
    }


def _accumulate_intervention(
    accumulator: dict[str, float],
    *,
    anchor: torch.Tensor,
    baseline_anchor: torch.Tensor,
    rank: torch.Tensor,
    baseline_rank: torch.Tensor,
    margin: torch.Tensor,
    baseline_margin: torch.Tensor,
) -> None:
    examples = int(rank.shape[0])
    accumulator["examples"] += examples
    accumulator["anchor_cosine_sum"] += float(
        (anchor.float() * baseline_anchor.float()).sum(dim=-1).sum()
    )
    accumulator["rank_improved"] += float((rank < baseline_rank).sum())
    accumulator["rank_worsened"] += float((rank > baseline_rank).sum())
    accumulator["rank_unchanged"] += float((rank == baseline_rank).sum())
    accumulator["margin_change_sum"] += float((margin - baseline_margin).sum())


def _finish_intervention(accumulator: Mapping[str, float]) -> dict[str, float]:
    examples = float(accumulator["examples"])
    if examples < 1:
        raise ValueError("diagnostic intervention accumulator is empty")
    return {
        "examples": examples,
        "anchor_cosine": accumulator["anchor_cosine_sum"] / examples,
        "target_rank_improved_fraction": accumulator["rank_improved"] / examples,
        "target_rank_worsened_fraction": accumulator["rank_worsened"] / examples,
        "target_rank_unchanged_fraction": accumulator["rank_unchanged"] / examples,
        "target_logit_margin_change": accumulator["margin_change_sum"] / examples,
    }


@torch.no_grad()
def evaluate_predictive_state(
    model: Any,
    loader: Iterable[dict[str, Any]],
    bank_fields: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    shuffle_seed: int,
) -> dict[str, Any]:
    """Measure context use without exposing evaluator labels to the model."""

    model.eval()
    if bank_fields.ndim != 2 or bank_fields.shape[0] < 2:
        raise ValueError("diagnostic bank fields must be [K,D]")
    context_accumulators = {
        length: _empty_metric_accumulator()
        for length in PREDICTIVE_STATE_CONTEXT_LENGTHS
    }
    shuffled_accumulators = {
        length: _empty_metric_accumulator()
        for length in PREDICTIVE_STATE_SHUFFLED_LENGTHS
    }
    interventions = {
        length: {
            "vs_last": _empty_intervention_accumulator(),
            **(
                {"vs_suffix4": _empty_intervention_accumulator()}
                if length > 4
                else {}
            ),
        }
        for length in PREDICTIVE_STATE_CONTEXT_LENGTHS
        if length > 1
    }

    examples = 0
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        targets = raw["target_index"].to(device, non_blocking=True)
        target_pixels = raw["continuation"][:, 0].to(device, non_blocking=True)
        if context.ndim != 5 or tuple(context.shape[-3:]) != (1, 32, 32):
            raise ValueError("diagnostic student context is not a raster stream")
        if not context.is_floating_point() or not bool(torch.isfinite(context).all()):
            raise ValueError("diagnostic student context must be finite float images")
        target_fields = model.field.encode_unit(target_pixels)
        anchors: dict[int, torch.Tensor] = {}
        ranks: dict[int, torch.Tensor] = {}
        margins: dict[int, torch.Tensor] = {}

        for length in PREDICTIVE_STATE_CONTEXT_LENGTHS:
            with _autocast(device, precision):
                anchor = model.language(context[:, -length:])["anchor_fields"][:, -1]
            anchor = anchor.float()
            logits = (
                model.contrastive_scale.float()
                * anchor
                @ bank_fields.float().transpose(0, 1)
            )
            rank, margin = _score_batch(
                context_accumulators[length],
                logits,
                targets,
                anchor,
                target_fields,
            )
            anchors[length] = anchor
            ranks[length] = rank
            margins[length] = margin

        for length in PREDICTIVE_STATE_SHUFFLED_LENGTHS:
            value = shuffle_prefix_preserving_suffix(
                context[:, -length:],
                first_index=examples,
                seed=shuffle_seed,
            )
            with _autocast(device, precision):
                anchor = model.language(value)["anchor_fields"][:, -1].float()
            logits = (
                model.contrastive_scale.float()
                * anchor
                @ bank_fields.float().transpose(0, 1)
            )
            _score_batch(
                shuffled_accumulators[length],
                logits,
                targets,
                anchor,
                target_fields,
            )

        for length, comparisons in interventions.items():
            _accumulate_intervention(
                comparisons["vs_last"],
                anchor=anchors[length],
                baseline_anchor=anchors[1],
                rank=ranks[length],
                baseline_rank=ranks[1],
                margin=margins[length],
                baseline_margin=margins[1],
            )
            if length > 4:
                _accumulate_intervention(
                    comparisons["vs_suffix4"],
                    anchor=anchors[length],
                    baseline_anchor=anchors[4],
                    rank=ranks[length],
                    baseline_rank=ranks[4],
                    margin=margins[length],
                    baseline_margin=margins[4],
                )
        examples += int(context.shape[0])

    context_curve = {
        str(length): _finish_metrics(context_accumulators[length])
        for length in PREDICTIVE_STATE_CONTEXT_LENGTHS
    }
    shuffled_curve = {
        str(length): _finish_metrics(shuffled_accumulators[length])
        for length in PREDICTIVE_STATE_SHUFFLED_LENGTHS
    }
    ordered_minus_shuffled: dict[str, dict[str, float]] = {}
    for length in PREDICTIVE_STATE_SHUFFLED_LENGTHS:
        ordered = context_curve[str(length)]
        shuffled = shuffled_curve[str(length)]
        ordered_minus_shuffled[str(length)] = {
            key: ordered[key] - shuffled[key]
            for key in (
                "top1",
                "top5",
                "target_log_probability",
                "target_cosine",
                "prediction_entropy",
                "mean_reciprocal_rank",
            )
        }
    return {
        "examples": float(examples),
        "context_curve": context_curve,
        "shuffled_curve": shuffled_curve,
        "ordered_minus_shuffled": ordered_minus_shuffled,
        "anchor_interventions": {
            str(length): {
                name: _finish_intervention(accumulator)
                for name, accumulator in comparisons.items()
            }
            for length, comparisons in interventions.items()
        },
    }


@torch.no_grad()
def evaluate_context_length_curve(
    model: Any,
    loader: Iterable[dict[str, Any]],
    bank_fields: torch.Tensor,
    *,
    lengths: Sequence[int],
    device: torch.device,
    precision: str,
) -> dict[str, dict[str, float]]:
    """Score an evaluator-declared context-length intervention."""

    ordered_lengths = tuple(int(length) for length in lengths)
    if (
        not ordered_lengths
        or len(set(ordered_lengths)) != len(ordered_lengths)
        or any(length < 1 or length > 64 for length in ordered_lengths)
    ):
        raise ValueError("diagnostic context lengths must be unique values in [1,64]")
    model.eval()
    accumulators = {
        length: _empty_metric_accumulator() for length in ordered_lengths
    }
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        targets = raw["target_index"].to(device, non_blocking=True)
        target_pixels = raw["continuation"][:, 0].to(device, non_blocking=True)
        if context.ndim != 5 or tuple(context.shape[-3:]) != (1, 32, 32):
            raise ValueError("diagnostic student context is not a raster stream")
        target_fields = model.field.encode_unit(target_pixels)
        for length in ordered_lengths:
            with _autocast(device, precision):
                anchor = model.language(context[:, -length:])[
                    "anchor_fields"
                ][:, -1].float()
            logits = (
                model.contrastive_scale.float()
                * anchor
                @ bank_fields.float().transpose(0, 1)
            )
            _score_batch(
                accumulators[length],
                logits,
                targets,
                anchor,
                target_fields,
            )
    return {
        str(length): _finish_metrics(accumulators[length])
        for length in ordered_lengths
    }


@torch.no_grad()
def collect_direct_actuator_predictions(
    model: Any,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, torch.Tensor]:
    """Collect V42's final visual proposal before its learned sampler."""

    model.eval()
    anchors: list[torch.Tensor] = []
    signed_images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_indices: list[torch.Tensor] = []
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        target = raw["continuation"][:, 0].to(device, non_blocking=True)
        target_index = raw["target_index"].to(device, non_blocking=True)
        if context.ndim != 5 or tuple(context.shape[-3:]) != (1, 32, 32):
            raise ValueError("direct actuator context must be [B,T,1,32,32]")
        if target.shape != (len(context), 1, 32, 32):
            raise ValueError("direct actuator target must be [B,1,32,32]")
        with _autocast(device, precision):
            anchor = model.language(context)["anchor_fields"][:, -1].float()
        field_dim = int(anchor.shape[-1])
        if field_dim != 1_024:
            raise ValueError("direct actuator requires the V42 1,024-D field")
        signed = model.field.signed_spatial(anchor * math.sqrt(field_dim))
        anchors.append(anchor)
        signed_images.append(signed)
        targets.append(target.float())
        target_indices.append(target_index)
    if not anchors:
        raise ValueError("direct actuator loader is empty")
    output = {
        "anchor_fields": torch.cat(anchors),
        "signed_images": torch.cat(signed_images),
        "target_pixels": torch.cat(targets),
        "target_indices": torch.cat(target_indices),
    }
    if not all(bool(torch.isfinite(value).all()) for value in output.values()):
        raise ValueError("direct actuator collection contains non-finite values")
    return output


def _binary_image_statistics(
    signed_images: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, torch.Tensor]:
    if signed_images.shape != target_pixels.shape:
        raise ValueError("direct actuator signed images and targets must align")
    if signed_images.ndim != 4 or tuple(signed_images.shape[1:]) != (1, 32, 32):
        raise ValueError("direct actuator images must be [N,1,32,32]")
    predicted = signed_images >= float(threshold)
    target = target_pixels >= 0.5
    true_positive = (predicted & target).flatten(1).sum(dim=1).float()
    precision = true_positive / predicted.flatten(1).sum(dim=1).clamp_min(1)
    recall = true_positive / target.flatten(1).sum(dim=1).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    predicted_density = predicted.flatten(1).sum(dim=1).float()
    target_density = target.flatten(1).sum(dim=1).float().clamp_min(1)
    return {
        "pixels": predicted.float(),
        "pixel_f1": f1,
        "blank": predicted_density < 2,
        "ink_density_ratio": predicted_density / target_density,
    }


def select_direct_actuator_threshold(
    signed_images: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    thresholds: Sequence[float] = DIRECT_ACTUATOR_THRESHOLDS,
) -> dict[str, float]:
    """Select one global threshold on evaluator-declared training imagery."""

    candidates = tuple(float(value) for value in thresholds)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("direct actuator thresholds must be nonempty and unique")
    rows: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        statistics = _binary_image_statistics(
            signed_images,
            target_pixels,
            threshold=threshold,
        )
        rows.append(
            (
                float(statistics["pixel_f1"].mean()),
                -abs(threshold),
                -threshold,
                threshold,
            )
        )
    selected = max(rows)
    statistics = _binary_image_statistics(
        signed_images,
        target_pixels,
        threshold=selected[3],
    )
    return {
        "threshold": selected[3],
        "pixel_f1": float(statistics["pixel_f1"].mean()),
        "blank_rate": float(statistics["blank"].float().mean()),
        "ink_density_ratio": float(statistics["ink_density_ratio"].mean()),
        "candidate_count": float(len(candidates)),
    }


@torch.no_grad()
def evaluate_direct_actuator_predictions(
    model: Any,
    predictions: Mapping[str, torch.Tensor],
    bank_fields: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    """Decode a continuous proposal directly and score its visible reread."""

    required = {
        "anchor_fields",
        "signed_images",
        "target_pixels",
        "target_indices",
    }
    if set(predictions) != required:
        raise ValueError("direct actuator prediction payload changed")
    anchors = predictions["anchor_fields"].float()
    target_pixels = predictions["target_pixels"].float()
    target_indices = predictions["target_indices"]
    statistics = _binary_image_statistics(
        predictions["signed_images"],
        target_pixels,
        threshold=threshold,
    )
    visible_pixels = statistics["pixels"]
    visible_fields = model.field.encode_unit(visible_pixels)
    target_fields = model.field.encode_unit(target_pixels)
    bank = bank_fields.float()
    if bank.ndim != 2 or bank.shape[1] != anchors.shape[1]:
        raise ValueError("direct actuator evaluator bank does not align")
    anchor_prediction = (anchors @ bank.transpose(0, 1)).argmax(dim=1)
    visible_prediction = (visible_fields @ bank.transpose(0, 1)).argmax(dim=1)
    examples = len(anchors)
    output = {
        "examples": float(examples),
        "threshold": float(threshold),
        "anchor_identity_top1": float(
            (anchor_prediction == target_indices).float().mean()
        ),
        "visible_identity_top1": float(
            (visible_prediction == target_indices).float().mean()
        ),
        "visible_pixel_f1": float(statistics["pixel_f1"].mean()),
        "visible_blank_rate": float(statistics["blank"].float().mean()),
        "visible_ink_density_ratio": float(
            statistics["ink_density_ratio"].mean()
        ),
        "proposal_visible_reread_cosine": float(
            (anchors * visible_fields).sum(dim=1).mean()
        ),
        "proposal_target_cosine": float(
            (anchors * target_fields).sum(dim=1).mean()
        ),
        "visible_target_cosine": float(
            (visible_fields * target_fields).sum(dim=1).mean()
        ),
        "rereads_visible_pixels": 1.0,
    }
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("direct actuator metrics are non-finite")
    return output


@torch.no_grad()
def field_geometry(bank_fields: torch.Tensor) -> dict[str, float]:
    """Describe an evaluator bank's continuous geometry without language claims."""

    if bank_fields.ndim != 2 or min(bank_fields.shape) < 2:
        raise ValueError("field geometry requires at least two bank fields")
    fields = bank_fields.detach().float()
    if not bool(torch.isfinite(fields).all()):
        raise ValueError("field geometry received non-finite values")
    norms = fields.norm(dim=1)
    unit = torch.nn.functional.normalize(fields, dim=1)
    gram = unit @ unit.transpose(0, 1)
    width = gram.shape[0]
    diagonal = torch.eye(width, dtype=torch.bool, device=gram.device)
    off_diagonal = gram[~diagonal]
    masked = gram.masked_fill(diagonal, -torch.inf)
    nearest_cosine, nearest_index = masked.max(dim=1)
    hub_counts = torch.bincount(nearest_index, minlength=width)

    centered = unit - unit.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    variance_probability = variance / variance.sum().clamp_min(1e-12)
    entropy = -(
        variance_probability
        * variance_probability.clamp_min(1e-30).log()
    ).sum()
    cumulative = variance_probability.cumsum(dim=0)
    quantiles = torch.quantile(
        off_diagonal,
        torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=fields.device),
    )
    self_top = gram.argmax(dim=1)
    strict_self = gram.diagonal() > masked.max(dim=1).values
    output = {
        "bank_size": float(width),
        "field_dimension": float(fields.shape[1]),
        "field_norm_mean": float(norms.mean()),
        "field_norm_standard_deviation": float(norms.std(unbiased=False)),
        "centroid_norm": float(unit.mean(dim=0).norm()),
        "off_diagonal_cosine_mean": float(off_diagonal.mean()),
        "off_diagonal_cosine_standard_deviation": float(
            off_diagonal.std(unbiased=False)
        ),
        "off_diagonal_cosine_q01": float(quantiles[0]),
        "off_diagonal_cosine_q05": float(quantiles[1]),
        "off_diagonal_cosine_q50": float(quantiles[2]),
        "off_diagonal_cosine_q95": float(quantiles[3]),
        "off_diagonal_cosine_q99": float(quantiles[4]),
        "nearest_neighbor_cosine_mean": float(nearest_cosine.mean()),
        "self_to_nearest_margin_mean": float((1.0 - nearest_cosine).mean()),
        "self_retrieval_top1": float(
            (self_top == torch.arange(width, device=gram.device)).float().mean()
        ),
        "strict_self_retrieval_fraction": float(strict_self.float().mean()),
        "nearest_neighbor_distinct_hubs": float((hub_counts > 0).sum()),
        "nearest_neighbor_max_hub_fraction": float(hub_counts.max()) / width,
        "centered_effective_rank": float(entropy.exp()),
        "pc1_variance_fraction": float(variance_probability[0]),
        "pc4_cumulative_variance_fraction": float(cumulative[min(3, len(cumulative) - 1)]),
        "pc16_cumulative_variance_fraction": float(
            cumulative[min(15, len(cumulative) - 1)]
        ),
        "pc64_cumulative_variance_fraction": float(
            cumulative[min(63, len(cumulative) - 1)]
        ),
    }
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("field geometry produced a non-finite measurement")
    return output


def partition_generalization_gaps(
    model_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return development-minus-train curves for each measured model."""

    output: dict[str, dict[str, float]] = {}
    for model_name, result in model_results.items():
        train = result["partitions"]["train_partition"]["context_curve"]
        development = result["partitions"]["development_partition"][
            "context_curve"
        ]
        gaps: dict[str, float] = {}
        for length in PREDICTIVE_STATE_CONTEXT_LENGTHS:
            key = str(length)
            for metric in ("top1", "target_log_probability"):
                gaps[f"context_{key}_{metric}"] = (
                    development[key][metric] - train[key][metric]
                )
        output[model_name] = gaps
    return output


__all__ = [
    "DIRECT_ACTUATOR_THRESHOLDS",
    "PREDICTIVE_STATE_CONTEXT_LENGTHS",
    "PREDICTIVE_STATE_SHUFFLED_LENGTHS",
    "audit_window_digest",
    "build_partition_audit_windows",
    "collect_direct_actuator_predictions",
    "evaluate_context_length_curve",
    "evaluate_direct_actuator_predictions",
    "evaluate_predictive_state",
    "field_geometry",
    "partition_generalization_gaps",
    "select_direct_actuator_threshold",
    "shuffle_prefix_preserving_suffix",
]

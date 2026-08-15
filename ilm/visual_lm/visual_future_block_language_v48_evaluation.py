from __future__ import annotations

import contextlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from .ink_jepa_data import VisualGrammarRecord
from .visual_cell_data import iter_split_writing
from .visual_cell_eval_data import VisualCharacterStatistics
from .visual_future_block_language_v48 import (
    VisualFutureBlockLanguageModelV48,
    visual_future_block_language_boundary_receipt_v48,
)


V48_AUDIT_SEED = 20264820
V48_GATE_EPSILON = 1e-12
V48_FROZEN_V42_FULL_TOP1 = 0.19970703125
V48_EXPECTED_OFFSET_TOP1 = (
    0.107421875,
    0.0361328125,
    0.02685546875,
    0.03173828125,
)


def _autocast(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("V48 precision must be fp32, fp16, or bf16")
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _empty_retrieval_accumulator() -> dict[str, Any]:
    return {
        "examples": 0,
        "correct_top1": 0.0,
        "correct_top5": 0.0,
        "target_log_probability_sum": 0.0,
        "target_cosine_sum": 0.0,
        "predictions": Counter(),
    }


def _accumulate_retrieval(
    accumulator: dict[str, Any],
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    anchors: torch.Tensor,
    target_fields: torch.Tensor,
) -> None:
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("V48 retrieval logits and targets do not align")
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    target_log_probability = logits.log_softmax(dim=1).gather(
        1, targets[:, None]
    )[:, 0]
    accumulator["examples"] += int(targets.shape[0])
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
    accumulator["predictions"].update(top[:, 0].detach().cpu().tolist())


def _finish_retrieval(accumulator: Mapping[str, Any]) -> dict[str, float]:
    examples = int(accumulator["examples"])
    if examples < 1:
        raise ValueError("V48 retrieval accumulator is empty")
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
        "distinct_top1_predictions": float(len(predictions)),
        "most_common_top1_fraction": float(most_common) / examples,
    }


def build_offset_conditional_counts_v48(
    records: Sequence[VisualGrammarRecord],
    statistics: VisualCharacterStatistics,
    *,
    script_views_mode: str,
) -> tuple[dict[str, tuple[tuple[int, int], ...]], ...]:
    """Build evaluator-only training counts at visual offsets one through four."""

    character_index = statistics.index
    rows: list[defaultdict[str, Counter[int]]] = [
        defaultdict(Counter) for _ in range(4)
    ]
    for _record, _script_view, writing in iter_split_writing(
        records,
        split="train",
        script_views_mode=script_views_mode,
    ):
        for position, previous in enumerate(writing[:-1]):
            for horizon in range(1, 5):
                target_position = position + horizon
                if target_position >= len(writing):
                    break
                target = character_index.get(writing[target_position])
                if target is not None:
                    rows[horizon - 1][previous][target] += 1
    return tuple(
        {
            previous: tuple(sorted(counter.items()))
            for previous, counter in sorted(horizon_rows.items())
        }
        for horizon_rows in rows
    )


def evaluate_offset_conditional_control_v48(
    counts: Sequence[Mapping[str, Sequence[tuple[int, int]]]],
    statistics: VisualCharacterStatistics,
    loader: Iterable[dict[str, Any]],
    *,
    alpha: float = 0.10,
) -> dict[str, dict[str, float]]:
    """Score host-only offset controls without exposing labels to the student."""

    if len(counts) != 4 or alpha <= 0.0:
        raise ValueError("V48 offset control requires four horizons and alpha > 0")
    width = len(statistics.characters)
    unigram = torch.tensor(statistics.counts, dtype=torch.float64) + alpha
    unigram /= unigram.sum()
    totals = [
        {
            "examples": 0.0,
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
        }
        for _ in range(4)
    ]
    for raw in loader:
        targets = raw["target_indices"]
        previous_characters = raw["last_character"]
        if targets.ndim != 2 or targets.shape[1] != 4:
            raise ValueError("V48 offset targets must be [B,4]")
        if len(previous_characters) != len(targets):
            raise ValueError("V48 offset contexts and targets do not align")
        for row_index, previous in enumerate(previous_characters):
            for horizon in range(4):
                sparse = counts[horizon].get(previous)
                if sparse:
                    probability = torch.full((width,), alpha, dtype=torch.float64)
                    for index, count in sparse:
                        probability[index] += count
                    probability /= probability.sum()
                else:
                    probability = unigram
                target = int(targets[row_index, horizon])
                # ``argmax`` has a stable lowest-index tie convention.  PyTorch's
                # top-k kernel is intentionally not required to preserve that
                # convention, so it is used only for the set-valued top-5 metric.
                prediction = probability.argmax()
                top = probability.topk(min(5, width)).indices
                totals[horizon]["examples"] += 1.0
                totals[horizon]["correct_top1"] += float(prediction == target)
                totals[horizon]["correct_top5"] += float((top == target).any())
                totals[horizon]["target_log_probability_sum"] += math.log(
                    float(probability[target])
                )
    output: dict[str, dict[str, float]] = {}
    for horizon, total in enumerate(totals, start=1):
        examples = total["examples"]
        if examples < 1:
            raise ValueError("V48 offset-control loader is empty")
        output[str(horizon)] = {
            "examples": examples,
            "top1": total["correct_top1"] / examples,
            "top5": total["correct_top5"] / examples,
            "target_log_probability": (
                total["target_log_probability_sum"] / examples
            ),
        }
    return output


@torch.no_grad()
def evaluate_four_future_fields_v48(
    model: VisualFutureBlockLanguageModelV48,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """Score all four direct visual forecasts against an evaluator-only bank."""

    model.eval()
    bank_fields = model.field.encode_unit(bank_images.to(device))
    accumulators = [_empty_retrieval_accumulator() for _ in range(4)]
    examples = 0
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        target_pixels = raw["future_pixels"].to(device, non_blocking=True)
        targets = raw["target_indices"].to(device, non_blocking=True)
        if target_pixels.shape != (len(context), 4, 1, 32, 32):
            raise ValueError("V48 future evaluator targets must be [B,4,1,32,32]")
        with _autocast(device, precision):
            anchors = model.language(context)["future_anchor_fields"][:, -1]
        target_fields = model.field.encode_unit(target_pixels)
        for horizon in range(4):
            anchor = anchors[:, horizon].float()
            logits = (
                model.contrastive_scale.float()
                * anchor
                @ bank_fields.float().transpose(0, 1)
            )
            _accumulate_retrieval(
                accumulators[horizon],
                logits=logits,
                targets=targets[:, horizon],
                anchors=anchor,
                target_fields=target_fields[:, horizon],
            )
        examples += len(context)
    return {
        "examples": float(examples),
        "horizons": {
            str(horizon): _finish_retrieval(accumulators[horizon - 1])
            for horizon in range(1, 5)
        },
        "candidate_bank_used_after_prediction": True,
    }


def _pixel_statistics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if predicted.shape != target.shape or predicted.ndim != 5:
        raise ValueError("V48 rollout images must align as [B,H,1,32,32]")
    predicted_ink = predicted >= 0.5
    target_ink = target >= 0.5
    true_positive = (predicted_ink & target_ink).flatten(2).sum(dim=2).float()
    precision = true_positive / predicted_ink.flatten(2).sum(dim=2).clamp_min(1)
    recall = true_positive / target_ink.flatten(2).sum(dim=2).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    density = predicted_ink.flatten(2).sum(dim=2)
    return {"pixel_f1": f1, "blank": density < 2}


@torch.no_grad()
def evaluate_closed_loop_generation_v48(
    model: VisualFutureBlockLanguageModelV48,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int = 256,
) -> dict[str, Any]:
    """Generate four visible cells, feeding back only reread raster outputs."""

    if maximum_examples < 1:
        raise ValueError("V48 closed-loop example limit must be positive")
    model.eval()
    generated_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    target_index_batches: list[torch.Tensor] = []
    examples = 0
    recurrent_visible = True
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target_pixels = raw["future_pixels"][:remaining].to(
            device, non_blocking=True
        )
        target_indices = raw["target_indices"][:remaining].to(
            device, non_blocking=True
        )
        with _autocast(device, precision):
            _sequence, trace = model.generate(context, new_cells=4)
        generated = trace["generated_cells"].float()
        recurrent_visible = recurrent_visible and bool(
            trace["rereads_generated_pixels"].item()
        )
        generated_batches.append(generated)
        target_batches.append(target_pixels.float())
        target_index_batches.append(target_indices)
        examples += len(context)
    if examples < 1:
        raise ValueError("V48 closed-loop loader is empty")

    # Generation is complete before any evaluator-bank field is constructed.
    generated = torch.cat(generated_batches)
    target_pixels = torch.cat(target_batches)
    target_indices = torch.cat(target_index_batches)
    finite_cells = torch.isfinite(generated).flatten(2).all(dim=2)
    safe_generated = torch.nan_to_num(generated, nan=0.0, posinf=1.0, neginf=0.0)
    pixel = _pixel_statistics(safe_generated, target_pixels)
    bank_fields = model.field.encode_unit(bank_images.to(device))
    visible_fields = model.field.encode_unit(safe_generated)
    predicted_indices = torch.einsum(
        "bhd,kd->bhk", visible_fields.float(), bank_fields.float()
    ).argmax(dim=2)
    identity = (predicted_indices == target_indices).float()
    steps: dict[str, dict[str, float]] = {}
    for horizon in range(4):
        steps[str(horizon + 1)] = {
            "identity_top1": float(identity[:, horizon].mean()),
            "pixel_f1": float(pixel["pixel_f1"][:, horizon].mean()),
            "blank_rate": float(pixel["blank"][:, horizon].float().mean()),
            "finite_fraction": float(finite_cells[:, horizon].float().mean()),
        }
    return {
        "examples": float(examples),
        "steps": steps,
        "mean_identity_top1": float(identity.mean()),
        "mean_pixel_f1": float(pixel["pixel_f1"].mean()),
        "blank_outputs": float(pixel["blank"].sum()),
        "nonfinite_outputs": float((~finite_cells).sum()),
        "generated_before_candidate_bank_scoring": True,
        "recurrent_cells_are_visible_rasters": recurrent_visible,
        "candidate_bank_deployed": False,
    }


def visual_future_block_language_boundary_is_clean_v48(
    model: VisualFutureBlockLanguageModelV48,
) -> bool:
    receipt = visual_future_block_language_boundary_receipt_v48(model)
    required_true = {
        "input_is_continuous_image_stream",
        "output_is_continuous_image_block",
        "output_is_direct_raster",
        "field_transform_is_fixed_and_invertible",
        "inverse_dct_threshold_is_fixed_zero",
        "causal_over_visual_time",
        "predicts_four_future_images_densely",
        "rereads_generated_pixels",
    }
    required_false = {
        "uses_stochastic_generator",
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_quantization",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    }
    return (
        receipt["architecture"] == "visual-future-block-language-v48"
        and not receipt["parameter_names_with_forbidden_fragments"]
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def finite_metric_tree_v48(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_metric_tree_v48(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_metric_tree_v48(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def visual_future_block_gate_report_v48(
    language: Mapping[str, float],
    future: Mapping[str, Any],
    pairs: Mapping[str, float],
    terminal: Mapping[str, Mapping[str, float]],
    direct: Mapping[str, float],
    closed_loop: Mapping[str, Any],
    *,
    boundary_clean: bool,
    trainable_parameters: int,
    peak_allocated_vram_gib: float,
    training_elapsed_seconds: float,
    matched_v42_full_top1: float,
) -> dict[str, bool]:
    """Apply the 16 frozen V48 development gates without reinterpretation."""

    def above(value: float, threshold: float) -> bool:
        return value - threshold > V48_GATE_EPSILON

    def below(value: float, threshold: float) -> bool:
        return threshold - value > V48_GATE_EPSILON

    def at_least(value: float, threshold: float) -> bool:
        return value + V48_GATE_EPSILON >= threshold

    def at_most(value: float, threshold: float) -> bool:
        return value <= threshold + V48_GATE_EPSILON

    horizons = future["horizons"]
    offset_control = future["offset_conditional_control"]
    horizon_advantage = all(
        above(
            float(horizons[str(index)]["top1"])
            - float(offset_control[str(index)]["top1"]),
            0.01,
        )
        for index in range(1, 5)
    )
    horizon_diversity = all(
        at_least(
            float(horizons[str(index)]["distinct_top1_predictions"]),
            128.0,
        )
        and at_most(
            float(horizons[str(index)]["most_common_top1_fraction"]),
            0.15,
        )
        for index in range(1, 5)
    )
    proposal_identity = float(direct["anchor_identity_top1"])
    visible_identity = float(direct["visible_identity_top1"])
    retention = visible_identity / max(proposal_identity, V48_GATE_EPSILON)
    return {
        "01_image_only_boundary_clean": bool(boundary_clean),
        "02_trainable_parameters_below_17m": trainable_parameters < 17_000_000,
        "03_peak_allocated_vram_below_18_gib": below(
            peak_allocated_vram_gib, 18.0
        ),
        "04_training_under_7200_seconds": below(training_elapsed_seconds, 7_200.0),
        "05_full_top1_beats_v42_by_0_005": above(
            float(language["full_top1"]) - matched_v42_full_top1, 0.005
        ),
        "06_full_top1_beats_bigram_by_0_03": above(
            float(language["full_top1"]) - float(language["bigram_top1"]),
            0.03,
        ),
        "07_ordered_top1_beats_shuffled_by_0_015": above(
            float(language["full_top1"])
            - float(language["shuffled_top1"]),
            0.015,
        ),
        "08_ordered_logp_beats_shuffled_by_0_10": above(
            float(language["full_target_log_probability"])
            - float(language["shuffled_target_log_probability"]),
            0.10,
        ),
        "09_all_horizons_beat_offset_control": horizon_advantage,
        "10_all_horizons_preserve_diversity": horizon_diversity,
        "11_counterfactual_arm_accuracy_above_0_55": above(
            float(pairs["full_arm_accuracy"]), 0.55
        ),
        "12_no_terminal_position_collapse": (
            at_least(
                float(terminal["64"]["top1"]),
                float(terminal["63"]["top1"]) - 0.01,
            )
            and below(
                float(terminal["64"]["most_common_top1_fraction"]),
                0.10,
            )
        ),
        "13_direct_visible_identity_above_0_15": above(visible_identity, 0.15),
        "14_direct_raster_f1_and_nonblank": (
            above(float(direct["visible_pixel_f1"]), 0.46)
            and below(float(direct["visible_blank_rate"]), 0.02)
        ),
        "15_direct_raster_retains_proposal": (
            at_least(retention, 0.85)
            and above(
                float(direct["proposal_visible_reread_cosine"]), 0.82
            )
        ),
        "16_closed_loop_beats_unigram_and_is_integral": (
            above(
                float(closed_loop["mean_identity_top1"])
                - float(language["unigram_top1"]),
                0.01,
            )
            and float(closed_loop["nonfinite_outputs"]) == 0.0
            and float(closed_loop["blank_outputs"]) == 0.0
            and closed_loop["recurrent_cells_are_visible_rasters"] is True
        ),
    }


__all__ = [
    "V48_AUDIT_SEED",
    "V48_EXPECTED_OFFSET_TOP1",
    "V48_FROZEN_V42_FULL_TOP1",
    "build_offset_conditional_counts_v48",
    "evaluate_closed_loop_generation_v48",
    "evaluate_four_future_fields_v48",
    "evaluate_offset_conditional_control_v48",
    "finite_metric_tree_v48",
    "visual_future_block_gate_report_v48",
    "visual_future_block_language_boundary_is_clean_v48",
]

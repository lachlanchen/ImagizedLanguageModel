from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from .canonical_glyph_language_evaluation import (
    V42_GATE_EPSILON,
    _autocast,
    _baseline_metrics,
    _paired_metrics,
    _shuffle_prefix,
    _top_metrics,
    canonical_language_gate_report,
)
from .scaled_retinal_glyph_language_v46 import (
    V46_ARCHITECTURE,
    V46_REFERENCE_RADIUS,
    ScaledRetinalGlyphLanguageModelV46,
    scaled_retinal_glyph_language_v46_boundary_receipt,
)
from .visual_cell_eval_data import VisualCharacterStatistics


V46_AUDIT_SEED = 20264220
V46_V42_FULL_TOP1 = 0.19970703125
V46_V42_FULL_TARGET_LOG_PROBABILITY = -5.255309844389558
V46_V42_GENERATED_IDENTITY_TOP1 = 0.08203125
V46_REQUIRED_TRAINABLE_PARAMETERS = 24_346_497


def _candidate_logits(
    model: ScaledRetinalGlyphLanguageModelV46,
    anchors: torch.Tensor,
    bank_fields: torch.Tensor,
) -> torch.Tensor:
    return model.contrastive_scale.float() * (
        model.field.directions(anchors)
        @ model.field.directions(bank_fields).transpose(0, 1)
    )


@torch.no_grad()
def evaluate_scaled_retinal_language_v46(
    model: ScaledRetinalGlyphLanguageModelV46,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float = 0.0,
) -> dict[str, float]:
    model.eval()
    bank_fields = model.field.encode(bank_images.to(device))
    variants = ("full", "suffix4", "last", "shuffled", "blank")
    totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
            "target_cosine_sum": 0.0,
            "radius_absolute_error_sum": 0.0,
            "radius_relative_error_sum": 0.0,
        }
        for name in variants
    }
    all_targets: list[int] = []
    all_last_characters: list[str] = []
    examples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        targets = raw["target_index"].to(device, non_blocking=True)
        target_pixels = raw["continuation"][:, 0].to(device, non_blocking=True)
        target_fields = model.field.encode(target_pixels)
        target_directions = model.field.directions(target_fields)
        target_radius = target_fields.float().norm(dim=-1).clamp_min(1e-8)
        contexts = {
            "full": context,
            "suffix4": context[:, -4:],
            "last": context[:, -1:],
            "shuffled": _shuffle_prefix(context, first_index=examples),
            "blank": torch.zeros_like(context),
        }
        for name, value in contexts.items():
            with _autocast(device, precision):
                anchor = model.language(value)["anchor_fields"][:, -1]
            logits = _candidate_logits(model, anchor, bank_fields)
            metrics = _top_metrics(logits, targets)
            for key, metric in metrics.items():
                totals[name][key] += metric
            anchor_directions = model.field.directions(anchor)
            anchor_radius = anchor.float().norm(dim=-1)
            absolute = (anchor_radius - target_radius).abs()
            totals[name]["target_cosine_sum"] += float(
                (anchor_directions * target_directions).sum(dim=-1).sum()
            )
            totals[name]["radius_absolute_error_sum"] += float(absolute.sum())
            totals[name]["radius_relative_error_sum"] += float(
                (absolute / target_radius).sum()
            )
        all_targets.extend(targets.cpu().tolist())
        all_last_characters.extend(raw["last_character"])
        examples += len(targets)
    if examples < 1:
        raise ValueError("V46 language audit loader is empty")
    baseline = _baseline_metrics(
        statistics,
        all_targets,
        all_last_characters,
    )
    result: dict[str, float] = {"examples": float(examples)}
    for name in variants:
        result[f"{name}_top1"] = totals[name]["correct_top1"] / examples
        result[f"{name}_top5"] = totals[name]["correct_top5"] / examples
        result[f"{name}_target_log_probability"] = (
            totals[name]["target_log_probability_sum"] / examples
        )
        result[f"{name}_target_cosine"] = (
            totals[name]["target_cosine_sum"] / examples
        )
        result[f"{name}_radius_mae"] = (
            totals[name]["radius_absolute_error_sum"] / examples
        )
        result[f"{name}_relative_radius_mae"] = (
            totals[name]["radius_relative_error_sum"] / examples
        )
    for name in ("unigram", "bigram"):
        result[f"{name}_top1"] = baseline[f"{name}_correct_top1"] / examples
        result[f"{name}_top5"] = baseline[f"{name}_correct_top5"] / examples
        result[f"{name}_target_log_probability"] = (
            baseline[f"{name}_target_log_probability_sum"] / examples
        )
    evaluator_peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result["evaluator_peak_allocated_vram_gib"] = evaluator_peak
    result["peak_allocated_vram_gib"] = max(
        evaluator_peak,
        checkpoint_peak_vram_gib,
    )
    return result


@torch.no_grad()
def evaluate_scaled_retinal_counterfactual_pairs_v46(
    model: ScaledRetinalGlyphLanguageModelV46,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    variants = ("full", "suffix4", "last", "shuffled")
    totals = {
        name: {
            "arms": 0.0,
            "arm_correct": 0.0,
            "both_correct": 0.0,
            "ties": 0.0,
            "margin_sum": 0.0,
        }
        for name in variants
    }
    pairs = 0
    for raw in loader:
        contexts = raw["contexts"].to(device, non_blocking=True)
        candidates = raw["candidates"].to(device, non_blocking=True)
        assignment = raw["assignment"].to(device, non_blocking=True)
        batch = contexts.shape[0]
        candidate_fields = model.field.directions(model.field.encode(candidates))
        flattened = contexts.reshape(batch * 2, *contexts.shape[2:])
        variant_contexts = {
            "full": flattened,
            "suffix4": flattened[:, -4:],
            "last": flattened[:, -1:],
            "shuffled": _shuffle_prefix(flattened, first_index=pairs * 2),
        }
        for name, value in variant_contexts.items():
            with _autocast(device, precision):
                anchor = model.language(value)["anchor_fields"][:, -1]
            anchor = model.field.directions(anchor).reshape(batch, 2, -1)
            logits = model.contrastive_scale.float() * torch.einsum(
                "bqd,bkd->bqk",
                anchor,
                candidate_fields,
            )
            metrics = _paired_metrics(logits, assignment)
            for key, metric in metrics.items():
                totals[name][key] += metric
        pairs += batch
    if pairs < 1:
        raise ValueError("V46 pair audit loader is empty")
    result = {"pairs": float(pairs)}
    for name in variants:
        arms = totals[name]["arms"]
        result[f"{name}_arm_accuracy"] = totals[name]["arm_correct"] / arms
        result[f"{name}_both_correct_rate"] = (
            totals[name]["both_correct"] / pairs
        )
        result[f"{name}_tie_rate"] = totals[name]["ties"] / arms
        result[f"{name}_mean_margin"] = totals[name]["margin_sum"] / arms
    return result


@torch.no_grad()
def evaluate_scaled_retinal_generated_fields_v46(
    model: ScaledRetinalGlyphLanguageModelV46,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
    samples: int,
    noise_scale: float = 1.0,
) -> dict[str, float]:
    if maximum_examples < 1 or samples < 1:
        raise ValueError("V46 generated audit settings must be positive")
    model.eval()
    bank_directions = model.field.directions(
        model.field.encode(bank_images.to(device))
    )
    generator = torch.Generator(device=device).manual_seed(V46_AUDIT_SEED + 77)
    totals = {
        "correct": 0.0,
        "anchor_correct": 0.0,
        "pixel_f1": 0.0,
        "target_cosine": 0.0,
        "blank": 0.0,
        "ink_ratio": 0.0,
        "radius_absolute_error": 0.0,
        "radius_relative_error": 0.0,
        "anchor_radius_absolute_error": 0.0,
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        target_index = raw["target_index"][:remaining].to(
            device,
            non_blocking=True,
        )
        with _autocast(device, precision):
            pixels, trace = model.sample_next(
                context,
                samples=samples,
                generator=generator,
                noise_scale=noise_scale,
            )
        selected = trace["selected_fields"].float()
        anchor = trace["anchor_fields"].float()
        selected_direction = model.field.directions(selected)
        anchor_direction = model.field.directions(anchor)
        predicted_index = (
            selected_direction @ bank_directions.transpose(0, 1)
        ).argmax(dim=1)
        anchor_index = (
            anchor_direction @ bank_directions.transpose(0, 1)
        ).argmax(dim=1)
        target_binary = (
            target >= model.config.binary_threshold
        ).to(dtype=pixels.dtype)
        predicted_ink = pixels >= 0.5
        target_ink = target_binary >= 0.5
        true_positive = (predicted_ink & target_ink).flatten(1).sum(dim=1).float()
        precision_value = true_positive / predicted_ink.flatten(1).sum(dim=1).clamp_min(1)
        recall_value = true_positive / target_ink.flatten(1).sum(dim=1).clamp_min(1)
        f1 = 2.0 * precision_value * recall_value / (
            precision_value + recall_value
        ).clamp_min(1e-8)
        target_fields = model.field.encode(target)
        target_direction = model.field.directions(target_fields)
        target_radius = target_fields.float().norm(dim=-1).clamp_min(1e-8)
        selected_radius = selected.norm(dim=-1)
        anchor_radius = anchor.norm(dim=-1)
        predicted_density = predicted_ink.flatten(1).sum(dim=1).float()
        target_density = target_ink.flatten(1).sum(dim=1).float().clamp_min(1)
        totals["correct"] += float((predicted_index == target_index).sum())
        totals["anchor_correct"] += float((anchor_index == target_index).sum())
        totals["pixel_f1"] += float(f1.sum())
        totals["target_cosine"] += float(
            (selected_direction * target_direction).sum()
        )
        totals["blank"] += float((predicted_density < 2).sum())
        totals["ink_ratio"] += float((predicted_density / target_density).sum())
        absolute = (selected_radius - target_radius).abs()
        totals["radius_absolute_error"] += float(absolute.sum())
        totals["radius_relative_error"] += float((absolute / target_radius).sum())
        totals["anchor_radius_absolute_error"] += float(
            (anchor_radius - target_radius).abs().sum()
        )
        examples += len(context)
    if examples < 1:
        raise ValueError("V46 generated audit loader is empty")
    return {
        "examples": float(examples),
        "generated_identity_top1": totals["correct"] / examples,
        "anchor_identity_top1": totals["anchor_correct"] / examples,
        "generated_pixel_f1": totals["pixel_f1"] / examples,
        "generated_target_cosine": totals["target_cosine"] / examples,
        "generated_blank_rate": totals["blank"] / examples,
        "generated_ink_density_ratio": totals["ink_ratio"] / examples,
        "generated_radius_mae": totals["radius_absolute_error"] / examples,
        "generated_relative_radius_mae": (
            totals["radius_relative_error"] / examples
        ),
        "anchor_radius_mae": (
            totals["anchor_radius_absolute_error"] / examples
        ),
        "rereads_generated_pixels": 1.0,
    }


def scaled_retinal_language_v46_boundary_is_clean(
    model: ScaledRetinalGlyphLanguageModelV46,
) -> bool:
    receipt = scaled_retinal_glyph_language_v46_boundary_receipt(model)
    required_true = {
        "input_is_continuous_image_stream",
        "output_is_full_continuous_image_field",
        "output_is_direct_raster",
        "field_transform_is_fixed_and_invertible",
        "field_preserves_direction_and_radius",
        "causal_over_visual_time",
        "rereads_generated_pixels",
    }
    required_false = {
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
        receipt["architecture"] == V46_ARCHITECTURE
        and not receipt["parameter_names_with_forbidden_fragments"]
        and receipt["field_trainable_parameters"] == 0
        and abs(receipt["reference_radius"] - V46_REFERENCE_RADIUS) <= 1e-12
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def scaled_retinal_language_v46_gate_report(
    language: dict[str, float],
    pairs: dict[str, float],
    generated: dict[str, float],
    *,
    boundary_clean: bool,
    protocol_integrity_clean: bool,
) -> dict[str, bool]:
    gates = canonical_language_gate_report(
        language,
        pairs,
        generated,
        boundary_clean=boundary_clean,
    )

    def above(value: float, threshold: float) -> bool:
        return value - threshold > V42_GATE_EPSILON

    gates.update(
        {
            "full_top1_beats_v42_by_0_01": above(
                language["full_top1"] - V46_V42_FULL_TOP1,
                0.01,
            ),
            "full_log_probability_beats_v42_by_0_05": above(
                language["full_target_log_probability"]
                - V46_V42_FULL_TARGET_LOG_PROBABILITY,
                0.05,
            ),
            "generated_identity_beats_v42_by_0_01": above(
                generated["generated_identity_top1"]
                - V46_V42_GENERATED_IDENTITY_TOP1,
                0.01,
            ),
            "protocol_integrity_and_runtime": protocol_integrity_clean,
        }
    )
    return gates


def finite_metric_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_metric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_metric_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(float(value))))
    return True


@torch.no_grad()
def scaled_retinal_field_roundtrip_receipt(
    model: ScaledRetinalGlyphLanguageModelV46,
    banks: Sequence[torch.Tensor],
) -> dict[str, float | bool]:
    if not banks:
        raise ValueError("V46 round-trip audit requires raster banks")
    maximum_error = 0.0
    exact_pixels = 0
    total_pixels = 0
    all_finite = True
    for pixels in banks:
        fields = model.field.encode(pixels, exact=True)
        decoded = model.field.decode_dct(fields, exact=True)
        source = model.field.retinal.dct.encode(pixels).to(torch.float64)
        maximum_error = max(
            maximum_error,
            float((decoded - source).abs().max()),
        )
        binary = model.field.binary(fields, exact=True)
        target = (pixels >= model.config.binary_threshold).to(binary.dtype)
        exact_pixels += int((binary == target).sum())
        total_pixels += target.numel()
        all_finite = all_finite and bool(
            torch.isfinite(fields).all() and torch.isfinite(decoded).all()
        )
    return {
        "all_finite": all_finite,
        "maximum_dct_absolute_error": maximum_error,
        "binary_pixel_accuracy": exact_pixels / total_pixels,
        "reference_radius": V46_REFERENCE_RADIUS,
    }

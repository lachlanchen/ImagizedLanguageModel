from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from .canonical_glyph_language import (
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_boundary_receipt,
)
from .visual_cell_eval_data import VisualCharacterStatistics


V42_AUDIT_SEED = 20264220
V42_GATE_EPSILON = 1e-12


def _autocast(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _shuffle_prefix(
    context: torch.Tensor,
    *,
    first_index: int,
    suffix_cells: int = 4,
) -> torch.Tensor:
    if context.ndim != 5 or not 0 < suffix_cells < context.shape[1]:
        raise ValueError("V42 shuffle requires [B,T,1,32,32] and a proper suffix")
    shuffled = context.clone()
    prefix = context.shape[1] - suffix_cells
    permutations = []
    for offset in range(context.shape[0]):
        generator = torch.Generator().manual_seed(
            V42_AUDIT_SEED + (first_index + offset) * 104_729
        )
        permutations.append(torch.randperm(prefix, generator=generator))
    permutation = torch.stack(permutations).to(context.device)
    gather = permutation[:, :, None, None, None].expand(
        -1, -1, *context.shape[2:]
    )
    shuffled[:, :prefix] = context[:, :prefix].gather(1, gather)
    return shuffled


def _top_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probability = logits.log_softmax(dim=1).gather(1, targets[:, None])[:, 0]
    return {
        "correct_top1": float((top[:, 0] == targets).sum()),
        "correct_top5": float((top == targets[:, None]).any(dim=1).sum()),
        "target_log_probability_sum": float(log_probability.sum()),
    }


def _baseline_metrics(
    statistics: VisualCharacterStatistics,
    targets: Sequence[int],
    last_characters: Sequence[str],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    if len(targets) != len(last_characters):
        raise ValueError("V42 baseline targets and contexts must align")
    width = len(statistics.characters)
    unigram = torch.tensor(statistics.counts, dtype=torch.float64) + alpha
    unigram /= unigram.sum()
    unigram_top = unigram.topk(min(5, width)).indices
    output = {
        "unigram_correct_top1": 0.0,
        "unigram_correct_top5": 0.0,
        "unigram_target_log_probability_sum": 0.0,
        "bigram_correct_top1": 0.0,
        "bigram_correct_top5": 0.0,
        "bigram_target_log_probability_sum": 0.0,
    }
    for target, previous in zip(targets, last_characters):
        output["unigram_correct_top1"] += float(unigram_top[0] == target)
        output["unigram_correct_top5"] += float((unigram_top == target).any())
        output["unigram_target_log_probability_sum"] += math.log(
            float(unigram[target])
        )
        sparse = statistics.bigram_rows.get(previous)
        if sparse:
            row = torch.full((width,), alpha, dtype=torch.float64)
            for index, count in sparse:
                row[index] += count
            row /= row.sum()
        else:
            row = unigram
        bigram_top = row.topk(min(5, width)).indices
        output["bigram_correct_top1"] += float(bigram_top[0] == target)
        output["bigram_correct_top5"] += float((bigram_top == target).any())
        output["bigram_target_log_probability_sum"] += math.log(float(row[target]))
    return output


def _candidate_logits(
    anchors: torch.Tensor,
    bank_fields: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return scale.float() * anchors.float() @ bank_fields.float().transpose(0, 1)


@torch.no_grad()
def evaluate_canonical_language(
    model: CanonicalGlyphLanguageModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float = 0.0,
) -> dict[str, float]:
    model.eval()
    bank_fields = model.field.encode_unit(bank_images.to(device))
    variants = ("full", "suffix4", "last", "shuffled", "blank")
    totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
            "target_cosine_sum": 0.0,
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
        target_fields = model.field.encode_unit(target_pixels)
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
            logits = _candidate_logits(anchor, bank_fields, model.contrastive_scale)
            metrics = _top_metrics(logits, targets)
            for key, metric in metrics.items():
                totals[name][key] += metric
            totals[name]["target_cosine_sum"] += float(
                (anchor.float() * target_fields.float()).sum(dim=-1).sum()
            )
        all_targets.extend(targets.cpu().tolist())
        all_last_characters.extend(raw["last_character"])
        examples += len(targets)
    if examples < 1:
        raise ValueError("V42 language audit loader is empty")
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


def _paired_metrics(logits: torch.Tensor, assignment: torch.Tensor) -> dict[str, float]:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 2):
        raise ValueError("V42 paired logits must be [B,2,2]")
    correct = logits.gather(2, assignment[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignment)[:, :, None])[:, :, 0]
    margins = correct - other
    ties = margins == 0
    credit = (margins > 0).float() + 0.5 * ties.float()
    return {
        "arms": float(margins.numel()),
        "arm_correct": float(credit.sum()),
        "both_correct": float((margins > 0).all(dim=1).sum()),
        "ties": float(ties.sum()),
        "margin_sum": float(margins.sum()),
    }


@torch.no_grad()
def evaluate_counterfactual_pairs(
    model: CanonicalGlyphLanguageModel,
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
        candidate_fields = model.field.encode_unit(candidates)
        flattened = contexts.reshape(batch * 2, *contexts.shape[2:])
        variant_contexts = {
            "full": flattened,
            "suffix4": flattened[:, -4:],
            "last": flattened[:, -1:],
            "shuffled": _shuffle_prefix(
                flattened,
                first_index=pairs * 2,
            ),
        }
        for name, value in variant_contexts.items():
            with _autocast(device, precision):
                anchor = model.language(value)["anchor_fields"][:, -1]
            anchor = anchor.reshape(batch, 2, -1)
            logits = model.contrastive_scale.float() * torch.einsum(
                "bqd,bkd->bqk",
                anchor.float(),
                candidate_fields.float(),
            )
            metrics = _paired_metrics(logits, assignment)
            for key, metric in metrics.items():
                totals[name][key] += metric
        pairs += batch
    if pairs < 1:
        raise ValueError("V42 pair audit loader is empty")
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
def evaluate_generated_fields(
    model: CanonicalGlyphLanguageModel,
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
        raise ValueError("V42 generated audit settings must be positive")
    model.eval()
    bank_fields = model.field.encode_unit(bank_images.to(device))
    generator = torch.Generator(device=device).manual_seed(V42_AUDIT_SEED + 77)
    totals = {
        "correct": 0.0,
        "anchor_correct": 0.0,
        "pixel_f1": 0.0,
        "target_cosine": 0.0,
        "blank": 0.0,
        "ink_ratio": 0.0,
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        target_index = raw["target_index"][:remaining].to(device, non_blocking=True)
        with _autocast(device, precision):
            pixels, trace = model.sample_next(
                context,
                samples=samples,
                generator=generator,
                noise_scale=noise_scale,
            )
        selected = trace["selected_fields"].float()
        anchor = trace["anchor_fields"].float()
        predicted_index = (selected @ bank_fields.transpose(0, 1)).argmax(dim=1)
        anchor_index = (anchor @ bank_fields.transpose(0, 1)).argmax(dim=1)
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
        target_fields = model.field.encode_unit(target)
        predicted_density = predicted_ink.flatten(1).sum(dim=1).float()
        target_density = target_ink.flatten(1).sum(dim=1).float().clamp_min(1)
        totals["correct"] += float((predicted_index == target_index).sum())
        totals["anchor_correct"] += float((anchor_index == target_index).sum())
        totals["pixel_f1"] += float(f1.sum())
        totals["target_cosine"] += float((selected * target_fields).sum())
        totals["blank"] += float((predicted_density < 2).sum())
        totals["ink_ratio"] += float((predicted_density / target_density).sum())
        examples += len(context)
    if examples < 1:
        raise ValueError("V42 generated audit loader is empty")
    return {
        "examples": float(examples),
        "generated_identity_top1": totals["correct"] / examples,
        "anchor_identity_top1": totals["anchor_correct"] / examples,
        "generated_pixel_f1": totals["pixel_f1"] / examples,
        "generated_target_cosine": totals["target_cosine"] / examples,
        "generated_blank_rate": totals["blank"] / examples,
        "generated_ink_density_ratio": totals["ink_ratio"] / examples,
        "rereads_generated_pixels": 1.0,
    }


def canonical_language_boundary_is_clean(
    model: CanonicalGlyphLanguageModel,
) -> bool:
    receipt = canonical_glyph_language_boundary_receipt(model)
    required_true = {
        "input_is_continuous_image_stream",
        "output_is_continuous_image_field",
        "output_is_direct_raster",
        "field_transform_is_fixed_and_invertible",
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
        receipt["architecture"] == "canonical-glyph-language-v42"
        and not receipt["parameter_names_with_forbidden_fragments"]
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def canonical_language_gate_report(
    language: dict[str, float],
    pairs: dict[str, float],
    generated: dict[str, float],
    *,
    boundary_clean: bool,
) -> dict[str, bool]:
    def above(value: float, threshold: float) -> bool:
        return value - threshold > V42_GATE_EPSILON

    def below(value: float, threshold: float) -> bool:
        return threshold - value > V42_GATE_EPSILON

    return {
        "full_top1_gain_over_unigram": above(
            language["full_top1"] - language["unigram_top1"], 0.03
        ),
        "full_top1_gain_over_bigram": above(
            language["full_top1"] - language["bigram_top1"], 0.01
        ),
        "ordered_log_probability_gain_over_shuffled": above(
            language["full_target_log_probability"]
            - language["shuffled_target_log_probability"],
            0.05,
        ),
        "ordered_top1_gain_over_shuffled": above(
            language["full_top1"] - language["shuffled_top1"], 0.015
        ),
        "counterfactual_arm_accuracy": above(
            pairs["full_arm_accuracy"], 0.60
        ),
        "generated_identity_beats_unigram": above(
            generated["generated_identity_top1"], language["unigram_top1"]
        ),
        "generated_pixel_f1": above(generated["generated_pixel_f1"], 0.55),
        "generated_blank_rate": below(generated["generated_blank_rate"], 0.02),
        "student_boundary_clean": boundary_clean,
        "peak_allocated_vram_below_18_gib": below(
            language["peak_allocated_vram_gib"], 18.0
        ),
    }

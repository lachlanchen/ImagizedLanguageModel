from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .canonical_glyph_language_evaluation import (
    V42_GATE_EPSILON,
    _autocast,
    evaluate_canonical_language,
    evaluate_counterfactual_pairs,
)
from .codec_spherical_glyph_language_v47 import (
    V47_ARCHITECTURE,
    V47_REQUIRED_CODEC_CHECKPOINT_SHA256,
    V47_REQUIRED_CODEC_STATE_SHA256,
    CodecSphericalGlyphLanguageModelV47,
    codec_spherical_glyph_language_v47_boundary_receipt,
)
from .visual_cell_eval_data import VisualCharacterStatistics


V47_AUDIT_SEED = 20264220
V47_V46_FULL_TOP1 = 0.20751953125
V47_V46_GENERATED_IDENTITY_TOP1 = 0.0859375
V47_TOTAL_PARAMETER_LIMIT = 32_000_000
V47_TRAINABLE_PARAMETER_LIMIT = 25_000_000


def _micro_ink_f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted_ink = predicted >= 0.5
    target_ink = target >= 0.5
    true_positive = float((predicted_ink & target_ink).sum())
    predicted_positive = float(predicted_ink.sum())
    target_positive = float(target_ink.sum())
    return 2.0 * true_positive / max(1.0, predicted_positive + target_positive)


@torch.no_grad()
def codec_spherical_field_preflight_v47(
    model: CodecSphericalGlyphLanguageModelV47,
    canonical_images: torch.Tensor,
    held_font_images: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if tuple(canonical_images.shape[1:]) != (1, 32, 32):
        raise ValueError("V47 preflight canonical bank must be [N,1,32,32]")
    expected_fonts = {"noto_sans_cjk_bold", "noto_serif_cjk_medium"}
    if set(held_font_images) != expected_fonts:
        raise ValueError("V47 preflight requires both frozen held-font banks")
    model.eval()
    canonical = canonical_images.to(device)
    canonical_fields = model.field.encode(canonical)
    reconstructed = model.field.binary(canonical_fields)
    reread = model.field.encode(reconstructed)
    reread_cosine = (canonical_fields * reread).sum(dim=-1)
    retrieval: dict[str, dict[str, float]] = {}
    all_finite = bool(
        torch.isfinite(canonical_fields).all()
        and torch.isfinite(reconstructed).all()
        and torch.isfinite(reread).all()
    )
    for name in sorted(held_font_images):
        images = held_font_images[name]
        if images.shape != canonical_images.shape:
            raise ValueError(f"V47 held-font bank {name!r} does not align")
        query = model.field.encode(images.to(device))
        similarities = query @ canonical_fields.transpose(0, 1)
        top = similarities.topk(min(5, len(canonical_fields)), dim=1).indices
        expected = torch.arange(len(query), device=device)
        retrieval[name] = {
            "paired_cosine": float((query * canonical_fields).sum(dim=-1).mean()),
            "top1": float((top[:, 0] == expected).float().mean()),
            "top5": float((top == expected[:, None]).any(dim=1).float().mean()),
        }
        all_finite = all_finite and bool(
            torch.isfinite(query).all() and torch.isfinite(similarities).all()
        )
    trainable = sum(
        parameter.numel()
        for parameter in model.field.parameters()
        if parameter.requires_grad
    )
    metrics = {
        "examples": len(canonical),
        "canonical_binary_reconstruction_ink_f1": _micro_ink_f1(
            reconstructed,
            canonical,
        ),
        "canonical_encode_decode_reread_cosine": float(reread_cosine.mean()),
        "canonical_minimum_reread_cosine": float(reread_cosine.min()),
        "retrieval": retrieval,
        "all_finite": all_finite,
        "field_trainable_parameters": trainable,
    }
    gates = {
        "canonical_reconstruction_ink_f1": (
            metrics["canonical_binary_reconstruction_ink_f1"]
            - 0.995
            > V42_GATE_EPSILON
        ),
        "canonical_reread_cosine": (
            metrics["canonical_encode_decode_reread_cosine"]
            - 0.995
            > V42_GATE_EPSILON
        ),
        "held_sans_top1": (
            retrieval["noto_sans_cjk_bold"]["top1"] - 0.95
            > V42_GATE_EPSILON
        ),
        "held_serif_top1": (
            retrieval["noto_serif_cjk_medium"]["top1"] - 0.90
            > V42_GATE_EPSILON
        ),
        "all_finite": all_finite,
        "field_trainable_parameters_zero": trainable == 0,
    }
    return {**metrics, "gates": gates, "pass": all(gates.values())}


@torch.no_grad()
def evaluate_codec_spherical_language_v47(
    model: CodecSphericalGlyphLanguageModelV47,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float = 0.0,
) -> dict[str, float]:
    # V47 deliberately reuses the frozen V42 audit and controls. The model
    # exposes the same image-only contract, while its field is the V34 sphere.
    result = evaluate_canonical_language(
        model,  # type: ignore[arg-type]
        loader,
        statistics,
        bank_images,
        device=device,
        precision=precision,
        checkpoint_peak_vram_gib=checkpoint_peak_vram_gib,
    )
    finite = 0
    total = 0
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        with _autocast(device, precision):
            anchor = model.language(context)["anchor_fields"][:, -1]
        finite += int(torch.isfinite(anchor).all(dim=-1).sum())
        total += len(anchor)
    result["anchor_finite_rate"] = finite / max(1, total)
    return result


@torch.no_grad()
def evaluate_codec_spherical_counterfactual_pairs_v47(
    model: CodecSphericalGlyphLanguageModelV47,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    return evaluate_counterfactual_pairs(
        model,  # type: ignore[arg-type]
        loader,
        device=device,
        precision=precision,
    )


@torch.no_grad()
def evaluate_codec_spherical_generated_fields_v47(
    model: CodecSphericalGlyphLanguageModelV47,
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
        raise ValueError("V47 generated audit settings must be positive")
    model.eval()
    bank_fields = model.field.encode(bank_images.to(device))
    generator = torch.Generator(device=device).manual_seed(V47_AUDIT_SEED + 77)
    totals = {
        "correct": 0.0,
        "anchor_correct": 0.0,
        "pixel_f1": 0.0,
        "target_cosine": 0.0,
        "blank": 0.0,
        "ink_ratio": 0.0,
        "proposal_reread_cosine": 0.0,
        "selected_proposal_reread_cosine": 0.0,
        "proposal_finite": 0.0,
        "reread_finite": 0.0,
    }
    examples = 0
    proposals_seen = 0
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
        selected = trace["selected_reread_fields"].float()
        selected_proposal = trace["selected_proposal_fields"].float()
        proposals = trace["sample_fields"].float()
        rereads = trace["reread_fields"].float()
        anchor = trace["anchor_fields"].float()
        predicted_index = (selected @ bank_fields.transpose(0, 1)).argmax(dim=1)
        anchor_index = (anchor @ bank_fields.transpose(0, 1)).argmax(dim=1)
        target_binary = (target >= model.config.binary_threshold).to(pixels.dtype)
        predicted_ink = pixels >= 0.5
        target_ink = target_binary >= 0.5
        true_positive = (predicted_ink & target_ink).flatten(1).sum(dim=1).float()
        precision_value = true_positive / predicted_ink.flatten(1).sum(dim=1).clamp_min(1)
        recall_value = true_positive / target_ink.flatten(1).sum(dim=1).clamp_min(1)
        f1 = 2.0 * precision_value * recall_value / (
            precision_value + recall_value
        ).clamp_min(1e-8)
        target_fields = model.field.encode(target)
        predicted_density = predicted_ink.flatten(1).sum(dim=1).float()
        target_density = target_ink.flatten(1).sum(dim=1).float().clamp_min(1)
        proposal_reread = (proposals * rereads).sum(dim=-1)
        selected_reread = (selected_proposal * selected).sum(dim=-1)
        totals["correct"] += float((predicted_index == target_index).sum())
        totals["anchor_correct"] += float((anchor_index == target_index).sum())
        totals["pixel_f1"] += float(f1.sum())
        totals["target_cosine"] += float((selected * target_fields).sum())
        totals["blank"] += float((predicted_density < 2).sum())
        totals["ink_ratio"] += float((predicted_density / target_density).sum())
        totals["proposal_reread_cosine"] += float(proposal_reread.sum())
        totals["selected_proposal_reread_cosine"] += float(selected_reread.sum())
        totals["proposal_finite"] += float(
            torch.isfinite(proposals).all(dim=-1).sum()
        )
        totals["reread_finite"] += float(
            torch.isfinite(rereads).all(dim=-1).sum()
        )
        examples += len(context)
        proposals_seen += proposals.shape[0] * proposals.shape[1]
    if examples < 1:
        raise ValueError("V47 generated audit loader is empty")
    return {
        "examples": float(examples),
        "proposals": float(proposals_seen),
        "generated_identity_top1": totals["correct"] / examples,
        "anchor_identity_top1": totals["anchor_correct"] / examples,
        "generated_pixel_f1": totals["pixel_f1"] / examples,
        "generated_target_cosine": totals["target_cosine"] / examples,
        "generated_blank_rate": totals["blank"] / examples,
        "generated_ink_density_ratio": totals["ink_ratio"] / examples,
        "mean_proposal_to_visible_reread_cosine": (
            totals["proposal_reread_cosine"] / proposals_seen
        ),
        "mean_selected_proposal_to_visible_reread_cosine": (
            totals["selected_proposal_reread_cosine"] / examples
        ),
        "proposal_finite_rate": totals["proposal_finite"] / proposals_seen,
        "reread_finite_rate": totals["reread_finite"] / proposals_seen,
        "rereads_generated_pixels": 1.0,
    }


def codec_spherical_language_v47_boundary_is_clean(
    model: CodecSphericalGlyphLanguageModelV47,
) -> bool:
    receipt = codec_spherical_glyph_language_v47_boundary_receipt(model)
    required_true = {
        "input_is_continuous_image_stream",
        "output_is_continuous_image_field",
        "output_is_direct_raster",
        "field_is_fixed_continuous_codec_sphere",
        "causal_over_visual_time",
        "selection_uses_visible_reread",
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
        receipt["architecture"] == V47_ARCHITECTURE
        and not receipt["parameter_names_with_forbidden_fragments"]
        and receipt["field_trainable_parameters"] == 0
        and receipt["codec_checkpoint_sha256"]
        == V47_REQUIRED_CODEC_CHECKPOINT_SHA256
        and receipt["codec_state_sha256"] == V47_REQUIRED_CODEC_STATE_SHA256
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def codec_spherical_language_v47_gate_report(
    preflight: Mapping[str, Any],
    language: Mapping[str, float],
    pairs: Mapping[str, float],
    generated: Mapping[str, float],
    *,
    boundary_clean: bool,
    protocol_integrity_clean: bool,
    updates_complete: bool,
    pair_rows_consumed: int,
    total_parameters: int,
    trainable_parameters: int,
    total_elapsed_seconds: float,
) -> dict[str, bool]:
    def above(value: float, threshold: float) -> bool:
        return value - threshold > V42_GATE_EPSILON

    def below(value: float, threshold: float) -> bool:
        return threshold - value > V42_GATE_EPSILON

    return {
        "field_preflight": bool(preflight.get("pass", False)),
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
        "full_top1_beats_v46_by_0_01": above(
            language["full_top1"] - V47_V46_FULL_TOP1,
            0.01,
        ),
        "counterfactual_arm_accuracy": above(
            pairs["full_arm_accuracy"], 0.60
        ),
        "generated_identity_beats_unigram": above(
            generated["generated_identity_top1"], language["unigram_top1"]
        ),
        "generated_identity_beats_v46_by_0_01": above(
            generated["generated_identity_top1"]
            - V47_V46_GENERATED_IDENTITY_TOP1,
            0.01,
        ),
        "generated_pixel_f1": above(generated["generated_pixel_f1"], 0.55),
        "generated_blank_rate": below(generated["generated_blank_rate"], 0.02),
        "selected_proposal_visible_reread_cosine": above(
            generated["mean_selected_proposal_to_visible_reread_cosine"],
            0.90,
        ),
        "student_and_recurrent_boundary_clean": boundary_clean,
        "updates_and_unique_pairs_complete": (
            updates_complete and pair_rows_consumed == 80_000
        ),
        "parameter_budget": (
            total_parameters < V47_TOTAL_PARAMETER_LIMIT
            and trainable_parameters < V47_TRAINABLE_PARAMETER_LIMIT
        ),
        "protocol_runtime_and_memory": (
            protocol_integrity_clean
            and total_elapsed_seconds < 35.0 * 60.0
            and language["peak_allocated_vram_gib"] < 18.0
        ),
    }


def finite_metric_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_metric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_metric_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(float(value))))
    return True


__all__ = [
    "V47_AUDIT_SEED",
    "V47_TOTAL_PARAMETER_LIMIT",
    "V47_TRAINABLE_PARAMETER_LIMIT",
    "codec_spherical_field_preflight_v47",
    "codec_spherical_language_v47_boundary_is_clean",
    "codec_spherical_language_v47_gate_report",
    "evaluate_codec_spherical_counterfactual_pairs_v47",
    "evaluate_codec_spherical_generated_fields_v47",
    "evaluate_codec_spherical_language_v47",
    "finite_metric_tree",
]

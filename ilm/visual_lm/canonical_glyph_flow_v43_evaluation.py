from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable

import torch

from .canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    canonical_glyph_flow_v43_boundary_receipt,
)


V43_AUDIT_SEED = 20264340


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


@torch.no_grad()
def evaluate_v43_generated_fields(
    model: CanonicalGlyphFlowV43,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
) -> dict[str, float]:
    if maximum_examples < 1:
        raise ValueError("V43 generated audit requires positive examples")
    model.eval()
    bank_fields = model.language_model.field.encode_unit(bank_images.to(device))
    generator = torch.Generator(device=device).manual_seed(V43_AUDIT_SEED + 77)
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
            pixels, trace = model.sample_next(context, generator=generator)
        selected = trace["selected_fields"].float()
        anchor = trace["anchor_fields"].float()
        predicted_index = (selected @ bank_fields.transpose(0, 1)).argmax(dim=1)
        anchor_index = (anchor @ bank_fields.transpose(0, 1)).argmax(dim=1)
        target_binary = target >= 0.5
        predicted_ink = pixels >= 0.5
        true_positive = (predicted_ink & target_binary).flatten(1).sum(dim=1).float()
        precision_value = true_positive / predicted_ink.flatten(1).sum(dim=1).clamp_min(
            1
        )
        recall_value = true_positive / target_binary.flatten(1).sum(dim=1).clamp_min(1)
        f1 = (
            2.0
            * precision_value
            * recall_value
            / (precision_value + recall_value).clamp_min(1e-8)
        )
        target_fields = model.language_model.field.encode_unit(target)
        predicted_density = predicted_ink.flatten(1).sum(dim=1).float()
        target_density = target_binary.flatten(1).sum(dim=1).float().clamp_min(1)
        totals["correct"] += float((predicted_index == target_index).sum())
        totals["anchor_correct"] += float((anchor_index == target_index).sum())
        totals["pixel_f1"] += float(f1.sum())
        totals["target_cosine"] += float((selected * target_fields).sum())
        totals["blank"] += float((predicted_density < 2).sum())
        totals["ink_ratio"] += float((predicted_density / target_density).sum())
        examples += len(context)
    if examples < 1:
        raise ValueError("V43 generated audit loader is empty")
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


def canonical_glyph_flow_v43_boundary_is_clean(
    model: CanonicalGlyphFlowV43,
) -> bool:
    receipt = canonical_glyph_flow_v43_boundary_receipt(model)
    required_true = (
        "input_is_continuous_image_stream",
        "output_is_direct_raster",
        "conditional_spatial_flow",
        "candidate_selection_uses_reader_reread",
        "rereads_generated_pixels",
    )
    required_false = (
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
    )
    language = receipt["language"]
    return (
        receipt["architecture"] == "canonical-glyph-flow-v43"
        and not receipt["parameter_names_with_forbidden_fragments"]
        and not language["parameter_names_with_forbidden_fragments"]
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )

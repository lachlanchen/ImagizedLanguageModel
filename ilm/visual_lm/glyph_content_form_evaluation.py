from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from .glyph_content_form import GlyphContentFormModel
from .glyph_content_form_data import glyph_content_form_student_batch
from .glyph_era_invariance import cross_era_retrieval_metrics


@dataclass
class BinaryGlyphMetrics:
    pixel_accuracy: float
    ink_iou: float
    ink_f1: float


def binary_glyph_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> BinaryGlyphMetrics:
    if logits.shape != targets.shape or targets.ndim != 4:
        raise ValueError("V40 binary glyph targets do not align")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("V40 binary threshold must be in [0,1]")
    predicted_ink = logits.float().sigmoid() < threshold
    target_ink = targets.float() < 0.5
    intersection = (predicted_ink & target_ink).flatten(1).sum(dim=1).float()
    union = (predicted_ink | target_ink).flatten(1).sum(dim=1).float()
    predicted_count = predicted_ink.flatten(1).sum(dim=1).float()
    target_count = target_ink.flatten(1).sum(dim=1).float()
    return BinaryGlyphMetrics(
        pixel_accuracy=float(predicted_ink.eq(target_ink).float().mean()),
        ink_iou=float(((intersection + 1e-6) / (union + 1e-6)).mean()),
        ink_f1=float(
            ((2.0 * intersection + 1e-6) / (predicted_count + target_count + 1e-6)).mean()
        ),
    )


def evaluate_glyph_content_form(
    model: GlyphContentFormModel,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    maximum_batches: int = 0,
) -> dict[str, Any]:
    if maximum_batches < 0:
        raise ValueError("V40 evaluation batch limit cannot be negative")
    was_training = model.training
    model.eval()
    model.to(device)
    collected: dict[str, list[torch.Tensor]] = {
        "anchor_content": [],
        "positive_content": [],
        "anchor_form": [],
        "positive_form": [],
        "anchor_reference_form": [],
        "positive_reference_form": [],
        "self_logits": [],
        "reference_logits": [],
        "targets": [],
    }
    families: list[str] = []
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches):
                if maximum_batches and batch_index >= maximum_batches:
                    break
                student = {
                    key: value.to(device=device, dtype=torch.float32)
                    for key, value in glyph_content_form_student_batch(batch).items()
                }
                output = model(**student)
                collected["anchor_content"].append(output.anchor_content.float().cpu())
                collected["positive_content"].append(output.positive_content.float().cpu())
                collected["anchor_form"].append(output.anchor_form.float().cpu())
                collected["positive_form"].append(output.positive_form.float().cpu())
                collected["anchor_reference_form"].append(
                    output.anchor_reference_form.float().cpu()
                )
                collected["positive_reference_form"].append(
                    output.positive_reference_form.float().cpu()
                )
                collected["self_logits"].append(
                    model.decode_surface(
                        torch.cat(
                            (output.anchor_self_surface, output.positive_self_surface),
                            dim=0,
                        )
                    )
                    .float()
                    .cpu()
                )
                collected["reference_logits"].append(
                    model.decode_surface(
                        torch.cat(
                            (
                                output.anchor_reference_surface,
                                output.positive_reference_surface,
                            ),
                            dim=0,
                        )
                    )
                    .float()
                    .cpu()
                )
                collected["targets"].append(
                    torch.cat(
                        (student["anchor_pixels"], student["positive_pixels"]),
                        dim=0,
                    )
                    .float()
                    .cpu()
                )
                metadata = batch.get("metadata")
                if not isinstance(metadata, list):
                    raise TypeError("V40 evaluation requires host family metadata")
                families.extend(str(row["character"]) for row in metadata)
    finally:
        model.train(was_training)
    if not families:
        raise ValueError("V40 evaluation received no examples")
    if len(set(families)) != len(families):
        raise ValueError("V40 retrieval evaluation requires unique families")
    values = {key: torch.cat(rows) for key, rows in collected.items()}
    retrieval = cross_era_retrieval_metrics(
        values["anchor_content"],
        values["positive_content"],
    )
    content_cosine = F.cosine_similarity(
        values["anchor_content"],
        values["positive_content"],
        dim=-1,
    )
    same_stage_form = torch.cat(
        (
            F.cosine_similarity(
                values["anchor_form"],
                values["anchor_reference_form"],
                dim=-1,
            ),
            F.cosine_similarity(
                values["positive_form"],
                values["positive_reference_form"],
                dim=-1,
            ),
        )
    )
    cross_stage_form = F.cosine_similarity(
        values["anchor_form"],
        values["positive_form"],
        dim=-1,
    )
    self_visual = binary_glyph_metrics(values["self_logits"], values["targets"])
    reference_visual = binary_glyph_metrics(
        values["reference_logits"],
        values["targets"],
    )
    return {
        "families": len(families),
        "content_retrieval": retrieval,
        "content_cosine_mean": float(content_cosine.mean()),
        "content_cosine_median": float(content_cosine.median()),
        "same_stage_form_cosine_mean": float(same_stage_form.mean()),
        "cross_stage_same_family_form_cosine_mean": float(cross_stage_form.mean()),
        "form_stage_margin": float(same_stage_form.mean() - cross_stage_form.mean()),
        "self_visual": self_visual.__dict__,
        "reference_style_visual": reference_visual.__dict__,
        "boundary": {
            "model_inputs": ["glyph_pixels"],
            "family_labels_used_by_host_evaluator": True,
            "family_labels_received_by_model": False,
            "stage_labels_received_by_model": False,
            "runtime_claim": False,
        },
    }


__all__ = [
    "BinaryGlyphMetrics",
    "binary_glyph_metrics",
    "evaluate_glyph_content_form",
]

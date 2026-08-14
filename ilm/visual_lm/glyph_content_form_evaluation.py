from __future__ import annotations

from dataclasses import dataclass
import math
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
        "anchor_surface": [],
        "positive_surface": [],
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
                collected["anchor_surface"].append(output.anchor_surface.float().cpu())
                collected["positive_surface"].append(
                    output.positive_surface.float().cpu()
                )
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
    surface_retrieval = cross_era_retrieval_metrics(
        values["anchor_surface"],
        values["positive_surface"],
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
        "frozen_surface_retrieval": surface_retrieval,
        "content_retrieval": retrieval,
        "content_top1_gain_over_surface": float(
            retrieval["argmax_top1"] - surface_retrieval["argmax_top1"]
        ),
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


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _criterion(
    value: float,
    threshold: float,
    *,
    operator: str,
) -> dict[str, float | str | bool]:
    if operator == ">=":
        passed = value >= threshold
    elif operator == ">":
        passed = value > threshold
    else:
        raise ValueError("V40 gate operator is invalid")
    return {
        "value": float(value),
        "threshold": float(threshold),
        "operator": operator,
        "pass": bool(passed),
    }


def v40_pilot_gate(
    zero_trained: Mapping[str, Any],
    trained: Mapping[str, Any],
) -> dict[str, Any]:
    content_top1 = float(trained["content_retrieval"]["argmax_top1"])
    surface_top1 = float(trained["frozen_surface_retrieval"]["argmax_top1"])
    zero_content_top1 = float(zero_trained["content_retrieval"]["argmax_top1"])
    reference_f1 = float(trained["reference_style_visual"]["ink_f1"])
    zero_reference_f1 = float(zero_trained["reference_style_visual"]["ink_f1"])
    form_margin = float(trained["form_stage_margin"])
    zero_form_margin = float(zero_trained["form_stage_margin"])
    criteria = {
        "finite": {
            "value": bool(_all_finite(trained)),
            "threshold": True,
            "operator": "is",
            "pass": bool(_all_finite(trained)),
        },
        "content_at_least_surface": _criterion(
            content_top1 - surface_top1,
            0.0,
            operator=">=",
        ),
        "content_gain_over_zero": _criterion(
            content_top1 - zero_content_top1,
            0.03,
            operator=">=",
        ),
        "reference_ink_f1_gain": _criterion(
            reference_f1 - zero_reference_f1,
            0.15,
            operator=">=",
        ),
        "form_stage_margin_gain": _criterion(
            form_margin - zero_form_margin,
            0.05,
            operator=">=",
        ),
    }
    return {
        "gate": "v40-bounded-pilot",
        "qualified": all(bool(value["pass"]) for value in criteria.values()),
        "criteria": criteria,
    }


def v40_development_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    content = report["content_retrieval"]
    criteria = {
        "content_top1": _criterion(
            float(content["argmax_top1"]),
            0.60,
            operator=">=",
        ),
        "content_top1_gain_over_surface": _criterion(
            float(report["content_top1_gain_over_surface"]),
            0.10,
            operator=">=",
        ),
        "content_mrr": _criterion(float(content["mrr"]), 0.70, operator=">="),
        "reference_ink_f1": _criterion(
            float(report["reference_style_visual"]["ink_f1"]),
            0.70,
            operator=">=",
        ),
        "self_ink_f1": _criterion(
            float(report["self_visual"]["ink_f1"]),
            0.75,
            operator=">=",
        ),
        "form_stage_margin": _criterion(
            float(report["form_stage_margin"]),
            0.10,
            operator=">=",
        ),
    }
    return {
        "gate": "v40-development",
        "qualified": _all_finite(report)
        and all(bool(value["pass"]) for value in criteria.values()),
        "criteria": criteria,
    }


def v40_sealed_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    content = report["content_retrieval"]
    criteria = {
        "content_top1": _criterion(
            float(content["argmax_top1"]),
            0.55,
            operator=">=",
        ),
        "content_mrr": _criterion(float(content["mrr"]), 0.65, operator=">="),
        "reference_ink_f1": _criterion(
            float(report["reference_style_visual"]["ink_f1"]),
            0.65,
            operator=">=",
        ),
        "form_stage_margin": _criterion(
            float(report["form_stage_margin"]),
            0.0,
            operator=">",
        ),
    }
    return {
        "gate": "v40-sealed",
        "qualified": _all_finite(report)
        and all(bool(value["pass"]) for value in criteria.values()),
        "criteria": criteria,
    }


__all__ = [
    "BinaryGlyphMetrics",
    "binary_glyph_metrics",
    "evaluate_glyph_content_form",
    "v40_development_gate",
    "v40_pilot_gate",
    "v40_sealed_gate",
]

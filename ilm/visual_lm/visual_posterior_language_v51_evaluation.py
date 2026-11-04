from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from .canonical_glyph_language_evaluation import _shuffle_prefix
from .hierarchical_raster_language import CausalRasterReader
from .visual_posterior_language_v51 import VisualGlyphScorePrior


V51_AUDIT_SEED = 20265120


def pixel_f1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_ink = predicted >= 0.5
    target_ink = target >= 0.5
    true_positive = (predicted_ink & target_ink).flatten(1).sum(dim=-1).float()
    total = predicted_ink.flatten(1).sum(dim=-1) + target_ink.flatten(1).sum(
        dim=-1
    )
    return 2.0 * true_positive / total.clamp_min(1)


def isolated_ink_fraction(images: torch.Tensor) -> torch.Tensor:
    ink = (images >= 0.5).float()
    neighbors = F.conv2d(
        ink,
        torch.ones(1, 1, 3, 3, device=ink.device),
        padding=1,
    ) - ink
    isolated = (ink > 0.5) & (neighbors <= 1.0)
    return isolated.flatten(1).sum(dim=-1).float() / ink.flatten(1).sum(
        dim=-1
    ).clamp_min(1.0)


def image_identity(
    reader: CausalRasterReader,
    images: torch.Tensor,
    bank_fields: torch.Tensor,
) -> torch.Tensor:
    fields = reader.field.encode_unit(images)
    return (fields.float() @ bank_fields.float().transpose(0, 1)).argmax(dim=-1)


@torch.no_grad()
def evaluate_posterior_generation(
    reader: CausalRasterReader,
    prior: VisualGlyphScorePrior,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    maximum_examples: int,
) -> dict[str, dict[str, float]]:
    if maximum_examples < 1:
        raise ValueError("V51 generation audit requires examples")
    reader.eval()
    prior.eval()
    bank_fields = reader.field.encode_unit(bank_images.to(device)).float()
    routes = ("ordered", "shuffled")
    totals = {
        route: {
            "examples": 0.0,
            "plan_correct": 0.0,
            "posterior_correct": 0.0,
            "plan_f1": 0.0,
            "posterior_f1": 0.0,
            "target_ink": 0.0,
            "plan_ink": 0.0,
            "posterior_ink": 0.0,
            "target_isolated": 0.0,
            "plan_isolated": 0.0,
            "posterior_isolated": 0.0,
            "blank": 0.0,
            "posterior_anchor_cosine": 0.0,
            "posterior_plan_identity_agreement": 0.0,
        }
        for route in routes
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        target_index = raw["target_index"][:remaining].to(
            device, non_blocking=True
        )
        contexts = {
            "ordered": context,
            "shuffled": _shuffle_prefix(context, first_index=examples),
        }
        for route, visible in contexts.items():
            language = reader.language(visible)
            anchor = language["anchor_fields"][:, -1].float()
            plan = reader.field.binary(anchor)
            generator = torch.Generator(device=device).manual_seed(
                V51_AUDIT_SEED + examples * 101
            )
            with torch.enable_grad():
                posterior, metadata = prior.posterior_sample(
                    plan,
                    anchor,
                    generator=generator,
                )
            plan_identity = image_identity(reader, plan, bank_fields)
            posterior_identity = image_identity(reader, posterior, bank_fields)
            count = len(context)
            item = totals[route]
            item["examples"] += count
            item["plan_correct"] += float((plan_identity == target_index).sum())
            item["posterior_correct"] += float(
                (posterior_identity == target_index).sum()
            )
            item["plan_f1"] += float(pixel_f1(plan, target).sum())
            item["posterior_f1"] += float(pixel_f1(posterior, target).sum())
            item["target_ink"] += float(target.flatten(1).mean(dim=-1).sum())
            item["plan_ink"] += float(plan.flatten(1).mean(dim=-1).sum())
            item["posterior_ink"] += float(
                posterior.flatten(1).mean(dim=-1).sum()
            )
            item["target_isolated"] += float(
                isolated_ink_fraction(target).sum()
            )
            item["plan_isolated"] += float(isolated_ink_fraction(plan).sum())
            item["posterior_isolated"] += float(
                isolated_ink_fraction(posterior).sum()
            )
            item["blank"] += float(
                (posterior.flatten(1).sum(dim=-1) < 2).sum()
            )
            item["posterior_anchor_cosine"] += float(
                (metadata["visual_fields"].float() * anchor).sum(dim=-1).sum()
            )
            item["posterior_plan_identity_agreement"] += float(
                (posterior_identity == plan_identity).sum()
            )
        examples += len(context)
    if examples < 1:
        raise ValueError("V51 generation loader is empty")
    result: dict[str, dict[str, float]] = {}
    for route, item in totals.items():
        count = item["examples"]
        plan_identity = item["plan_correct"] / count
        posterior_identity = item["posterior_correct"] / count
        plan_f1 = item["plan_f1"] / count
        posterior_f1 = item["posterior_f1"] / count
        target_isolated = item["target_isolated"] / count
        posterior_isolated = item["posterior_isolated"] / count
        result[route] = {
            "examples": count,
            "plan_identity_top1": plan_identity,
            "posterior_identity_top1": posterior_identity,
            "posterior_identity_retention_of_plan": (
                posterior_identity / plan_identity if plan_identity > 0.0 else 0.0
            ),
            "posterior_plan_identity_agreement": item[
                "posterior_plan_identity_agreement"
            ]
            / count,
            "plan_pixel_f1": plan_f1,
            "posterior_pixel_f1": posterior_f1,
            "posterior_pixel_f1_delta_from_plan": posterior_f1 - plan_f1,
            "target_ink_fraction": item["target_ink"] / count,
            "plan_ink_fraction": item["plan_ink"] / count,
            "posterior_ink_fraction": item["posterior_ink"] / count,
            "target_isolated_ink_fraction": target_isolated,
            "plan_isolated_ink_fraction": item["plan_isolated"] / count,
            "posterior_isolated_ink_fraction": posterior_isolated,
            "posterior_isolated_ratio_to_target": (
                posterior_isolated / target_isolated
                if target_isolated > 0.0
                else 0.0
            ),
            "posterior_blank_rate": item["blank"] / count,
            "posterior_anchor_cosine": item["posterior_anchor_cosine"] / count,
        }
    return result


@torch.no_grad()
def evaluate_posterior_feedback(
    reader: CausalRasterReader,
    prior: VisualGlyphScorePrior,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    maximum_examples: int,
    steps: int = 4,
) -> dict[str, Any]:
    if min(maximum_examples, steps) < 1:
        raise ValueError("V51 feedback settings are invalid")
    reader.eval()
    prior.eval()
    bank_fields = reader.field.encode_unit(bank_images.to(device)).float()
    totals = {
        route: {
            "examples": 0.0,
            "correct": torch.zeros(steps),
            "f1": torch.zeros(steps),
            "blank": torch.zeros(steps),
        }
        for route in ("ordered", "shuffled")
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        source = raw["context"][:remaining].to(device, non_blocking=True)
        continuation = raw["continuation"][:remaining, :steps].to(
            device, non_blocking=True
        )
        initial = {
            "ordered": source,
            "shuffled": _shuffle_prefix(source, first_index=examples),
        }
        target_identity = image_identity(
            reader,
            continuation.flatten(0, 1),
            bank_fields,
        ).reshape(len(source), steps)
        for route, sequence in initial.items():
            for step in range(steps):
                language = reader.language(sequence)
                anchor = language["anchor_fields"][:, -1].float()
                plan = reader.field.binary(anchor)
                generator = torch.Generator(device=device).manual_seed(
                    V51_AUDIT_SEED + examples * 101 + step * 10_007
                )
                with torch.enable_grad():
                    generated, _ = prior.posterior_sample(
                        plan,
                        anchor,
                        generator=generator,
                    )
                identity = image_identity(reader, generated, bank_fields)
                totals[route]["correct"][step] += float(
                    (identity == target_identity[:, step]).sum()
                )
                totals[route]["f1"][step] += float(
                    pixel_f1(generated, continuation[:, step]).sum()
                )
                totals[route]["blank"][step] += float(
                    (generated.flatten(1).sum(dim=-1) < 2).sum()
                )
                sequence = torch.cat((sequence, generated[:, None]), dim=1)[
                    :, -reader.config.maximum_cells :
                ]
            totals[route]["examples"] += len(source)
        examples += len(source)
    if examples < 1:
        raise ValueError("V51 feedback loader is empty")
    result: dict[str, Any] = {}
    for route, item in totals.items():
        count = item["examples"]
        identity = (item["correct"] / count).tolist()
        f1 = (item["f1"] / count).tolist()
        blank = (item["blank"] / count).tolist()
        result[route] = {
            "examples": count,
            "identity_top1_by_step": identity,
            "pixel_f1_by_step": f1,
            "blank_rate_by_step": blank,
            "mean_identity_top1": sum(identity) / len(identity),
            "mean_pixel_f1": sum(f1) / len(f1),
        }
    return result

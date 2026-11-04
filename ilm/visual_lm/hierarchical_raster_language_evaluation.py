from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import torch

from .canonical_glyph_language_evaluation import _shuffle_prefix
from .hierarchical_raster_language import (
    HierarchicalRasterLanguageModel,
    raster_to_generation_units,
)
from .hierarchical_raster_language_training import raster_mixture_nll


V50_AUDIT_SEED = 20265020


def _autocast(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _pixel_f1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_ink = predicted >= 0.5
    target_ink = target >= 0.5
    true_positive = (predicted_ink & target_ink).flatten(1).sum(dim=-1).float()
    total = predicted_ink.flatten(1).sum(dim=-1) + target_ink.flatten(1).sum(
        dim=-1
    )
    return 2.0 * true_positive / total.clamp_min(1)


@torch.no_grad()
def evaluate_raster_likelihood(
    model: HierarchicalRasterLanguageModel,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
) -> dict[str, float]:
    if maximum_examples < 1:
        raise ValueError("V50 likelihood audit requires examples")
    model.eval()
    totals = {
        "examples": 0.0,
        "ordered_coarse": 0.0,
        "ordered_detail": 0.0,
        "shuffled_coarse": 0.0,
        "shuffled_detail": 0.0,
        "unconditioned_coarse": 0.0,
        "unconditioned_detail": 0.0,
        "ordered_better_coarse": 0.0,
        "ordered_better_full": 0.0,
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        shuffled_context = _shuffle_prefix(context, first_index=examples)
        units = raster_to_generation_units(target, config=model.config)
        with _autocast(device, precision):
            ordered_language = model.language(context)
            shuffled_language = model.language(shuffled_context)
            ordered = model.writer(
                ordered_language["hidden_states"][:, -1],
                ordered_language["anchor_fields"][:, -1],
                units,
                model.plan_units(ordered_language["anchor_fields"][:, -1]),
            )
            shuffled = model.writer(
                shuffled_language["hidden_states"][:, -1],
                shuffled_language["anchor_fields"][:, -1],
                units,
                model.plan_units(shuffled_language["anchor_fields"][:, -1]),
            )
            unconditioned = model.writer(
                torch.zeros_like(ordered_language["hidden_states"][:, -1]),
                torch.zeros_like(ordered_language["anchor_fields"][:, -1]),
                units,
                torch.zeros_like(units),
            )
        ordered_patch, _ = raster_mixture_nll(ordered, units)
        shuffled_patch, _ = raster_mixture_nll(shuffled, units)
        unconditioned_patch, _ = raster_mixture_nll(unconditioned, units)
        coarse_units = model.config.coarse_units
        ordered_full = 0.5 * (
            ordered_patch[:, :coarse_units].mean(dim=-1)
            + ordered_patch[:, coarse_units:].mean(dim=-1)
        )
        shuffled_full = 0.5 * (
            shuffled_patch[:, :coarse_units].mean(dim=-1)
            + shuffled_patch[:, coarse_units:].mean(dim=-1)
        )
        count = len(context)
        totals["examples"] += count
        totals["ordered_coarse"] += float(
            ordered_patch[:, :coarse_units].mean(dim=-1).sum()
        )
        totals["ordered_detail"] += float(
            ordered_patch[:, coarse_units:].mean(dim=-1).sum()
        )
        totals["shuffled_coarse"] += float(
            shuffled_patch[:, :coarse_units].mean(dim=-1).sum()
        )
        totals["shuffled_detail"] += float(
            shuffled_patch[:, coarse_units:].mean(dim=-1).sum()
        )
        totals["unconditioned_coarse"] += float(
            unconditioned_patch[:, :coarse_units].mean(dim=-1).sum()
        )
        totals["unconditioned_detail"] += float(
            unconditioned_patch[:, coarse_units:].mean(dim=-1).sum()
        )
        totals["ordered_better_coarse"] += float(
            (
                ordered_patch[:, :coarse_units].mean(dim=-1)
                < shuffled_patch[:, :coarse_units].mean(dim=-1)
            ).sum()
        )
        totals["ordered_better_full"] += float(
            (ordered_full < shuffled_full).sum()
        )
        examples += count
    if examples < 1:
        raise ValueError("V50 likelihood loader is empty")
    count = totals["examples"]
    result = {key: value / count for key, value in totals.items() if key != "examples"}
    result["examples"] = count
    result["ordered_coarse_nll_gain_over_shuffled"] = (
        result["shuffled_coarse"] - result["ordered_coarse"]
    )
    result["ordered_full_nll_gain_over_shuffled"] = 0.5 * (
        result["shuffled_coarse"]
        + result["shuffled_detail"]
        - result["ordered_coarse"]
        - result["ordered_detail"]
    )
    return result


def _identity(
    model: HierarchicalRasterLanguageModel,
    cells: torch.Tensor,
    bank_fields: torch.Tensor,
) -> torch.Tensor:
    fields = model.reader.field.encode_unit(cells)
    return (fields.float() @ bank_fields.float().transpose(0, 1)).argmax(dim=-1)


@torch.no_grad()
def evaluate_raster_generation(
    model: HierarchicalRasterLanguageModel,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
    samples: int,
    temperature: float = 1.0,
) -> dict[str, dict[str, float]]:
    if min(maximum_examples, samples) < 1 or temperature <= 0.0:
        raise ValueError("V50 generation settings are invalid")
    model.eval()
    bank_fields = model.reader.field.encode_unit(bank_images.to(device)).float()
    routes = ("ordered", "shuffled", "unconditioned")
    totals = {
        route: {
            "examples": 0.0,
            "greedy_correct": 0.0,
            "sample_correct": 0.0,
            "selected_correct": 0.0,
            "coverage": 0.0,
            "greedy_f1": 0.0,
            "sample_f1": 0.0,
            "selected_f1": 0.0,
            "greedy_coarse_f1": 0.0,
            "blank": 0.0,
            "unique": 0.0,
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
        target_index = raw["target_index"][:remaining].to(device, non_blocking=True)
        shuffled_context = _shuffle_prefix(context, first_index=examples)
        with _autocast(device, precision):
            ordered_language = model.language(context)
            shuffled_language = model.language(shuffled_context)
        route_conditions = {
            "ordered": (
                ordered_language["hidden_states"][:, -1],
                ordered_language["anchor_fields"][:, -1],
            ),
            "shuffled": (
                shuffled_language["hidden_states"][:, -1],
                shuffled_language["anchor_fields"][:, -1],
            ),
            "unconditioned": (
                torch.zeros_like(ordered_language["hidden_states"][:, -1]),
                torch.zeros_like(ordered_language["anchor_fields"][:, -1]),
            ),
        }
        target_coarse = raster_to_generation_units(
            target, config=model.config
        )[:, : model.config.coarse_units]
        rows = torch.arange(len(context), device=device)
        for route, (hidden, anchor) in route_conditions.items():
            plan = (
                model.plan_units(anchor)
                if route != "unconditioned"
                else torch.zeros(
                    len(anchor),
                    model.config.generation_units,
                    model.config.pixels_per_patch,
                    device=anchor.device,
                    dtype=anchor.dtype,
                )
            )
            with _autocast(device, precision):
                greedy, _ = model.writer.generate(
                    hidden,
                    anchor,
                    plan,
                    stochastic=False,
                    temperature=temperature,
                )
            repeated_hidden = hidden[:, None].expand(-1, samples, -1).flatten(0, 1)
            repeated_anchor = anchor[:, None].expand(-1, samples, -1).flatten(0, 1)
            repeated_plan = plan[:, None].expand(-1, samples, -1, -1).flatten(0, 1)
            generator = torch.Generator(device=device).manual_seed(
                V50_AUDIT_SEED + examples * 17
            )
            with _autocast(device, precision):
                sampled, log_probability = model.writer.generate(
                    repeated_hidden,
                    repeated_anchor,
                    repeated_plan,
                    stochastic=True,
                    generator=generator,
                    temperature=temperature,
                )
            sampled = sampled.reshape(len(context), samples, 1, 32, 32)
            log_probability = log_probability.reshape(len(context), samples)
            identities = _identity(
                model,
                sampled.flatten(0, 1),
                bank_fields,
            ).reshape(len(context), samples)
            greedy_identity = _identity(model, greedy, bank_fields)
            selection = log_probability.argmax(dim=-1)
            selected = sampled[rows, selection]
            selected_identity = identities[rows, selection]
            greedy_coarse = raster_to_generation_units(
                greedy, config=model.config
            )[:, : model.config.coarse_units]
            item = totals[route]
            item["examples"] += len(context)
            item["greedy_correct"] += float(
                (greedy_identity == target_index).sum()
            )
            item["sample_correct"] += float(
                (identities[:, 0] == target_index).sum()
            )
            item["selected_correct"] += float(
                (selected_identity == target_index).sum()
            )
            item["coverage"] += float(
                (identities == target_index[:, None]).any(dim=-1).sum()
            )
            item["greedy_f1"] += float(_pixel_f1(greedy, target).sum())
            item["sample_f1"] += float(_pixel_f1(sampled[:, 0], target).sum())
            item["selected_f1"] += float(_pixel_f1(selected, target).sum())
            item["greedy_coarse_f1"] += float(
                _pixel_f1(greedy_coarse, target_coarse).sum()
            )
            item["blank"] += float(
                (selected.flatten(1).sum(dim=-1) < 2).sum()
            )
            item["unique"] += float(
                sum(len(torch.unique(row)) for row in identities)
            )
        examples += len(context)
    if examples < 1:
        raise ValueError("V50 generation loader is empty")
    result: dict[str, dict[str, float]] = {}
    for route, item in totals.items():
        count = item["examples"]
        result[route] = {
            "examples": count,
            "greedy_identity_top1": item["greedy_correct"] / count,
            "first_sample_identity_top1": item["sample_correct"] / count,
            "self_selected_identity_top1": item["selected_correct"] / count,
            "target_coverage": item["coverage"] / count,
            "greedy_pixel_f1": item["greedy_f1"] / count,
            "first_sample_pixel_f1": item["sample_f1"] / count,
            "self_selected_pixel_f1": item["selected_f1"] / count,
            "greedy_coarse_f1": item["greedy_coarse_f1"] / count,
            "self_selected_blank_rate": item["blank"] / count,
            "mean_unique_identities": item["unique"] / count,
        }
    return result


@torch.no_grad()
def evaluate_visible_feedback(
    model: HierarchicalRasterLanguageModel,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
    steps: int = 4,
) -> dict[str, Any]:
    if min(maximum_examples, steps) < 1:
        raise ValueError("V50 feedback settings are invalid")
    model.eval()
    bank_fields = model.reader.field.encode_unit(bank_images.to(device)).float()
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
        target_identity = _identity(
            model,
            continuation.flatten(0, 1),
            bank_fields,
        ).reshape(len(source), steps)
        for route, sequence in initial.items():
            for step in range(steps):
                with _autocast(device, precision):
                    generated, _ = model.sample_next(sequence, stochastic=False)
                identity = _identity(model, generated, bank_fields)
                totals[route]["correct"][step] += float(
                    (identity == target_identity[:, step]).sum()
                )
                totals[route]["f1"][step] += float(
                    _pixel_f1(generated, continuation[:, step]).sum()
                )
                totals[route]["blank"][step] += float(
                    (generated.flatten(1).sum(dim=-1) < 2).sum()
                )
                sequence = torch.cat((sequence, generated[:, None]), dim=1)[
                    :, -model.reader_config.maximum_cells :
                ]
            totals[route]["examples"] += len(source)
        examples += len(source)
    if examples < 1:
        raise ValueError("V50 feedback loader is empty")
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

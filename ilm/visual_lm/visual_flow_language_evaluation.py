from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import torch

from .canonical_glyph_language_evaluation import _shuffle_prefix
from .ink_writer import sample_foveal_ink
from .visual_flow_language import VisualFlowLanguageModel


VISUAL_FLOW_LANGUAGE_AUDIT_SEED = 20264920


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
    true_positive = (predicted_ink & target_ink).flatten(-3).sum(dim=-1).float()
    total = predicted_ink.flatten(-3).sum(dim=-1) + target_ink.flatten(-3).sum(
        dim=-1
    )
    return 2.0 * true_positive / total.clamp_min(1)


@torch.no_grad()
def evaluate_visual_flow_generation(
    model: VisualFlowLanguageModel,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
    samples: int = 4,
    steps: int = 16,
    guidance_scale: float = 1.25,
) -> dict[str, dict[str, float]]:
    if min(maximum_examples, samples, steps) < 1 or guidance_scale < 0.0:
        raise ValueError("visual-flow generation settings are invalid")
    model.eval()
    bank_fields = model.language_model.field.encode_unit(bank_images.to(device)).float()
    routes = ("ordered", "shuffled", "unconditioned")
    totals = {
        route: {
            "examples": 0.0,
            "first_correct": 0.0,
            "selected_correct": 0.0,
            "coverage": 0.0,
            "first_f1": 0.0,
            "selected_f1": 0.0,
            "blank": 0.0,
            "unique": 0.0,
        }
        for route in routes
    }
    generator = torch.Generator(device=device).manual_seed(
        VISUAL_FLOW_LANGUAGE_AUDIT_SEED + 77
    )
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        target_index = raw["target_index"][:remaining].to(device, non_blocking=True)
        shuffled = _shuffle_prefix(context, first_index=examples)
        with _autocast(device, precision):
            ordered_output = model.language_model(context)
            shuffled_output = model.language_model(shuffled)
        initial_noise = torch.randn(
            (len(context) * samples, 1, 32, 32),
            device=device,
            dtype=ordered_output["hidden_states"].dtype,
            generator=generator,
        )
        route_inputs = {
            "ordered": (
                ordered_output["hidden_states"][:, -1],
                ordered_output["anchor_fields"][:, -1],
                guidance_scale,
            ),
            "shuffled": (
                shuffled_output["hidden_states"][:, -1],
                shuffled_output["anchor_fields"][:, -1],
                guidance_scale,
            ),
            "unconditioned": (
                ordered_output["hidden_states"][:, -1],
                ordered_output["anchor_fields"][:, -1],
                0.0,
            ),
        }
        for route, (hidden, anchor, guidance) in route_inputs.items():
            repeated_hidden = hidden[:, None].expand(-1, samples, -1).flatten(0, 1)
            zero_plan = torch.zeros(
                (len(repeated_hidden), 1, 32, 32),
                device=device,
                dtype=hidden.dtype,
            )
            with _autocast(device, precision):
                signed = sample_foveal_ink(
                    model.writer,
                    repeated_hidden,
                    zero_plan,
                    steps=steps,
                    guidance_scale=guidance,
                    initial_noise=initial_noise,
                )
            candidates = (
                (signed.add(1.0).div(2.0) >= 0.5)
                .float()
                .reshape(len(context), samples, 1, 32, 32)
            )
            fields = model.language_model.field.encode_unit(
                candidates.flatten(0, 1)
            ).reshape(len(context), samples, -1)
            identities = (fields.float() @ bank_fields.transpose(0, 1)).argmax(-1)
            selection = torch.einsum(
                "bsd,bd->bs", fields.float(), anchor.float()
            ).argmax(1)
            rows = torch.arange(len(context), device=device)
            first = candidates[:, 0]
            selected = candidates[rows, selection]
            first_identity = identities[:, 0]
            selected_identity = identities[rows, selection]
            item = totals[route]
            item["examples"] += len(context)
            item["first_correct"] += float((first_identity == target_index).sum())
            item["selected_correct"] += float(
                (selected_identity == target_index).sum()
            )
            item["coverage"] += float(
                (identities == target_index[:, None]).any(1).sum()
            )
            item["first_f1"] += float(_pixel_f1(first, target).sum())
            item["selected_f1"] += float(_pixel_f1(selected, target).sum())
            item["blank"] += float(
                ((selected >= 0.5).flatten(1).sum(1) < 2).sum()
            )
            item["unique"] += float(
                sum(len(torch.unique(row)) for row in identities)
            )
        examples += len(context)
    if examples < 1:
        raise ValueError("visual-flow generation loader is empty")
    result: dict[str, dict[str, float]] = {}
    for route, item in totals.items():
        count = item["examples"]
        result[route] = {
            "examples": count,
            "first_identity_top1": item["first_correct"] / count,
            "selected_identity_top1": item["selected_correct"] / count,
            "target_coverage": item["coverage"] / count,
            "first_pixel_f1": item["first_f1"] / count,
            "selected_pixel_f1": item["selected_f1"] / count,
            "selected_blank_rate": item["blank"] / count,
            "mean_unique_identities": item["unique"] / count,
        }
    return result

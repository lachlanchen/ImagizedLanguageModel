from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .canonical_glyph_language_evaluation import _shuffle_prefix
from .hierarchical_raster_language import CausalRasterReader
from .noisy_visual_energy_guidance_v52 import (
    GuidedVisualLanguageSampler,
    NoisyVisualEnergyCritic,
)
from .visual_posterior_language_v51_evaluation import (
    image_identity,
    isolated_ink_fraction,
    pixel_f1,
)


V52_AUDIT_SEED = 20265220


@torch.no_grad()
def evaluate_clean_visual_energy(
    reader: CausalRasterReader,
    critic: NoisyVisualEnergyCritic,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    maximum_examples: int,
) -> dict[str, dict[str, float]]:
    if maximum_examples < 1:
        raise ValueError("V52 clean-energy audit requires examples")
    reader.eval()
    critic.eval()
    bank = bank_images.to(device)
    bank_fields = reader.field.encode_unit(bank).float()
    bank_signed = bank.float().mul(2.0).sub(1.0)
    bank_times = torch.zeros(len(bank), device=device, dtype=torch.long)
    bank_keys = critic.image_keys(bank_signed, bank_times).float()
    totals = {
        route: {
            "examples": 0.0,
            "correct": 0.0,
            "log_probability": 0.0,
            "v42_correct": 0.0,
        }
        for route in ("ordered", "shuffled")
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        targets = raw["target_index"][:remaining].to(device, non_blocking=True)
        contexts = {
            "ordered": context,
            "shuffled": _shuffle_prefix(context, first_index=examples),
        }
        for route, visible in contexts.items():
            language = reader.language(visible)
            hidden = language["hidden_states"][:, -1].float()
            anchors = language["anchor_fields"][:, -1].float()
            times = torch.zeros(len(visible), device=device, dtype=torch.long)
            query = critic.context_queries(hidden, anchors, times).float()
            logits = critic.contrastive_scale.float() * query @ bank_keys.transpose(
                0, 1
            )
            selected = logits.argmax(dim=-1)
            v42_selected = (anchors @ bank_fields.transpose(0, 1)).argmax(dim=-1)
            log_probability = logits.log_softmax(dim=-1).gather(
                1, targets[:, None]
            )[:, 0]
            item = totals[route]
            item["examples"] += len(visible)
            item["correct"] += float((selected == targets).sum())
            item["v42_correct"] += float((v42_selected == targets).sum())
            item["log_probability"] += float(log_probability.sum())
        examples += len(context)
    if examples < 1:
        raise ValueError("V52 clean-energy loader is empty")
    return {
        route: {
            "examples": item["examples"],
            "identity_top1": item["correct"] / item["examples"],
            "target_log_probability": item["log_probability"]
            / item["examples"],
            "v42_identity_top1": item["v42_correct"] / item["examples"],
        }
        for route, item in totals.items()
    }


def _empty_generation_totals() -> dict[str, Any]:
    return {
        "examples": 0.0,
        "first_correct": 0.0,
        "selected_correct": 0.0,
        "first_f1": 0.0,
        "selected_f1": 0.0,
        "target_ink": 0.0,
        "first_ink": 0.0,
        "selected_ink": 0.0,
        "first_isolated": 0.0,
        "selected_isolated": 0.0,
        "first_blank": 0.0,
        "selected_blank": 0.0,
        "gradient_rms": 0.0,
        "selected_identities": set(),
    }


@torch.no_grad()
def evaluate_guided_generation(
    sampler: GuidedVisualLanguageSampler,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    maximum_examples: int,
) -> dict[str, dict[str, float]]:
    if maximum_examples < 1:
        raise ValueError("V52 generation audit requires examples")
    sampler.eval()
    bank_fields = sampler.reader.field.encode_unit(bank_images.to(device)).float()
    totals = {
        route: _empty_generation_totals()
        for route in ("ordered", "shuffled", "zero_guidance")
    }
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(
            device, non_blocking=True
        )
        target_index = raw["target_index"][:remaining].to(
            device, non_blocking=True
        )
        shuffled = _shuffle_prefix(context, first_index=examples)
        route_contexts = {
            "ordered": context,
            "shuffled": shuffled,
            "zero_guidance": context,
        }
        for route, visible in route_contexts.items():
            language = sampler.reader.language(visible)
            hidden = language["hidden_states"][:, -1].float()
            anchors = language["anchor_fields"][:, -1].float()
            generator = torch.Generator(device=device).manual_seed(
                V52_AUDIT_SEED + examples * 104_729
            )
            strength = 0.0 if route == "zero_guidance" else None
            with torch.enable_grad():
                selected, trace = sampler.sample_from_state(
                    hidden,
                    anchors,
                    generator=generator,
                    guidance_strength=strength,
                )
            first = trace["first_samples"]
            first_identity = image_identity(
                sampler.reader, first, bank_fields
            )
            selected_identity = image_identity(
                sampler.reader, selected, bank_fields
            )
            item = totals[route]
            item["examples"] += len(context)
            item["first_correct"] += float(
                (first_identity == target_index).sum()
            )
            item["selected_correct"] += float(
                (selected_identity == target_index).sum()
            )
            item["first_f1"] += float(pixel_f1(first, target).sum())
            item["selected_f1"] += float(pixel_f1(selected, target).sum())
            item["target_ink"] += float(target.flatten(1).mean(dim=-1).sum())
            item["first_ink"] += float(first.flatten(1).mean(dim=-1).sum())
            item["selected_ink"] += float(
                selected.flatten(1).mean(dim=-1).sum()
            )
            item["first_isolated"] += float(
                isolated_ink_fraction(first).sum()
            )
            item["selected_isolated"] += float(
                isolated_ink_fraction(selected).sum()
            )
            item["first_blank"] += float(
                (first.flatten(1).sum(dim=-1) < 2).sum()
            )
            item["selected_blank"] += float(
                (selected.flatten(1).sum(dim=-1) < 2).sum()
            )
            item["gradient_rms"] += float(
                trace["gradient_rms"].mean(dim=(1, 2)).sum()
            )
            item["selected_identities"].update(
                int(index) for index in selected_identity.detach().cpu().tolist()
            )
        examples += len(context)
    if examples < 1:
        raise ValueError("V52 generation loader is empty")
    result: dict[str, dict[str, float]] = {}
    for route, item in totals.items():
        count = item["examples"]
        target_ink = item["target_ink"] / count
        selected_ink = item["selected_ink"] / count
        result[route] = {
            "examples": count,
            "first_identity_top1": item["first_correct"] / count,
            "selected_identity_top1": item["selected_correct"] / count,
            "first_pixel_f1": item["first_f1"] / count,
            "selected_pixel_f1": item["selected_f1"] / count,
            "target_ink_fraction": target_ink,
            "first_ink_fraction": item["first_ink"] / count,
            "selected_ink_fraction": selected_ink,
            "selected_to_target_ink_ratio": (
                selected_ink / target_ink if target_ink > 0.0 else 0.0
            ),
            "first_isolated_ink_fraction": item["first_isolated"] / count,
            "selected_isolated_ink_fraction": item["selected_isolated"]
            / count,
            "first_blank_rate": item["first_blank"] / count,
            "selected_blank_rate": item["selected_blank"] / count,
            "mean_guidance_gradient_rms": item["gradient_rms"] / count,
            "distinct_selected_identities": float(
                len(item["selected_identities"])
            ),
        }
    return result


@torch.no_grad()
def evaluate_guided_feedback(
    sampler: GuidedVisualLanguageSampler,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    maximum_examples: int,
    steps: int = 4,
) -> dict[str, dict[str, Any]]:
    if min(maximum_examples, steps) < 1:
        raise ValueError("V52 feedback settings are invalid")
    sampler.eval()
    bank_fields = sampler.reader.field.encode_unit(bank_images.to(device)).float()
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
            sampler.reader,
            continuation.flatten(0, 1),
            bank_fields,
        ).reshape(len(source), steps)
        for route, sequence in initial.items():
            for step in range(steps):
                generator = torch.Generator(device=device).manual_seed(
                    V52_AUDIT_SEED + examples * 104_729 + step * 10_007
                )
                with torch.enable_grad():
                    generated, _ = sampler.sample_next(
                        sequence[:, -sampler.reader.config.maximum_cells :],
                        generator=generator,
                    )
                identity = image_identity(
                    sampler.reader, generated, bank_fields
                )
                totals[route]["correct"][step] += float(
                    (identity == target_identity[:, step]).sum()
                )
                totals[route]["f1"][step] += float(
                    pixel_f1(generated, continuation[:, step]).sum()
                )
                totals[route]["blank"][step] += float(
                    (generated.flatten(1).sum(dim=-1) < 2).sum()
                )
                sequence = torch.cat((sequence, generated[:, None]), dim=1)
            totals[route]["examples"] += len(source)
        examples += len(source)
    if examples < 1:
        raise ValueError("V52 feedback loader is empty")
    result: dict[str, dict[str, Any]] = {}
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


__all__ = [
    "V52_AUDIT_SEED",
    "evaluate_clean_visual_energy",
    "evaluate_guided_feedback",
    "evaluate_guided_generation",
]

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .noise_limited_retinal_field_v45 import NoiseLimitedRetinalFieldV45


V45_AUDIT_SEED = 20264500
V45_GATE_EPSILON = 1e-12


def _normalized_weights(
    count: int,
    weights: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weights is None:
        return torch.full((count,), 1.0 / count, device=device, dtype=dtype)
    value = weights.to(device=device, dtype=dtype)
    if value.shape != (count,) or not bool(torch.isfinite(value).all()):
        raise ValueError("V45 geometry weights must align and be finite")
    if not bool((value > 0).all()):
        raise ValueError("V45 geometry weights must be positive")
    return value / value.sum()


def covariance_rank_metrics(
    fields: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> dict[str, float]:
    if fields.ndim != 2 or fields.shape[1] != 1024:
        raise ValueError("V45 rank metrics require [N,1024] fields")
    if len(fields) < 2 or not bool(torch.isfinite(fields).all()):
        raise ValueError("V45 rank fields must be finite and nontrivial")
    work = fields.float()
    normalized = _normalized_weights(
        len(work),
        weights,
        device=work.device,
        dtype=work.dtype,
    )
    mean = (normalized[:, None] * work).sum(dim=0)
    centered = work - mean
    covariance = (centered.transpose(0, 1) * normalized[None]) @ centered
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    probabilities = (eigenvalues / total).clamp_min(1e-30)
    effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
    stable_rank = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    return {
        "effective_rank": float(effective_rank),
        "stable_rank": float(stable_rank),
        "trace": float(total),
        "maximum_eigenvalue": float(eigenvalues.max()),
    }


def field_geometry_metrics(
    directions: torch.Tensor,
    radii: torch.Tensor,
    *,
    weights: torch.Tensor,
) -> dict[str, float]:
    if directions.ndim != 2 or directions.shape[1] != 1024:
        raise ValueError("V45 field geometry requires [N,1024] directions")
    if radii.shape != (len(directions),):
        raise ValueError("V45 radii must align with directions")
    normalized = _normalized_weights(
        len(directions),
        weights,
        device=directions.device,
        dtype=directions.dtype,
    )
    resultant = (normalized[:, None] * directions).sum(dim=0).norm()
    ranks = covariance_rank_metrics(directions, weights=normalized)
    return {
        "examples": float(len(directions)),
        "weighted_resultant_length": float(resultant),
        "radius_minimum": float(radii.min()),
        "radius_median": float(radii.median()),
        "radius_maximum": float(radii.max()),
        **ranks,
    }


@torch.no_grad()
def matrix_power_control_directions(
    field: NoiseLimitedRetinalFieldV45,
    dct_fields: torch.Tensor,
    *,
    power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= power <= 0.5:
        raise ValueError("V45 control matrix power must lie in [0,0.5]")
    field._validate_dct(dct_fields)
    work = dct_fields.float()
    mean = field.mean_dct.to(device=work.device, dtype=work.dtype)
    vectors = field.eigenvectors.to(device=work.device, dtype=work.dtype)
    values = field.eigenvalues.to(device=work.device, dtype=work.dtype)
    mean_variance = values.clamp_min(0.0).sum() / field.config.field_dim
    scale = (
        values.clamp_min(0.0) + field.config.ridge_ratio * mean_variance
    ).pow(-power)
    transformed = ((work - mean) @ vectors * scale) @ vectors.transpose(0, 1)
    radius = transformed.norm(dim=-1)
    direction = F.normalize(transformed, dim=-1)
    return direction, radius


@torch.no_grad()
def retrieval_metrics(
    query: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    if query.shape != bank.shape or query.ndim != 2:
        raise ValueError("V45 retrieval query and bank must have equal [N,D] shapes")
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(bank).all()):
        raise ValueError("V45 retrieval fields must be finite")
    query = F.normalize(query.float(), dim=-1)
    bank = F.normalize(bank.float(), dim=-1)
    logits = query @ bank.transpose(0, 1)
    target = torch.arange(len(query), device=query.device)
    target_cosine = logits[target, target]
    top1 = logits.argmax(dim=-1) == target
    top5 = logits.topk(min(5, len(bank)), dim=-1).indices
    return {
        "examples": float(len(query)),
        "top1": float(top1.float().mean()),
        "top5": float((top5 == target[:, None]).any(dim=1).float().mean()),
        "target_cosine": float(target_cosine.mean()),
        "target_cosine_minimum": float(target_cosine.min()),
    }


@torch.no_grad()
def roundtrip_metrics(
    field: NoiseLimitedRetinalFieldV45,
    pixels: torch.Tensor,
) -> dict[str, float | bool]:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (1, 32, 32):
        raise ValueError("V45 roundtrip pixels must be [N,1,32,32]")
    work = pixels.detach().to(device="cpu", dtype=torch.float32)
    target = (work >= field.config.binary_threshold).to(torch.float64)
    source_dct = field.dct.encode(work).to(torch.float64)
    components = field.encode_dct(source_dct, exact=True)
    reconstructed_dct = field.decode_dct(
        components.direction,
        components.log_radius,
        exact=True,
    )
    reconstructed = field.binary(
        components.direction,
        components.log_radius,
        exact=True,
    ).to(torch.float64)
    true_positive = (reconstructed * target).sum()
    false_positive = (reconstructed * (1.0 - target)).sum()
    false_negative = ((1.0 - reconstructed) * target).sum()
    denominator = 2.0 * true_positive + false_positive + false_negative
    ink_f1 = 2.0 * true_positive / denominator.clamp_min(1.0)
    blank = reconstructed.flatten(1).sum(dim=1) == 0
    finite = (
        torch.isfinite(components.direction).all()
        and torch.isfinite(components.log_radius).all()
        and torch.isfinite(reconstructed_dct).all()
    )
    return {
        "examples": float(len(work)),
        "maximum_dct_absolute_error": float(
            (reconstructed_dct - source_dct).abs().max()
        ),
        "binary_pixel_accuracy": float((reconstructed == target).float().mean()),
        "ink_f1": float(ink_f1),
        "blank_rate": float(blank.float().mean()),
        "finite": bool(finite),
    }


def combine_roundtrip_metrics(
    metrics: Mapping[str, Mapping[str, float | bool]],
) -> dict[str, float | bool]:
    if not metrics:
        raise ValueError("V45 roundtrip summary cannot be empty")
    return {
        "banks": float(len(metrics)),
        "examples": sum(float(item["examples"]) for item in metrics.values()),
        "maximum_dct_absolute_error": max(
            float(item["maximum_dct_absolute_error"]) for item in metrics.values()
        ),
        "minimum_binary_pixel_accuracy": min(
            float(item["binary_pixel_accuracy"]) for item in metrics.values()
        ),
        "minimum_ink_f1": min(float(item["ink_f1"]) for item in metrics.values()),
        "maximum_blank_rate": max(
            float(item["blank_rate"]) for item in metrics.values()
        ),
        "all_finite": all(bool(item["finite"]) for item in metrics.values()),
    }


def pair_displacement_metrics(fields: torch.Tensor) -> dict[str, float]:
    if fields.ndim != 3 or tuple(fields.shape[1:]) != (2, 1024):
        raise ValueError("V45 pair fields must be [N,2,1024]")
    if not bool(torch.isfinite(fields).all()):
        raise ValueError("V45 pair fields must be finite")
    unit = F.normalize(fields.float(), dim=-1)
    pair_cosine = (unit[:, 0] * unit[:, 1]).sum(dim=-1)
    delta = unit[:, 0] - unit[:, 1]
    norms = delta.norm(dim=-1)
    ranks = covariance_rank_metrics(delta)
    return {
        "pairs": float(len(fields)),
        "candidate_pair_cosine": float(pair_cosine.mean()),
        "candidate_pair_separation": float((1.0 - pair_cosine).mean()),
        "delta_norm_p05": float(torch.quantile(norms, 0.05)),
        "delta_norm_median": float(norms.median()),
        "delta_norm_p95": float(torch.quantile(norms, 0.95)),
        "delta_effective_rank": ranks["effective_rank"],
        "delta_stable_rank": ranks["stable_rank"],
    }


def noise_limited_retinal_field_v45_gate_report(
    *,
    roundtrip: Mapping[str, float | bool],
    raw_geometry: Mapping[str, float],
    field_geometry: Mapping[str, float],
    held_fonts: Mapping[str, Mapping[str, Mapping[str, float]]],
    shifts: Mapping[str, Mapping[str, Mapping[str, float]]],
    raw_pairs: Mapping[str, float],
    field_pairs: Mapping[str, float],
    fit_boundary_clean: bool,
    frozen_partition_opened: bool,
    peak_allocated_vram_gib: float,
    elapsed_seconds: float,
) -> dict[str, bool]:
    font_raw = [item["raw"]["top1"] for item in held_fonts.values()]
    font_field = [item["v45"]["top1"] for item in held_fonts.values()]
    shift_raw = [item["raw"]["top1"] for item in shifts.values()]
    shift_field = [item["v45"]["top1"] for item in shifts.values()]
    return {
        "finite_and_nonblank": (
            bool(roundtrip["all_finite"])
            and float(roundtrip["maximum_blank_rate"]) == 0.0
        ),
        "fp64_dct_roundtrip": (
            float(roundtrip["maximum_dct_absolute_error"]) < 2e-8
        ),
        "exact_binary_roundtrip": (
            float(roundtrip["minimum_binary_pixel_accuracy"]) == 1.0
            and float(roundtrip["minimum_ink_f1"]) == 1.0
        ),
        "common_resultant_removed": (
            float(field_geometry["weighted_resultant_length"]) < 0.05
            and float(field_geometry["weighted_resultant_length"])
            < 0.10 * float(raw_geometry["weighted_resultant_length"])
        ),
        "effective_rank_gain": (
            float(field_geometry["effective_rank"])
            >= 1.20 * float(raw_geometry["effective_rank"])
        ),
        "held_font_continuity": (
            all(new >= old - 0.01 - V45_GATE_EPSILON for old, new in zip(font_raw, font_field))
            and sum(font_field) / len(font_field)
            >= sum(font_raw) / len(font_raw) - V45_GATE_EPSILON
        ),
        "one_pixel_shift_continuity": (
            all(new >= old - 0.02 - V45_GATE_EPSILON for old, new in zip(shift_raw, shift_field))
            and sum(shift_field) / len(shift_field)
            >= sum(shift_raw) / len(shift_raw) - V45_GATE_EPSILON
        ),
        "pair_cosine_reduction": (
            float(raw_pairs["candidate_pair_cosine"])
            - float(field_pairs["candidate_pair_cosine"])
            >= 0.25 - V45_GATE_EPSILON
        ),
        "pair_low_quantile_norm_gain": (
            float(field_pairs["delta_norm_p05"])
            >= 1.50 * float(raw_pairs["delta_norm_p05"])
        ),
        "pair_effective_rank_gain": (
            float(field_pairs["delta_effective_rank"])
            >= 1.10 * float(raw_pairs["delta_effective_rank"])
        ),
        "pair_stable_rank_gain": (
            float(field_pairs["delta_stable_rank"])
            >= 1.08 * float(raw_pairs["delta_stable_rank"])
        ),
        "image_only_training_boundary": (
            fit_boundary_clean and not frozen_partition_opened
        ),
        "resource_bound": (
            peak_allocated_vram_gib < 4.0 and elapsed_seconds < 20.0 * 60.0
        ),
    }


def finite_report(payload: object) -> bool:
    if isinstance(payload, Mapping):
        return all(finite_report(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(finite_report(value) for value in payload)
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return True
    if isinstance(payload, (int, float)):
        return math.isfinite(float(payload))
    return False

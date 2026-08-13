from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .visual_semantic_raster_transducer import (
    VisualSemanticRasterOutput,
    VisualSemanticRasterTransducer,
)


V32_LATENT_STANDARD_DEVIATION_FLOOR = 0.20


@dataclass(frozen=True)
class VisualSemanticRasterLossWeights:
    state: float = 1.0
    pixel: float = 1.0
    edge: float = 0.25
    ink: float = 0.25
    stop: float = 0.20
    variance: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V32 loss weights must be non-negative")


V32_LOSS_WEIGHTS = VisualSemanticRasterLossWeights()
V32_TRAINING_KEYS = (
    "prompt_pixels",
    "prompt_mask",
    "answer_cells",
    "answer_mask",
    "stop_targets",
    "stop_mask",
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[: mask.ndim] != mask.shape:
        raise ValueError("V32 loss mask does not align with values")
    active = mask.float().sum()
    if not bool(active > 0):
        raise ValueError("V32 loss mask contains no active values")
    elements_per_active = values[0, 0].numel() if values.ndim > mask.ndim else 1
    expanded = mask.float()
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return (values.float() * expanded).sum() / (active * elements_per_active)


def diagonal_gaussian_state_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != target.shape or log_scale.shape != target.shape:
        raise ValueError("V32 state distributions and targets must have one shape")
    if active_mask.shape != target.shape[:2]:
        raise ValueError("V32 state mask does not align with target states")
    bounded_log_scale = log_scale.float().clamp(-4.0, 2.0)
    standardized = (target.detach().float() - mean.float()) * (
        -bounded_log_scale
    ).exp()
    values = 0.5 * standardized.square() + bounded_log_scale
    return _masked_mean(values, active_mask)


def raster_pixel_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != target.shape or active_mask.shape != target.shape[:2]:
        raise ValueError("V32 raster BCE tensors do not align")
    values = F.binary_cross_entropy_with_logits(
        logits.float(),
        target.float(),
        reduction="none",
    )
    return _masked_mean(values, active_mask)


def sobel_edges(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim != 5 or pixels.shape[2] != 1:
        raise ValueError("V32 Sobel input must be [B,A,1,H,W]")
    kernel_x = pixels.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    flat = pixels.float().reshape(-1, 1, pixels.shape[-2], pixels.shape[-1])
    horizontal = F.conv2d(flat, kernel_x.float(), padding=1) / 8.0
    vertical = F.conv2d(flat, kernel_y.float(), padding=1) / 8.0
    return torch.cat((horizontal, vertical), dim=1).reshape(
        pixels.shape[0],
        pixels.shape[1],
        2,
        pixels.shape[-2],
        pixels.shape[-1],
    )


def raster_edge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != target.shape or active_mask.shape != target.shape[:2]:
        raise ValueError("V32 edge tensors do not align")
    difference = (sobel_edges(logits.float().sigmoid()) - sobel_edges(target)).abs()
    return _masked_mean(difference, active_mask)


def raster_ink_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if logits.shape != target.shape or active_mask.shape != target.shape[:2]:
        raise ValueError("V32 Dice tensors do not align")
    probabilities = logits.float().sigmoid().flatten(2)
    target = target.float().flatten(2)
    overlap = 2.0 * (probabilities * target).sum(dim=-1)
    scale = probabilities.sum(dim=-1) + target.sum(dim=-1)
    loss_per_cell = 1.0 - (overlap + epsilon) / (scale + epsilon)
    return _masked_mean(loss_per_cell, active_mask)


def stop_position_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or active_mask.shape != targets.shape:
        raise ValueError("V32 stop tensors do not align")
    values = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
    )
    return _masked_mean(values, active_mask)


def latent_variance_floor_loss(
    states: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    standard_deviation_floor: float = V32_LATENT_STANDARD_DEVIATION_FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 3 or active_mask.shape != states.shape[:2]:
        raise ValueError("V32 latent states and mask do not align")
    if standard_deviation_floor <= 0.0:
        raise ValueError("V32 latent standard-deviation floor must be positive")
    weight = active_mask.float().unsqueeze(-1)
    count = weight.sum()
    if not bool(count > 0):
        raise ValueError("V32 latent mask contains no active states")
    mean = (states.float() * weight).sum(dim=(0, 1)) / count
    variance = ((states.float() - mean).square() * weight).sum(dim=(0, 1)) / count
    standard_deviation = (variance + 1e-6).sqrt()
    penalty = F.relu(standard_deviation_floor - standard_deviation).square().mean()
    return penalty, standard_deviation


def visual_semantic_raster_loss_terms(
    output: VisualSemanticRasterOutput,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    missing = set(V32_TRAINING_KEYS).difference(batch)
    if missing:
        raise ValueError(f"V32 training batch is missing {sorted(missing)}")
    answer = batch["answer_cells"]
    answer_mask = batch["answer_mask"]
    variance, latent_std = latent_variance_floor_loss(
        output.target_states,
        answer_mask,
    )
    losses = {
        "state": diagonal_gaussian_state_nll(
            output.state_mean,
            output.state_log_scale,
            output.target_states,
            answer_mask,
        ),
        "pixel": raster_pixel_bce(output.raster_logits, answer, answer_mask),
        "edge": raster_edge_loss(output.raster_logits, answer, answer_mask),
        "ink": raster_ink_dice_loss(output.raster_logits, answer, answer_mask),
        "stop": stop_position_loss(
            output.stop_logits,
            batch["stop_targets"],
            batch["stop_mask"],
        ),
        "variance": variance,
    }
    with torch.no_grad():
        active = answer_mask.float()
        raster_mae = _masked_mean(
            (output.raster_logits.sigmoid() - answer).abs(),
            active,
        )
        raster_accuracy = _masked_mean(
            ((output.raster_logits.sigmoid() >= 0.5) == (answer >= 0.5)).float(),
            active,
        )
        stop_accuracy = _masked_mean(
            ((output.stop_logits >= 0.0) == (batch["stop_targets"] >= 0.5)).float(),
            batch["stop_mask"],
        )
    metrics = {
        "raster_mae": raster_mae.detach(),
        "raster_binary_accuracy": raster_accuracy.detach(),
        "stop_binary_accuracy": stop_accuracy.detach(),
        "latent_std_mean": latent_std.mean().detach(),
        "latent_std_min": latent_std.min().detach(),
        "state_mean_norm": output.state_mean.float().norm(dim=-1).mean().detach(),
    }
    return losses, metrics


def weighted_visual_semantic_raster_loss(
    losses: Mapping[str, torch.Tensor],
    *,
    weights: VisualSemanticRasterLossWeights = V32_LOSS_WEIGHTS,
) -> torch.Tensor:
    required = {"state", "pixel", "edge", "ink", "stop", "variance"}
    if set(losses) != required:
        raise ValueError(f"V32 losses must contain exactly {sorted(required)}")
    return (
        weights.state * losses["state"]
        + weights.pixel * losses["pixel"]
        + weights.edge * losses["edge"]
        + weights.ink * losses["ink"]
        + weights.stop * losses["stop"]
        + weights.variance * losses["variance"]
    )


def visual_semantic_raster_training_microstep(
    model: VisualSemanticRasterTransducer,
    batch: Mapping[str, torch.Tensor],
    *,
    feedback_mode: str = "decoded",
    generator: torch.Generator | None = None,
    weights: VisualSemanticRasterLossWeights = V32_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(
        batch["prompt_pixels"],
        batch["prompt_mask"],
        batch["answer_cells"],
        batch["answer_mask"],
        feedback_mode=feedback_mode,
        generator=generator,
    )
    losses, metrics = visual_semantic_raster_loss_terms(output, batch)
    total = weighted_visual_semantic_raster_loss(losses, weights=weights)
    return total, {
        "loss": total.detach(),
        **{f"loss_{name}": value.detach() for name, value in losses.items()},
        **metrics,
    }


def raster_warmup_microstep(
    model: VisualSemanticRasterTransducer,
    batch: Mapping[str, torch.Tensor],
    *,
    generator: torch.Generator | None = None,
    weights: VisualSemanticRasterLossWeights = V32_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    answer = batch["answer_cells"]
    answer_mask = batch["answer_mask"]
    context = answer.new_zeros(
        answer.shape[0],
        answer.shape[1],
        model.config.planner_dim,
    )
    states = model.encode_target_states(answer, context)
    perturbed = model.perturb_target_states(
        states,
        answer_mask,
        generator=generator,
    )
    logits = model.raster_decoder(perturbed)
    pixel = raster_pixel_bce(logits, answer, answer_mask)
    edge = raster_edge_loss(logits, answer, answer_mask)
    ink = raster_ink_dice_loss(logits, answer, answer_mask)
    variance, latent_std = latent_variance_floor_loss(states, answer_mask)
    total = (
        weights.pixel * pixel
        + weights.edge * edge
        + weights.ink * ink
        + weights.variance * variance
    )
    return total, {
        "loss": total.detach(),
        "loss_pixel": pixel.detach(),
        "loss_edge": edge.detach(),
        "loss_ink": ink.detach(),
        "loss_variance": variance.detach(),
        "latent_std_mean": latent_std.mean().detach(),
        "latent_std_min": latent_std.min().detach(),
    }


__all__ = [
    "V32_LATENT_STANDARD_DEVIATION_FLOOR",
    "V32_LOSS_WEIGHTS",
    "VisualSemanticRasterLossWeights",
    "diagonal_gaussian_state_nll",
    "latent_variance_floor_loss",
    "raster_edge_loss",
    "raster_ink_dice_loss",
    "raster_pixel_bce",
    "raster_warmup_microstep",
    "sobel_edges",
    "stop_position_loss",
    "visual_semantic_raster_loss_terms",
    "visual_semantic_raster_training_microstep",
    "weighted_visual_semantic_raster_loss",
]

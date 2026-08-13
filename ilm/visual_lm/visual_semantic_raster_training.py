from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
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


def stage_cosine_learning_rate(
    update: int,
    *,
    peak: float,
    warmup: int,
    total: int,
    minimum_ratio: float = 0.10,
) -> float:
    if not 1 <= update <= total:
        raise ValueError("V32 learning-rate update must be inside the stage")
    if peak <= 0.0 or not 0 <= warmup < total:
        raise ValueError("V32 learning-rate schedule is invalid")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("V32 minimum learning-rate ratio must be in [0,1]")
    if warmup and update <= warmup:
        return peak * update / warmup
    if update == total:
        return peak * minimum_ratio
    progress = (update - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _normalization_and_bias_parameter_ids(model: nn.Module) -> set[int]:
    normalization_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )
    no_decay: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias" or isinstance(module, normalization_types):
                no_decay.add(id(parameter))
    return no_decay


def visual_semantic_raster_optimizer_groups(
    model: VisualSemanticRasterTransducer,
    *,
    writer_learning_rate: float = 3e-4,
    reader_learning_rate: float = 2e-5,
    weight_decay: float = 0.05,
    reader_final_blocks: int = 2,
) -> list[dict[str, Any]]:
    if writer_learning_rate <= 0 or reader_learning_rate <= 0:
        raise ValueError("V32 optimizer learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("V32 optimizer weight decay must be non-negative")
    if not 1 <= reader_final_blocks <= len(model.reader.encoder.layer):
        raise ValueError("V32 optimizer reader block count is invalid")

    reader_ids: set[int] = set()
    for block in model.reader.encoder.layer[-reader_final_blocks:]:
        reader_ids.update(id(parameter) for parameter in block.parameters())
    reader_ids.update(id(parameter) for parameter in model.reader.layernorm.parameters())
    no_decay_ids = _normalization_and_bias_parameter_ids(model)
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("writer", True): [],
        ("writer", False): [],
        ("reader", True): [],
        ("reader", False): [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("reader.") and id(parameter) not in reader_ids:
            continue
        role = "reader" if id(parameter) in reader_ids else "writer"
        decay = id(parameter) not in no_decay_ids
        groups[(role, decay)].append(parameter)

    output: list[dict[str, Any]] = []
    for role, decay in (("writer", True), ("writer", False), ("reader", True), ("reader", False)):
        parameters = groups[(role, decay)]
        if not parameters:
            continue
        learning_rate = (
            writer_learning_rate if role == "writer" else reader_learning_rate
        )
        output.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "weight_decay": weight_decay if decay else 0.0,
                "role": role,
                "decay": decay,
            }
        )
    optimized_ids = [id(parameter) for group in output for parameter in group["params"]]
    if len(optimized_ids) != len(set(optimized_ids)):
        raise RuntimeError("V32 optimizer contains duplicate parameters")
    return output


def set_visual_semantic_raster_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    writer: float,
    reader: float,
) -> None:
    for group in optimizer.param_groups:
        role = group.get("role")
        if role == "writer":
            group["lr"] = float(writer)
        elif role == "reader":
            group["lr"] = float(reader)
        else:
            raise ValueError("V32 optimizer group has no recognized role")


def visual_semantic_raster_optimizer_receipt(
    model: VisualSemanticRasterTransducer,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    rows = []
    for group in groups:
        parameters = list(group["params"])
        rows.append(
            {
                "role": group["role"],
                "decay": bool(group["decay"]),
                "weight_decay": float(group["weight_decay"]),
                "parameters": sum(parameter.numel() for parameter in parameters),
                "tensors": len(parameters),
            }
        )
    optimized = {
        names[id(parameter)] for group in groups for parameter in group["params"]
    }
    return {
        "groups": rows,
        "optimized_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in optimized
        ),
        "permanently_frozen_reader_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("reader.") and name not in optimized
        ),
        "optimized_parameter_names": sorted(optimized),
    }


class SelectiveExponentialMovingAverage:
    def __init__(
        self,
        model: nn.Module,
        parameter_names: Sequence[str],
        *,
        decay: float = 0.999,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("V32 EMA decay must be in [0,1)")
        source = dict(model.named_parameters())
        names = tuple(dict.fromkeys(parameter_names))
        missing = set(names).difference(source)
        if missing:
            raise ValueError(f"V32 EMA parameters are missing: {sorted(missing)}")
        self.decay = float(decay)
        self.names = names
        self.shadow = {
            name: source[name].detach().float().clone() for name in self.names
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].lerp_(source[name].detach().float(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        destination = dict(model.named_parameters())
        for name in self.names:
            destination[name].copy_(self.shadow[name].to(destination[name].dtype))

    def state_dict(self, *, cpu: bool = True) -> dict[str, Any]:
        shadow = {
            name: value.detach().cpu().clone() if cpu else value.detach().clone()
            for name, value in self.shadow.items()
        }
        return {"decay": self.decay, "names": list(self.names), "shadow": shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        names = tuple(state["names"])
        if names != self.names:
            raise ValueError("V32 EMA parameter names differ from the run")
        if float(state["decay"]) != self.decay:
            raise ValueError("V32 EMA decay differs from the run")
        shadow = state["shadow"]
        for name in self.names:
            value = shadow[name]
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"V32 EMA shape differs for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name]))


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
    "SelectiveExponentialMovingAverage",
    "VisualSemanticRasterLossWeights",
    "diagonal_gaussian_state_nll",
    "latent_variance_floor_loss",
    "raster_edge_loss",
    "raster_ink_dice_loss",
    "raster_pixel_bce",
    "raster_warmup_microstep",
    "set_visual_semantic_raster_learning_rates",
    "sobel_edges",
    "stage_cosine_learning_rate",
    "stop_position_loss",
    "visual_semantic_raster_loss_terms",
    "visual_semantic_raster_training_microstep",
    "visual_semantic_raster_optimizer_groups",
    "visual_semantic_raster_optimizer_receipt",
    "weighted_visual_semantic_raster_loss",
]

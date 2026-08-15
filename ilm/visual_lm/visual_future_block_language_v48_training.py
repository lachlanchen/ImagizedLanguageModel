from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_language_training import dynamic_visual_contrastive_loss
from .visual_future_block_language_v48 import VisualFutureBlockLanguageModelV48


@dataclass(frozen=True)
class VisualFutureBlockLossWeightsV48:
    contrastive: float = 1.00
    angular: float = 0.25
    pixel: float = 0.20
    pixel_dice: float = 0.50
    horizons: tuple[float, float, float, float] = (1.0, 0.5, 0.25, 0.125)

    def __post_init__(self) -> None:
        scalar = (
            self.contrastive,
            self.angular,
            self.pixel,
            self.pixel_dice,
            *self.horizons,
        )
        if any(value < 0.0 for value in scalar):
            raise ValueError("V48 loss weights must be non-negative")
        if self.horizons != (1.0, 0.5, 0.25, 0.125):
            raise ValueError("V48 horizon weights are frozen")


V48_LOSS_WEIGHTS = VisualFutureBlockLossWeightsV48()


@dataclass
class VisualFutureHorizonLossV48:
    loss: torch.Tensor
    contrastive: torch.Tensor
    angular: torch.Tensor
    pixel: torch.Tensor
    pixel_bce: torch.Tensor
    pixel_dice: torch.Tensor
    in_batch_top1: torch.Tensor


@dataclass
class VisualFutureBlockLossV48:
    loss: torch.Tensor
    horizons: tuple[
        VisualFutureHorizonLossV48,
        VisualFutureHorizonLossV48,
        VisualFutureHorizonLossV48,
        VisualFutureHorizonLossV48,
    ]
    contrastive_scale: torch.Tensor
    selected_positions: int

    def detached_metrics(self) -> dict[str, float]:
        metrics = {
            "loss": float(self.loss.detach()),
            "contrastive_scale": float(self.contrastive_scale.detach()),
            "selected_positions": float(self.selected_positions),
        }
        for index, horizon in enumerate(self.horizons, start=1):
            metrics.update(
                {
                    f"h{index}_loss": float(horizon.loss.detach()),
                    f"h{index}_contrastive": float(
                        horizon.contrastive.detach()
                    ),
                    f"h{index}_angular": float(horizon.angular.detach()),
                    f"h{index}_pixel": float(horizon.pixel.detach()),
                    f"h{index}_pixel_bce": float(horizon.pixel_bce.detach()),
                    f"h{index}_pixel_dice": float(
                        horizon.pixel_dice.detach()
                    ),
                    f"h{index}_in_batch_top1": float(
                        horizon.in_batch_top1.detach()
                    ),
                }
            )
        return metrics


def _selected_indices_v48(
    count: int,
    *,
    maximum: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1 or maximum < 1:
        raise ValueError("V48 position selection requires positive sizes")
    if count <= maximum:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device, generator=generator)[:maximum]


def _decoded_pixel_loss_v48(
    model: VisualFutureBlockLanguageModelV48,
    fields: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fields.ndim != 2 or fields.shape[1] != model.config.field_dim:
        raise ValueError("V48 pixel fields must be [N,1024]")
    if target_pixels.shape != (len(fields), 1, 32, 32):
        raise ValueError("V48 pixel targets must be [N,1,32,32]")
    target = (target_pixels.float() >= model.config.binary_threshold).float()
    full_fields = fields.float() * (model.config.field_dim**0.5)
    logits = (
        model.config.decoder_sharpness
        * model.field.signed_spatial(full_fields)
    )
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    overlap = 2.0 * (probability * target).flatten(1).sum(dim=1)
    scale = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    return bce + dice_weight * dice, bce, dice


def visual_future_block_language_loss_v48(
    model: VisualFutureBlockLanguageModelV48,
    output: Mapping[str, torch.Tensor],
    future_pixels: torch.Tensor,
    *,
    generator: torch.Generator,
    maximum_positions: int = 512,
    weights: VisualFutureBlockLossWeightsV48 = V48_LOSS_WEIGHTS,
) -> VisualFutureBlockLossV48:
    required = {"hidden_states", "future_anchor_fields"}
    if not required.issubset(output):
        raise ValueError("V48 model output lacks future visual states")
    hidden = output["hidden_states"]
    anchors = output["future_anchor_fields"]
    expected_pixels = (*hidden.shape[:2], 4, 1, 32, 32)
    if future_pixels.shape != expected_pixels:
        raise ValueError("V48 future pixels do not align with hidden states")
    if anchors.shape != (*hidden.shape[:2], 4, model.config.field_dim):
        raise ValueError("V48 future anchors do not align with hidden states")
    target_fields = model.field.encode_unit(future_pixels).detach()
    count = hidden.shape[0] * hidden.shape[1]
    selected = _selected_indices_v48(
        count,
        maximum=maximum_positions,
        device=anchors.device,
        generator=generator,
    )

    horizon_losses: list[VisualFutureHorizonLossV48] = []
    total = anchors.new_zeros((), dtype=torch.float32)
    for horizon in range(model.config.future_horizons):
        predicted = anchors[:, :, horizon].flatten(0, 1)
        targets = target_fields[:, :, horizon].flatten(0, 1)
        pixels = future_pixels[:, :, horizon].flatten(0, 1)
        contrastive, in_batch_top1 = dynamic_visual_contrastive_loss(
            predicted[selected],
            targets[selected],
            scale=model.contrastive_scale,
        )
        angular = (
            1.0 - (predicted.float() * targets.float()).sum(dim=-1)
        ).mean()
        pixel, pixel_bce, pixel_dice = _decoded_pixel_loss_v48(
            model,
            predicted[selected],
            pixels[selected],
            dice_weight=weights.pixel_dice,
        )
        loss = (
            weights.contrastive * contrastive
            + weights.angular * angular
            + weights.pixel * pixel
        )
        total = total + weights.horizons[horizon] * loss
        horizon_losses.append(
            VisualFutureHorizonLossV48(
                loss=loss,
                contrastive=contrastive,
                angular=angular,
                pixel=pixel,
                pixel_bce=pixel_bce,
                pixel_dice=pixel_dice,
                in_batch_top1=in_batch_top1,
            )
        )
    if len(horizon_losses) != 4:
        raise RuntimeError("V48 did not produce four horizon losses")
    return VisualFutureBlockLossV48(
        loss=total,
        horizons=(
            horizon_losses[0],
            horizon_losses[1],
            horizon_losses[2],
            horizon_losses[3],
        ),
        contrastive_scale=model.contrastive_scale,
        selected_positions=len(selected),
    )


__all__ = [
    "V48_LOSS_WEIGHTS",
    "VisualFutureBlockLossV48",
    "VisualFutureBlockLossWeightsV48",
    "VisualFutureHorizonLossV48",
    "visual_future_block_language_loss_v48",
]

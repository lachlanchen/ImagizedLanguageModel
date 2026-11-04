from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .causal_visual_block_diffusion_v53 import CausalVisualBlockDiffusion


@dataclass(frozen=True)
class VisualBlockLossWeights:
    foreground: float = 0.10
    dice: float = 0.10
    field: float = 0.10
    representation: float = 0.10
    order: float = 0.25
    order_margin: float = 0.02


V53_LOSS_WEIGHTS = VisualBlockLossWeights()


@dataclass
class VisualBlockDiffusionLoss:
    loss: torch.Tensor
    velocity: torch.Tensor
    foreground: torch.Tensor
    dice: torch.Tensor
    field: torch.Tensor
    representation: torch.Tensor
    order: torch.Tensor
    ordered_field_cosine: torch.Tensor
    shuffled_field_cosine: torch.Tensor
    dropped_condition_fraction: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "velocity_mse": float(self.velocity.detach()),
            "foreground_clean_mse": float(self.foreground.detach()),
            "soft_ink_dice_loss": float(self.dice.detach()),
            "visual_field_loss": float(self.field.detach()),
            "representation_loss": float(self.representation.detach()),
            "order_margin_loss": float(self.order.detach()),
            "ordered_high_noise_field_cosine": float(
                self.ordered_field_cosine.detach()
            ),
            "shuffled_high_noise_field_cosine": float(
                self.shuffled_field_cosine.detach()
            ),
            "ordered_minus_shuffled_field_cosine": float(
                (self.ordered_field_cosine - self.shuffled_field_cosine).detach()
            ),
            "dropped_condition_fraction": float(
                self.dropped_condition_fraction.detach()
            ),
        }


def stratified_future_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    synchronous: bool = False,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("V53 timestep batch must be positive")
    if synchronous:
        return (
            torch.randint(0, 100, (batch_size, 1), device=device, generator=generator)
            .expand(-1, 4)
            .clone()
        )
    ranges = ((0, 24), (25, 59), (60, 89), (90, 99))
    rows: list[torch.Tensor] = []
    for _ in range(batch_size):
        values = torch.stack(
            [
                torch.randint(
                    lower,
                    upper + 1,
                    (),
                    device=device,
                    generator=generator,
                )
                for lower, upper in ranges
            ]
        )
        rows.append(values[torch.randperm(4, device=device, generator=generator)])
    return torch.stack(rows)


def _visual_fields(
    model: CausalVisualBlockDiffusion,
    signed_cells: torch.Tensor,
) -> torch.Tensor:
    # Do not use encode_unit here: it thresholds pixels and cuts gradients.
    basis = model.reader.field.basis.float()
    spatial = signed_cells.flatten(0, 1)[:, 0].float()
    spectral = basis @ spatial @ basis.transpose(0, 1)
    fields = F.normalize(spectral.flatten(1), dim=-1, eps=1e-8)
    return fields.reshape(len(signed_cells), 4, -1)


def visual_block_diffusion_loss(
    model: CausalVisualBlockDiffusion,
    *,
    context: torch.Tensor,
    shuffled_context: torch.Tensor,
    target_cells: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    drop_condition: torch.Tensor,
    weights: VisualBlockLossWeights = V53_LOSS_WEIGHTS,
) -> VisualBlockDiffusionLoss:
    if target_cells.shape != (len(context), 4, 1, 32, 32):
        raise ValueError("V53 targets must contain four visible cells")
    if drop_condition.shape != (len(context),) or drop_condition.dtype != torch.bool:
        raise ValueError("V53 condition-drop mask must be a boolean row vector")
    with torch.no_grad():
        hidden, anchors, plan = model.read_condition(context)
        shuffled_hidden, shuffled_anchors, shuffled_plan = model.read_condition(
            shuffled_context
        )
    keep = (~drop_condition).float()[:, None]
    primary_hidden = hidden * keep
    primary_anchors = anchors * keep
    primary_plan = plan * keep[:, :, None, None]
    noisy = model.add_noise(target_cells, timesteps, noise)
    target_velocity = model.velocity_target(target_cells, timesteps, noise)
    predicted_velocity, representation = model.denoiser(
        noisy,
        primary_hidden,
        primary_anchors,
        primary_plan,
        timesteps,
    )
    predicted_clean, _ = model.velocity_to_clean_noise(
        noisy, predicted_velocity, timesteps
    )
    target_signed = target_cells.float().mul(2.0).sub(1.0)
    velocity = F.mse_loss(predicted_velocity.float(), target_velocity.float())
    foreground_weight = 1.0 + 2.0 * target_cells.float()
    foreground = (
        (predicted_clean - target_signed).square() * foreground_weight
    ).sum() / foreground_weight.sum()
    probability = predicted_clean.add(1.0).mul(0.5).clamp(0.0, 1.0)
    overlap = 2.0 * (probability * target_cells).flatten(2).sum(dim=-1)
    scale = probability.flatten(2).sum(dim=-1) + target_cells.flatten(2).sum(dim=-1)
    dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    target_fields = _visual_fields(model, target_signed).detach()
    predicted_fields = _visual_fields(model, predicted_clean)
    field = (1.0 - (predicted_fields * target_fields).sum(dim=-1)).mean()
    representation_loss = (
        1.0 - (representation.float() * target_fields).sum(dim=-1)
    ).mean()

    shuffled_velocity, _ = model.denoiser(
        noisy,
        shuffled_hidden,
        shuffled_anchors,
        shuffled_plan,
        timesteps,
    )
    shuffled_clean, _ = model.velocity_to_clean_noise(
        noisy, shuffled_velocity, timesteps
    )
    shuffled_fields = _visual_fields(model, shuffled_clean)
    ordered_cosine = (predicted_fields * target_fields).sum(dim=-1)
    shuffled_cosine = (shuffled_fields * target_fields).sum(dim=-1)
    eligible = (timesteps >= 60) & (~drop_condition[:, None])
    count = eligible.sum().clamp_min(1)
    ordered_high = (ordered_cosine * eligible).sum() / count
    shuffled_high = (shuffled_cosine * eligible).sum() / count
    order = (
        F.softplus(weights.order_margin - (ordered_cosine - shuffled_cosine)) * eligible
    ).sum() / count
    total = (
        velocity
        + weights.foreground * foreground
        + weights.dice * dice
        + weights.field * field
        + weights.representation * representation_loss
        + weights.order * order
    )
    return VisualBlockDiffusionLoss(
        loss=total,
        velocity=velocity,
        foreground=foreground,
        dice=dice,
        field=field,
        representation=representation_loss,
        order=order,
        ordered_field_cosine=ordered_high,
        shuffled_field_cosine=shuffled_high,
        dropped_condition_fraction=drop_condition.float().mean(),
    )


__all__ = [
    "V53_LOSS_WEIGHTS",
    "VisualBlockDiffusionLoss",
    "VisualBlockLossWeights",
    "stratified_future_timesteps",
    "visual_block_diffusion_loss",
]

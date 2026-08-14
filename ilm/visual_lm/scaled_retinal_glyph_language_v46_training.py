from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_language_training import empirical_energy_score
from .scaled_retinal_glyph_language_v46 import (
    ScaledRetinalGlyphLanguageModelV46,
)


@dataclass(frozen=True)
class ScaledRetinalGlyphLanguageV46LossWeights:
    contrastive: float = 1.0
    anchor: float = 0.25
    pixel: float = 0.20
    energy: float = 0.50
    sample: float = 0.25
    pixel_dice: float = 0.50

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V46 loss weights must be non-negative")


V46_LOSS_WEIGHTS = ScaledRetinalGlyphLanguageV46LossWeights()


@dataclass
class ScaledRetinalGlyphLanguageV46Loss:
    loss: torch.Tensor
    contrastive: torch.Tensor
    anchor: torch.Tensor
    pixel: torch.Tensor
    pixel_bce: torch.Tensor
    pixel_dice: torch.Tensor
    energy: torch.Tensor
    energy_target_distance: torch.Tensor
    energy_sample_distance: torch.Tensor
    sample: torch.Tensor
    sample_target_cosine: torch.Tensor
    sample_best_target_cosine: torch.Tensor
    anchor_radius_mae: torch.Tensor
    anchor_relative_radius_mae: torch.Tensor
    sample_radius_mae: torch.Tensor
    in_batch_top1: torch.Tensor
    contrastive_scale: torch.Tensor
    contrastive_positions: int
    energy_positions: int
    energy_samples: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "contrastive": float(self.contrastive.detach()),
            "anchor": float(self.anchor.detach()),
            "pixel": float(self.pixel.detach()),
            "pixel_bce": float(self.pixel_bce.detach()),
            "pixel_dice": float(self.pixel_dice.detach()),
            "energy": float(self.energy.detach()),
            "energy_target_distance": float(self.energy_target_distance.detach()),
            "energy_sample_distance": float(self.energy_sample_distance.detach()),
            "sample": float(self.sample.detach()),
            "sample_target_cosine": float(self.sample_target_cosine.detach()),
            "sample_best_target_cosine": float(
                self.sample_best_target_cosine.detach()
            ),
            "anchor_radius_mae": float(self.anchor_radius_mae.detach()),
            "anchor_relative_radius_mae": float(
                self.anchor_relative_radius_mae.detach()
            ),
            "sample_radius_mae": float(self.sample_radius_mae.detach()),
            "in_batch_top1": float(self.in_batch_top1.detach()),
            "contrastive_scale": float(self.contrastive_scale.detach()),
            "contrastive_positions": float(self.contrastive_positions),
            "energy_positions": float(self.energy_positions),
            "energy_samples": float(self.energy_samples),
        }


def _selected_indices(
    count: int,
    *,
    maximum: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1 or maximum < 1:
        raise ValueError("V46 position selection requires positive sizes")
    if count <= maximum:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device, generator=generator)[:maximum]


def exact_raster_positive_mask(target_pixels: torch.Tensor) -> torch.Tensor:
    if target_pixels.ndim != 4 or tuple(target_pixels.shape[1:]) != (1, 32, 32):
        raise ValueError("V46 positive rasters must be [N,1,32,32]")
    bits = (target_pixels.float() >= 0.5).flatten(1).reshape(-1, 32, 32)
    powers = torch.bitwise_left_shift(
        torch.ones(32, dtype=torch.int64, device=target_pixels.device),
        torch.arange(32, dtype=torch.int64, device=target_pixels.device),
    )
    packed = (bits.to(torch.int64) * powers).sum(dim=-1)
    positives = (packed[:, None, :] == packed[None, :, :]).all(dim=-1)
    positives.fill_diagonal_(True)
    if not positives.any(dim=1).all():
        raise RuntimeError("every V46 contrastive row requires a visual positive")
    return positives


def dynamic_scaled_retinal_contrastive_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.ndim != 2 or targets.shape != predicted.shape:
        raise ValueError("V46 contrastive fields must be matching [N,1024]")
    if positive_mask.shape != (len(predicted), len(predicted)):
        raise ValueError("V46 positive mask does not align with contrastive rows")
    if positive_mask.dtype != torch.bool:
        raise TypeError("V46 positive mask must be boolean")
    predicted = F.normalize(predicted.float(), dim=-1, eps=1e-8)
    targets = F.normalize(targets.float(), dim=-1, eps=1e-8)
    logits = scale.float() * predicted @ targets.transpose(0, 1)
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    loss = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    selected = logits.argmax(dim=1)
    accuracy = positive_mask[
        torch.arange(len(logits), device=logits.device),
        selected,
    ].float().mean()
    return loss, accuracy


def _decoded_pixel_loss(
    model: ScaledRetinalGlyphLanguageModelV46,
    anchors: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchors.ndim != 2 or anchors.shape[1] != model.config.field_dim:
        raise ValueError("V46 pixel anchors must be [N,1024]")
    if target_pixels.shape != (len(anchors), 1, 32, 32):
        raise ValueError("V46 pixel targets must be [N,1,32,32]")
    target = (target_pixels.float() >= model.config.binary_threshold).float()
    logits = model.config.decoder_sharpness * model.field.signed_spatial(anchors)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    overlap = 2.0 * (probability * target).flatten(1).sum(dim=1)
    scale = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    return bce + dice_weight * dice, bce, dice


def scaled_retinal_glyph_language_v46_loss(
    model: ScaledRetinalGlyphLanguageModelV46,
    output: Mapping[str, torch.Tensor],
    target_pixels: torch.Tensor,
    *,
    generator: torch.Generator,
    maximum_contrastive_positions: int = 512,
    maximum_energy_positions: int = 128,
    energy_samples: int = 4,
    weights: ScaledRetinalGlyphLanguageV46LossWeights = V46_LOSS_WEIGHTS,
) -> ScaledRetinalGlyphLanguageV46Loss:
    required = {"hidden_states", "anchor_fields"}
    if not required.issubset(output):
        raise ValueError("V46 model output lacks causal visual state")
    hidden = output["hidden_states"]
    anchors = output["anchor_fields"]
    if hidden.shape[:2] != target_pixels.shape[:2]:
        raise ValueError("V46 hidden states and target stream do not align")
    if anchors.shape != (*target_pixels.shape[:2], model.config.field_dim):
        raise ValueError("V46 anchors and target stream do not align")
    target_fields = model.field.encode(target_pixels).detach()
    flat_hidden = hidden.flatten(0, 1)
    flat_anchors = anchors.flatten(0, 1)
    flat_targets = target_fields.flatten(0, 1)
    flat_pixels = target_pixels.flatten(0, 1)
    count = len(flat_anchors)

    contrastive_indices = _selected_indices(
        count,
        maximum=maximum_contrastive_positions,
        device=flat_anchors.device,
        generator=generator,
    )
    contrastive, in_batch_top1 = dynamic_scaled_retinal_contrastive_loss(
        flat_anchors[contrastive_indices],
        flat_targets[contrastive_indices],
        scale=model.contrastive_scale,
        positive_mask=exact_raster_positive_mask(
            flat_pixels[contrastive_indices]
        ),
    )
    anchor_directions = model.field.directions(flat_anchors)
    target_directions = model.field.directions(flat_targets)
    anchor = (
        1.0 - (anchor_directions * target_directions).sum(dim=-1)
    ).mean()
    pixel, pixel_bce, pixel_dice = _decoded_pixel_loss(
        model,
        flat_anchors[contrastive_indices],
        flat_pixels[contrastive_indices],
        dice_weight=weights.pixel_dice,
    )

    energy_indices = _selected_indices(
        count,
        maximum=maximum_energy_positions,
        device=flat_anchors.device,
        generator=generator,
    )
    generated = model.sample_fields(
        flat_hidden[energy_indices],
        flat_anchors[energy_indices],
        samples=energy_samples,
        generator=generator,
    )
    selected_targets = flat_targets[energy_indices]
    energy, target_distance, sample_distance = empirical_energy_score(
        generated,
        selected_targets,
    )
    generated_directions = model.field.directions(generated)
    selected_directions = target_directions[energy_indices]
    sample_cosine = torch.einsum(
        "nsd,nd->ns",
        generated_directions,
        selected_directions,
    )
    best_cosine, best_indices = sample_cosine.max(dim=1)
    sample = (1.0 - best_cosine).mean()

    anchor_radius = flat_anchors.float().norm(dim=-1)
    target_radius = flat_targets.float().norm(dim=-1).clamp_min(1e-8)
    anchor_radius_mae = (anchor_radius - target_radius).abs().mean()
    anchor_relative_radius_mae = (
        (anchor_radius - target_radius).abs() / target_radius
    ).mean()
    best_samples = generated[
        torch.arange(len(generated), device=generated.device),
        best_indices,
    ]
    sample_radius_mae = (
        best_samples.float().norm(dim=-1) - selected_targets.float().norm(dim=-1)
    ).abs().mean()

    loss = (
        weights.contrastive * contrastive
        + weights.anchor * anchor
        + weights.pixel * pixel
        + weights.energy * energy
        + weights.sample * sample
    )
    return ScaledRetinalGlyphLanguageV46Loss(
        loss=loss,
        contrastive=contrastive,
        anchor=anchor,
        pixel=pixel,
        pixel_bce=pixel_bce,
        pixel_dice=pixel_dice,
        energy=energy,
        energy_target_distance=target_distance,
        energy_sample_distance=sample_distance,
        sample=sample,
        sample_target_cosine=sample_cosine.mean(),
        sample_best_target_cosine=best_cosine.mean(),
        anchor_radius_mae=anchor_radius_mae,
        anchor_relative_radius_mae=anchor_relative_radius_mae,
        sample_radius_mae=sample_radius_mae,
        in_batch_top1=in_batch_top1,
        contrastive_scale=model.contrastive_scale,
        contrastive_positions=len(contrastive_indices),
        energy_positions=len(energy_indices),
        energy_samples=energy_samples,
    )

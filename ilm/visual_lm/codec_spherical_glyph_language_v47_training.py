from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_language_training import empirical_energy_score
from .codec_spherical_glyph_language_v47 import (
    CodecSphericalGlyphLanguageModelV47,
)


@dataclass(frozen=True)
class CodecSphericalGlyphLanguageV47LossWeights:
    contrastive: float = 1.00
    anchor: float = 0.25
    pixel: float = 0.20
    energy: float = 0.50
    sample: float = 0.25
    cycle: float = 0.10
    sample_pixel: float = 0.10
    pixel_dice: float = 0.50

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V47 loss weights must be non-negative")


V47_LOSS_WEIGHTS = CodecSphericalGlyphLanguageV47LossWeights()


@dataclass
class CodecSphericalGlyphLanguageV47Loss:
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
    cycle: torch.Tensor
    soft_proposal_reread_cosine: torch.Tensor
    sample_pixel: torch.Tensor
    sample_pixel_bce: torch.Tensor
    sample_pixel_dice: torch.Tensor
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
            "cycle": float(self.cycle.detach()),
            "soft_proposal_reread_cosine": float(
                self.soft_proposal_reread_cosine.detach()
            ),
            "sample_pixel": float(self.sample_pixel.detach()),
            "sample_pixel_bce": float(self.sample_pixel_bce.detach()),
            "sample_pixel_dice": float(self.sample_pixel_dice.detach()),
            "in_batch_top1": float(self.in_batch_top1.detach()),
            "contrastive_scale": float(self.contrastive_scale.detach()),
            "contrastive_positions": float(self.contrastive_positions),
            "energy_positions": float(self.energy_positions),
            "energy_samples": float(self.energy_samples),
        }


@dataclass
class CodecSphericalGlyphLanguageV47PairLoss:
    loss: torch.Tensor
    arm_accuracy: torch.Tensor
    both_correct_rate: torch.Tensor
    mean_margin: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "pair": float(self.loss.detach()),
            "pair_arm_accuracy": float(self.arm_accuracy.detach()),
            "pair_both_correct_rate": float(self.both_correct_rate.detach()),
            "pair_mean_margin": float(self.mean_margin.detach()),
        }


def _selected_indices(
    count: int,
    *,
    maximum: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1 or maximum < 1:
        raise ValueError("V47 position selection requires positive sizes")
    if count <= maximum:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device, generator=generator)[:maximum]


def exact_raster_positive_mask_v47(target_pixels: torch.Tensor) -> torch.Tensor:
    if target_pixels.ndim != 4 or tuple(target_pixels.shape[1:]) != (1, 32, 32):
        raise ValueError("V47 positive rasters must be [N,1,32,32]")
    bits = (target_pixels.float() >= 0.5).flatten(1).reshape(-1, 32, 32)
    powers = torch.bitwise_left_shift(
        torch.ones(32, dtype=torch.int64, device=target_pixels.device),
        torch.arange(32, dtype=torch.int64, device=target_pixels.device),
    )
    packed = (bits.to(torch.int64) * powers).sum(dim=-1)
    positives = (packed[:, None, :] == packed[None, :, :]).all(dim=-1)
    positives.fill_diagonal_(True)
    if not positives.any(dim=1).all():
        raise RuntimeError("every V47 contrastive row requires a visual positive")
    return positives


def dynamic_codec_spherical_contrastive_loss_v47(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.ndim != 2 or targets.shape != predicted.shape:
        raise ValueError("V47 contrastive fields must be matching [N,768]")
    if positive_mask.shape != (len(predicted), len(predicted)):
        raise ValueError("V47 positive mask does not align with contrastive rows")
    if positive_mask.dtype != torch.bool:
        raise TypeError("V47 positive mask must be boolean")
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
    model: CodecSphericalGlyphLanguageModelV47,
    fields: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fields.ndim != 2 or fields.shape[1] != model.config.field_dim:
        raise ValueError("V47 pixel fields must be [N,768]")
    if target_pixels.shape != (len(fields), 1, 32, 32):
        raise ValueError("V47 pixel targets must be [N,1,32,32]")
    target = (target_pixels.float() >= model.config.binary_threshold).float()
    logits = model.field.ink_logits(fields)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    overlap = 2.0 * (probability * target).flatten(1).sum(dim=1)
    scale = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    return bce + dice_weight * dice, bce, dice


def codec_spherical_glyph_language_v47_loss(
    model: CodecSphericalGlyphLanguageModelV47,
    output: Mapping[str, torch.Tensor],
    target_pixels: torch.Tensor,
    *,
    generator: torch.Generator,
    maximum_contrastive_positions: int = 512,
    maximum_energy_positions: int = 128,
    energy_samples: int = 4,
    weights: CodecSphericalGlyphLanguageV47LossWeights = V47_LOSS_WEIGHTS,
) -> CodecSphericalGlyphLanguageV47Loss:
    required = {"hidden_states", "anchor_fields"}
    if not required.issubset(output):
        raise ValueError("V47 model output lacks causal visual state")
    hidden = output["hidden_states"]
    anchors = output["anchor_fields"]
    if hidden.shape[:2] != target_pixels.shape[:2]:
        raise ValueError("V47 hidden states and target stream do not align")
    if anchors.shape != (*target_pixels.shape[:2], model.config.field_dim):
        raise ValueError("V47 anchors and target stream do not align")
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
    contrastive, in_batch_top1 = dynamic_codec_spherical_contrastive_loss_v47(
        flat_anchors[contrastive_indices],
        flat_targets[contrastive_indices],
        scale=model.contrastive_scale,
        positive_mask=exact_raster_positive_mask_v47(
            flat_pixels[contrastive_indices]
        ),
    )
    anchor = (
        1.0 - (flat_anchors.float() * flat_targets.float()).sum(dim=-1)
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
    sample_cosine = torch.einsum(
        "nsd,nd->ns",
        generated.float(),
        selected_targets.float(),
    )
    best_cosine, best_indices = sample_cosine.max(dim=1)
    sample = (1.0 - best_cosine).mean()

    soft_reread = model.field.soft_reread(generated)
    proposal_reread_cosine = (generated.float() * soft_reread.float()).sum(dim=-1)
    cycle = (1.0 - proposal_reread_cosine).mean()
    rows = torch.arange(len(generated), device=generated.device)
    best_samples = generated[rows, best_indices]
    sample_pixel, sample_pixel_bce, sample_pixel_dice = _decoded_pixel_loss(
        model,
        best_samples,
        flat_pixels[energy_indices],
        dice_weight=weights.pixel_dice,
    )

    loss = (
        weights.contrastive * contrastive
        + weights.anchor * anchor
        + weights.pixel * pixel
        + weights.energy * energy
        + weights.sample * sample
        + weights.cycle * cycle
        + weights.sample_pixel * sample_pixel
    )
    return CodecSphericalGlyphLanguageV47Loss(
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
        cycle=cycle,
        soft_proposal_reread_cosine=proposal_reread_cosine.mean(),
        sample_pixel=sample_pixel,
        sample_pixel_bce=sample_pixel_bce,
        sample_pixel_dice=sample_pixel_dice,
        in_batch_top1=in_batch_top1,
        contrastive_scale=model.contrastive_scale,
        contrastive_positions=len(contrastive_indices),
        energy_positions=len(energy_indices),
        energy_samples=energy_samples,
    )


def codec_spherical_glyph_language_v47_pair_loss(
    model: CodecSphericalGlyphLanguageModelV47,
    contexts: torch.Tensor,
    candidates: torch.Tensor,
    assignment: torch.Tensor,
) -> CodecSphericalGlyphLanguageV47PairLoss:
    logits = model.pair_logits(contexts, candidates)
    if assignment.shape != logits.shape[:2]:
        raise ValueError("V47 pair assignments do not align with pair logits")
    loss = F.cross_entropy(logits.flatten(0, 1), assignment.flatten())
    selected = logits.argmax(dim=-1)
    correct = selected == assignment
    assigned = logits.gather(2, assignment[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignment)[:, :, None])[:, :, 0]
    return CodecSphericalGlyphLanguageV47PairLoss(
        loss=loss,
        arm_accuracy=correct.float().mean(),
        both_correct_rate=correct.all(dim=1).float().mean(),
        mean_margin=(assigned - other).mean(),
    )


__all__ = [
    "CodecSphericalGlyphLanguageV47Loss",
    "CodecSphericalGlyphLanguageV47LossWeights",
    "CodecSphericalGlyphLanguageV47PairLoss",
    "V47_LOSS_WEIGHTS",
    "codec_spherical_glyph_language_v47_loss",
    "codec_spherical_glyph_language_v47_pair_loss",
    "dynamic_codec_spherical_contrastive_loss_v47",
    "exact_raster_positive_mask_v47",
]

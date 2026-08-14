from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_language import CanonicalGlyphLanguageModel


@dataclass(frozen=True)
class CanonicalGlyphLanguageLossWeights:
    contrastive: float = 1.0
    anchor: float = 0.25
    pixel: float = 0.20
    energy: float = 0.50
    sample: float = 0.25
    pixel_dice: float = 0.50

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V42 loss weights must be non-negative")


V42_LOSS_WEIGHTS = CanonicalGlyphLanguageLossWeights()


@dataclass
class CanonicalGlyphLanguageLoss:
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
        raise ValueError("V42 position selection requires positive sizes")
    if count <= maximum:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device, generator=generator)[:maximum]


def exact_field_positive_mask(target_fields: torch.Tensor) -> torch.Tensor:
    """Use Parseval equality: any differing binary pixel lowers cosine by >=2/1024."""

    if target_fields.ndim != 2 or target_fields.shape[1] != 1024:
        raise ValueError("V42 positive fields must be [N,1024]")
    normalized = F.normalize(target_fields.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    positives = similarity >= 1.0 - 1e-6
    positives.fill_diagonal_(True)
    if not positives.any(dim=1).all():
        raise RuntimeError("every V42 contrastive row requires a visual positive")
    return positives


def dynamic_visual_contrastive_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.ndim != 2 or targets.shape != predicted.shape:
        raise ValueError("V42 contrastive fields must be matching [N,1024]")
    predicted = F.normalize(predicted.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    positives = exact_field_positive_mask(targets)
    logits = scale.float() * predicted @ targets.transpose(0, 1)
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    loss = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    selected = logits.argmax(dim=1)
    accuracy = positives[
        torch.arange(len(logits), device=logits.device), selected
    ].float().mean()
    return loss, accuracy


def empirical_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 2 E||X-y||^beta - E||X-X'||^beta without diagonal pairs."""

    if samples.ndim != 3 or target.shape != samples.shape[::2]:
        expected = (samples.shape[0], samples.shape[2]) if samples.ndim == 3 else None
        if expected is None or target.shape != expected:
            raise ValueError("V42 energy samples and targets do not align")
    if samples.shape[1] < 2:
        raise ValueError("V42 energy score requires at least two samples")
    if not 0.0 < beta < 2.0:
        raise ValueError("V42 energy beta must lie in (0,2)")
    target_distance = (
        (samples.float() - target[:, None].float())
        .square()
        .sum(dim=-1)
        .clamp_min(1e-12)
        .pow(beta / 2.0)
        .mean()
    )
    pairwise = (
        (samples[:, :, None].float() - samples[:, None, :].float())
        .square()
        .sum(dim=-1)
        .clamp_min(1e-12)
        .pow(beta / 2.0)
    )
    count = samples.shape[1]
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=samples.device)
    sample_distance = pairwise[:, off_diagonal].mean()
    return 2.0 * target_distance - sample_distance, target_distance, sample_distance


def _decoded_pixel_loss(
    model: CanonicalGlyphLanguageModel,
    anchors: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchors.ndim != 2 or anchors.shape[1] != model.config.field_dim:
        raise ValueError("V42 pixel anchors must be [N,1024]")
    if target_pixels.shape != (len(anchors), 1, 32, 32):
        raise ValueError("V42 pixel targets must be [N,1,32,32]")
    target = (target_pixels.float() >= model.config.binary_threshold).float()
    full_fields = anchors.float() * (model.config.field_dim**0.5)
    logits = (
        model.config.decoder_sharpness * model.field.signed_spatial(full_fields)
    )
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    overlap = 2.0 * (probability * target).flatten(1).sum(dim=1)
    scale = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    return bce + dice_weight * dice, bce, dice


def canonical_glyph_language_loss(
    model: CanonicalGlyphLanguageModel,
    output: Mapping[str, torch.Tensor],
    target_pixels: torch.Tensor,
    *,
    generator: torch.Generator,
    maximum_contrastive_positions: int = 512,
    maximum_energy_positions: int = 128,
    energy_samples: int = 4,
    weights: CanonicalGlyphLanguageLossWeights = V42_LOSS_WEIGHTS,
) -> CanonicalGlyphLanguageLoss:
    required = {"hidden_states", "anchor_fields"}
    if not required.issubset(output):
        raise ValueError("V42 model output lacks causal visual state")
    hidden = output["hidden_states"]
    anchors = output["anchor_fields"]
    if hidden.shape[:2] != target_pixels.shape[:2]:
        raise ValueError("V42 hidden states and target stream do not align")
    if anchors.shape != (*target_pixels.shape[:2], model.config.field_dim):
        raise ValueError("V42 anchors and target stream do not align")
    target_fields = model.field.encode_unit(target_pixels).detach()
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
    contrastive, in_batch_top1 = dynamic_visual_contrastive_loss(
        flat_anchors[contrastive_indices],
        flat_targets[contrastive_indices],
        scale=model.contrastive_scale,
    )
    anchor = (
        1.0
        - (flat_anchors.float() * flat_targets.float()).sum(dim=-1)
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
    energy, target_distance, sample_distance = empirical_energy_score(
        generated,
        flat_targets[energy_indices],
    )
    sample_cosine = torch.einsum(
        "nsd,nd->ns",
        generated.float(),
        flat_targets[energy_indices].float(),
    )
    best_cosine = sample_cosine.max(dim=1).values
    sample = (1.0 - best_cosine).mean()

    loss = (
        weights.contrastive * contrastive
        + weights.anchor * anchor
        + weights.pixel * pixel
        + weights.energy * energy
        + weights.sample * sample
    )
    return CanonicalGlyphLanguageLoss(
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
        in_batch_top1=in_batch_top1,
        contrastive_scale=model.contrastive_scale,
        contrastive_positions=len(contrastive_indices),
        energy_positions=len(energy_indices),
        energy_samples=energy_samples,
    )

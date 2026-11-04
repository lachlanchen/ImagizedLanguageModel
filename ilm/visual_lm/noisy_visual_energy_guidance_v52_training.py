from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hierarchical_raster_language import CausalRasterReader
from .noisy_visual_energy_guidance_v52 import NoisyVisualEnergyCritic
from .visual_posterior_language_v51 import VisualGlyphScorePrior


@dataclass(frozen=True)
class NoisyVisualEnergyLossWeights:
    row_contrastive: float = 1.0
    symmetric_contrastive: float = 0.25
    energy_transport: float = 0.50
    order_margin: float = 0.25
    order_margin_value: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V52 loss weights must be non-negative")


V52_LOSS_WEIGHTS = NoisyVisualEnergyLossWeights()


@dataclass
class NoisyVisualEnergyLoss:
    loss: torch.Tensor
    row_contrastive: torch.Tensor
    symmetric_contrastive: torch.Tensor
    energy_transport: torch.Tensor
    order_margin: torch.Tensor
    ordered_endpoint_log_mass: torch.Tensor
    shuffled_endpoint_log_mass: torch.Tensor
    in_batch_top1: torch.Tensor
    teacher_in_batch_top1: torch.Tensor
    contrastive_scale: torch.Tensor
    timestep: int
    selected_images: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "row_contrastive": float(self.row_contrastive.detach()),
            "symmetric_contrastive": float(
                self.symmetric_contrastive.detach()
            ),
            "energy_transport": float(self.energy_transport.detach()),
            "order_margin": float(self.order_margin.detach()),
            "ordered_endpoint_log_mass": float(
                self.ordered_endpoint_log_mass.detach()
            ),
            "shuffled_endpoint_log_mass": float(
                self.shuffled_endpoint_log_mass.detach()
            ),
            "ordered_minus_shuffled_log_mass": float(
                (
                    self.ordered_endpoint_log_mass
                    - self.shuffled_endpoint_log_mass
                ).detach()
            ),
            "in_batch_top1": float(self.in_batch_top1.detach()),
            "teacher_in_batch_top1": float(
                self.teacher_in_batch_top1.detach()
            ),
            "contrastive_scale": float(self.contrastive_scale.detach()),
            "timestep": float(self.timestep),
            "selected_images": float(self.selected_images),
        }


def exact_raster_positive_mask(
    reader: CausalRasterReader,
    clean_images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean_images.ndim != 4 or tuple(clean_images.shape[1:]) != (1, 32, 32):
        raise ValueError("V52 clean targets must be [N,1,32,32]")
    fields = reader.field.encode_unit(clean_images).float()
    similarity = fields @ fields.transpose(0, 1)
    positives = similarity >= 1.0 - 1e-6
    positives.fill_diagonal_(True)
    if not bool(positives.any(dim=1).all()):
        raise RuntimeError("every V52 contrastive row requires a positive")
    return positives, fields


def multi_positive_log_mass(
    logits: torch.Tensor,
    positives: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or positives.shape != logits.shape:
        raise ValueError("V52 logits and positive mask must align")
    if positives.dtype != torch.bool or not bool(positives.any(dim=1).all()):
        raise ValueError("V52 each score row requires a boolean positive")
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    return torch.logsumexp(positive_logits, dim=-1) - torch.logsumexp(
        logits, dim=-1
    )


def noise_clean_images(
    prior: VisualGlyphScorePrior,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    if timesteps.shape != (len(clean_images),) or noise.shape != clean_images.shape:
        raise ValueError("V52 corruption inputs do not align")
    noisy = prior.add_noise(clean_images, timesteps, noise)
    clean_rows = timesteps == 0
    if bool(clean_rows.any()):
        noisy = noisy.clone()
        noisy[clean_rows] = clean_images[clean_rows].float().mul(2.0).sub(1.0)
    return noisy


def noisy_visual_energy_loss(
    critic: NoisyVisualEnergyCritic,
    reader: CausalRasterReader,
    prior: VisualGlyphScorePrior,
    *,
    hidden: torch.Tensor,
    anchors: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    ordered_endpoint_hidden: torch.Tensor,
    ordered_endpoint_anchor: torch.Tensor,
    shuffled_endpoint_hidden: torch.Tensor,
    shuffled_endpoint_anchor: torch.Tensor,
    endpoint_images: torch.Tensor,
    endpoint_noise: torch.Tensor,
    endpoint_timesteps: torch.Tensor,
    weights: NoisyVisualEnergyLossWeights = V52_LOSS_WEIGHTS,
) -> NoisyVisualEnergyLoss:
    noisy = noise_clean_images(prior, clean_images, timesteps, noise)
    positives, target_fields = exact_raster_positive_mask(reader, clean_images)
    logits = critic.score_matrix(hidden, anchors, noisy, timesteps).float()
    row_log_mass = multi_positive_log_mass(logits, positives)
    row_contrastive = -row_log_mass.mean()
    symmetric_contrastive = -multi_positive_log_mass(
        logits.transpose(0, 1), positives.transpose(0, 1)
    ).mean()

    teacher_logits = (
        reader.contrastive_scale.float()
        * anchors.float()
        @ target_fields.transpose(0, 1)
    )
    teacher_probability = teacher_logits.detach().softmax(dim=-1)
    energy_transport = -(
        teacher_probability * logits.log_softmax(dim=-1)
    ).sum(dim=-1).mean()

    endpoint_noisy = noise_clean_images(
        prior,
        endpoint_images,
        endpoint_timesteps,
        endpoint_noise,
    )
    endpoint_positives, _ = exact_raster_positive_mask(reader, endpoint_images)
    ordered_logits = critic.score_matrix(
        ordered_endpoint_hidden,
        ordered_endpoint_anchor,
        endpoint_noisy,
        endpoint_timesteps,
    ).float()
    shuffled_logits = critic.score_matrix(
        shuffled_endpoint_hidden,
        shuffled_endpoint_anchor,
        endpoint_noisy,
        endpoint_timesteps,
    ).float()
    ordered_log_mass = multi_positive_log_mass(
        ordered_logits, endpoint_positives
    )
    shuffled_log_mass = multi_positive_log_mass(
        shuffled_logits, endpoint_positives
    )
    order_margin = F.softplus(
        weights.order_margin_value - (ordered_log_mass - shuffled_log_mass)
    ).mean()

    selected = logits.argmax(dim=-1)
    teacher_selected = teacher_logits.argmax(dim=-1)
    rows = torch.arange(len(logits), device=logits.device)
    in_batch_top1 = positives[rows, selected].float().mean()
    teacher_in_batch_top1 = positives[rows, teacher_selected].float().mean()
    total = (
        weights.row_contrastive * row_contrastive
        + weights.symmetric_contrastive * symmetric_contrastive
        + weights.energy_transport * energy_transport
        + weights.order_margin * order_margin
    )
    return NoisyVisualEnergyLoss(
        loss=total,
        row_contrastive=row_contrastive,
        symmetric_contrastive=symmetric_contrastive,
        energy_transport=energy_transport,
        order_margin=order_margin,
        ordered_endpoint_log_mass=ordered_log_mass.mean(),
        shuffled_endpoint_log_mass=shuffled_log_mass.mean(),
        in_batch_top1=in_batch_top1,
        teacher_in_batch_top1=teacher_in_batch_top1,
        contrastive_scale=critic.contrastive_scale,
        timestep=int(timesteps[0]),
        selected_images=len(clean_images),
    )


__all__ = [
    "NoisyVisualEnergyLoss",
    "NoisyVisualEnergyLossWeights",
    "V52_LOSS_WEIGHTS",
    "exact_raster_positive_mask",
    "multi_positive_log_mass",
    "noise_clean_images",
    "noisy_visual_energy_loss",
]

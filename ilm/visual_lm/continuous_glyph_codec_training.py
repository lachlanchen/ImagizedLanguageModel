from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .continuous_glyph_codec import ContinuousGlyphCodecOutput
from .direct_visual_patch_training import sobel_edges


@dataclass(frozen=True)
class ContinuousGlyphCodecLossWeights:
    pixel: float = 1.0
    edge: float = 0.5
    ink: float = 0.5
    boundary_boost: float = 2.0

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V34 loss weights must be non-negative")


V34_LOSS_WEIGHTS = ContinuousGlyphCodecLossWeights()


@dataclass
class ContinuousGlyphCodecLoss:
    loss: torch.Tensor
    pixel: torch.Tensor
    edge: torch.Tensor
    ink: torch.Tensor
    patches: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "pixel": float(self.pixel.detach()),
            "edge": float(self.edge.detach()),
            "ink": float(self.ink.detach()),
            "patches": float(self.patches),
        }


def _patch_edges(patches: torch.Tensor) -> torch.Tensor:
    if patches.ndim != 4 or patches.shape[1] != 1:
        raise ValueError("V34 edge input must have shape [B,1,H,W]")
    return sobel_edges(patches.unsqueeze(1)).squeeze(1)


def continuous_glyph_codec_loss(
    output: ContinuousGlyphCodecOutput,
    targets: torch.Tensor,
    *,
    weights: ContinuousGlyphCodecLossWeights = V34_LOSS_WEIGHTS,
) -> ContinuousGlyphCodecLoss:
    if output.logits.shape != targets.shape:
        raise ValueError("V34 codec logits and targets do not align")
    if targets.ndim != 4 or targets.shape[1:] != (1, 32, 32):
        raise ValueError("V34 codec targets must have shape [B,1,32,32]")
    if targets.shape[0] < 1:
        raise ValueError("V34 codec loss requires at least one patch")
    if not targets.is_floating_point():
        raise TypeError("V34 codec targets must be floating point")

    target = targets.float()
    probability = output.logits.float().sigmoid()
    target_edges = _patch_edges(target)
    predicted_edges = _patch_edges(probability)

    boundary_strength = torch.linalg.vector_norm(
        target_edges,
        dim=1,
        keepdim=True,
    ).clamp(0.0, 1.0)
    pixel_weights = 1.0 + weights.boundary_boost * boundary_strength
    pixel_values = F.binary_cross_entropy_with_logits(
        output.logits.float(),
        target,
        reduction="none",
    )
    pixel = (pixel_values * pixel_weights).sum() / pixel_weights.sum()
    edge = (predicted_edges - target_edges).abs().mean()

    target_ink = (1.0 - target).flatten(1)
    predicted_ink = (1.0 - probability).flatten(1)
    overlap = 2.0 * (predicted_ink * target_ink).sum(dim=1)
    scale = predicted_ink.sum(dim=1) + target_ink.sum(dim=1)
    ink = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    loss = weights.pixel * pixel + weights.edge * edge + weights.ink * ink
    return ContinuousGlyphCodecLoss(
        loss=loss,
        pixel=pixel,
        edge=edge,
        ink=ink,
        patches=targets.shape[0],
    )


def training_latent_noise(
    latents: torch.Tensor,
    *,
    seed: int,
    update: int,
    selection_probability: float = 0.5,
    maximum_sigma: float = 0.05,
) -> torch.Tensor:
    if latents.ndim != 2 or not latents.is_floating_point():
        raise ValueError("V34 latent noise requires floating [B,D] latents")
    if seed < 0 or update < 1:
        raise ValueError("V34 latent-noise seed and update must be positive")
    if not 0.0 <= selection_probability <= 1.0 or maximum_sigma < 0.0:
        raise ValueError("V34 latent-noise settings are invalid")
    generator = torch.Generator(device=latents.device)
    generator.manual_seed(seed + update * 1_000_003)
    selected = (
        torch.rand(
            (latents.shape[0], 1),
            generator=generator,
            device=latents.device,
        )
        < selection_probability
    )
    sigma = torch.rand(
        (latents.shape[0], 1),
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    ).mul(maximum_sigma)
    gaussian = torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    return gaussian * sigma * selected.to(latents.dtype)


def fixed_latent_noise(
    latents: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    if latents.ndim != 2 or not latents.is_floating_point():
        raise ValueError("V34 fixed latent noise requires floating [B,D] latents")
    if sigma < 0.0 or seed < 0:
        raise ValueError("V34 fixed latent-noise settings are invalid")
    generator = torch.Generator(device=latents.device)
    generator.manual_seed(seed)
    return torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    ).mul(sigma)

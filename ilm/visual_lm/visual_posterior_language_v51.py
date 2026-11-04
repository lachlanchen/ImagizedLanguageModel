from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.diffusion.unet2d import UNet2D

from .canonical_glyph_language import OrthonormalGlyphField


V51_ARCHITECTURE = "visual-posterior-language-v51"


@dataclass(frozen=True)
class VisualPosteriorLanguageConfig:
    cell_size: int = 32
    diffusion_steps: int = 100
    base_channels: int = 64
    unet_depth: int = 2
    time_dimension: int = 128
    cosine_offset: float = 0.008
    foreground_clean_weight: float = 0.10
    dice_weight: float = 0.10
    image_to_image_start: int = 30
    sampling_steps: int = 16
    guidance_strength: float = 0.15
    guidance_rms_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V51 requires 32x32 visible writing cells")
        if self.diffusion_steps < 16:
            raise ValueError("V51 diffusion requires at least 16 noise levels")
        if self.base_channels < 16 or self.base_channels % 8:
            raise ValueError("V51 base channels must be a positive multiple of 8")
        if self.unet_depth not in {1, 2, 3} or self.time_dimension < 32:
            raise ValueError("V51 U-Net geometry is invalid")
        if not 0.0 < self.cosine_offset < 0.1:
            raise ValueError("V51 cosine offset is invalid")
        if min(self.foreground_clean_weight, self.dice_weight) < 0.0:
            raise ValueError("V51 loss weights must be non-negative")
        if not 1 <= self.image_to_image_start < self.diffusion_steps:
            raise ValueError("V51 image-to-image start is outside the schedule")
        if not 2 <= self.sampling_steps <= self.image_to_image_start + 1:
            raise ValueError("V51 sampling-step count is invalid")
        if self.guidance_strength < 0.0 or self.guidance_rms_epsilon <= 0.0:
            raise ValueError("V51 guidance settings are invalid")


@dataclass
class VisualPriorLoss:
    loss: torch.Tensor
    noise: torch.Tensor
    foreground_clean: torch.Tensor
    dice: torch.Tensor
    mean_timestep: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "noise_mse": float(self.noise.detach()),
            "foreground_clean_mse": float(self.foreground_clean.detach()),
            "soft_ink_dice_loss": float(self.dice.detach()),
            "mean_timestep": float(self.mean_timestep.detach()),
        }


def cosine_diffusion_schedule(
    steps: int,
    *,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps < 2 or not 0.0 < offset < 0.1:
        raise ValueError("invalid cosine diffusion schedule")
    times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos(
        (times + offset) / (1.0 + offset) * math.pi / 2.0
    ).square()
    cumulative = cumulative / cumulative[0]
    betas = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0).float()
    return betas.float(), alpha_bars


class VisualGlyphScorePrior(nn.Module):
    """Unconditional writing prior used through explicit visual guidance."""

    def __init__(self, config: VisualPosteriorLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = UNet2D(
            in_ch=1,
            base_ch=config.base_channels,
            depth=config.unet_depth,
            out_ch=1,
            time_dim=config.time_dimension,
        )
        betas, alpha_bars = cosine_diffusion_schedule(
            config.diffusion_steps,
            offset=config.cosine_offset,
        )
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alpha_bars", alpha_bars, persistent=True)
        self.visual_field = OrthonormalGlyphField(size=config.cell_size)

    def _validate_images(self, images: torch.Tensor) -> None:
        if not images.is_floating_point():
            raise TypeError("V51 images must be floating point")
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 32, 32):
            raise ValueError("V51 images must have shape [B,1,32,32]")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("V51 images must be finite")
        if not bool(((images >= 0.0) & (images <= 1.0)).all()):
            raise ValueError("V51 images must lie in [0,1]")

    def _validate_timesteps(self, timesteps: torch.Tensor, count: int) -> None:
        if timesteps.shape != (count,) or timesteps.dtype != torch.long:
            raise ValueError("V51 timesteps must be a long vector")
        if not bool(
            ((timesteps >= 0) & (timesteps < self.config.diffusion_steps)).all()
        ):
            raise ValueError("V51 timestep is outside the schedule")

    def normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / max(1, self.config.diffusion_steps - 1)

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_images(clean_images)
        self._validate_timesteps(timesteps, len(clean_images))
        if noise.shape != clean_images.shape or not bool(torch.isfinite(noise).all()):
            raise ValueError("V51 Gaussian noise must align with clean images")
        signed = clean_images.float().mul(2.0).sub(1.0)
        alpha = self.alpha_bars[timesteps].reshape(-1, 1, 1, 1)
        return alpha.sqrt() * signed + (1.0 - alpha).sqrt() * noise.float()

    def predict_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.ndim != 4 or tuple(noisy.shape[1:]) != (1, 32, 32):
            raise ValueError("V51 noisy images must have shape [B,1,32,32]")
        self._validate_timesteps(timesteps, len(noisy))
        return self.denoiser(noisy, self.normalized_time(timesteps))

    def predict_clean_signed(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if predicted_noise is None:
            predicted_noise = self.predict_noise(noisy, timesteps)
        alpha = self.alpha_bars[timesteps].reshape(-1, 1, 1, 1)
        return (
            (noisy.float() - (1.0 - alpha).sqrt() * predicted_noise.float())
            / alpha.sqrt().clamp_min(1e-6)
        ).clamp(-1.0, 1.0)

    def training_loss(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> VisualPriorLoss:
        noisy = self.add_noise(clean_images, timesteps, noise)
        predicted_noise = self.predict_noise(noisy, timesteps)
        clean_signed = clean_images.float().mul(2.0).sub(1.0)
        predicted_clean = self.predict_clean_signed(
            noisy,
            timesteps,
            predicted_noise,
        )
        noise_loss = F.mse_loss(predicted_noise.float(), noise.float())
        foreground_weight = 1.0 + 2.0 * clean_images.float()
        foreground_clean = (
            (predicted_clean - clean_signed).square() * foreground_weight
        ).sum() / foreground_weight.sum()
        probability = predicted_clean.add(1.0).mul(0.5).clamp(0.0, 1.0)
        target = clean_images.float()
        overlap = 2.0 * (probability * target).flatten(1).sum(dim=-1)
        scale = probability.flatten(1).sum(dim=-1) + target.flatten(1).sum(dim=-1)
        dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
        loss = (
            noise_loss
            + self.config.foreground_clean_weight * foreground_clean
            + self.config.dice_weight * dice
        )
        return VisualPriorLoss(
            loss=loss,
            noise=noise_loss,
            foreground_clean=foreground_clean,
            dice=dice,
            mean_timestep=timesteps.float().mean(),
        )

    def soft_visual_field(self, signed_images: torch.Tensor) -> torch.Tensor:
        if signed_images.ndim != 4 or tuple(signed_images.shape[1:]) != (
            1,
            32,
            32,
        ):
            raise ValueError("V51 soft field requires signed 32x32 images")
        basis = self.visual_field.basis.float().to(signed_images.device)
        spatial = signed_images[:, 0].float()
        spectral = torch.matmul(torch.matmul(basis, spatial), basis.transpose(0, 1))
        return F.normalize(spectral.flatten(1), dim=-1, eps=1e-8)

    def sampling_schedule(
        self,
        *,
        start_step: int | None = None,
        sampling_steps: int | None = None,
    ) -> tuple[int, ...]:
        start = (
            self.config.image_to_image_start if start_step is None else start_step
        )
        count = self.config.sampling_steps if sampling_steps is None else sampling_steps
        if not 1 <= start < self.config.diffusion_steps:
            raise ValueError("V51 sampling start is outside the schedule")
        if not 2 <= count <= start + 1:
            raise ValueError("V51 sampling-step count is invalid")
        values = torch.linspace(start, 0, count).round().to(torch.long).tolist()
        schedule: list[int] = []
        for value in values:
            if not schedule or value != schedule[-1]:
                schedule.append(int(value))
        if schedule[-1] != 0:
            schedule.append(0)
        return tuple(schedule)

    def _guided_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        anchors: torch.Tensor,
        *,
        guidance_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_leaf = noisy.detach().requires_grad_(guidance_strength > 0.0)
        predicted_noise = self.predict_noise(noisy_leaf, timesteps)
        predicted_clean = self.predict_clean_signed(
            noisy_leaf,
            timesteps,
            predicted_noise,
        )
        if guidance_strength <= 0.0:
            return predicted_noise.detach(), predicted_clean.detach()
        similarity = (
            self.soft_visual_field(predicted_clean) * anchors.float()
        ).sum(dim=-1)
        gradient = torch.autograd.grad(similarity.sum(), noisy_leaf)[0]
        rms = gradient.square().flatten(1).mean(dim=-1).sqrt().clamp_min(
            self.config.guidance_rms_epsilon
        )
        normalized_gradient = gradient / rms[:, None, None, None]
        alpha = self.alpha_bars[timesteps].reshape(-1, 1, 1, 1)
        guided_noise = (
            predicted_noise.float()
            - guidance_strength * (1.0 - alpha).sqrt() * normalized_gradient
        )
        guided_clean = self.predict_clean_signed(
            noisy_leaf,
            timesteps,
            guided_noise,
        )
        return guided_noise.detach(), guided_clean.detach()

    def posterior_sample(
        self,
        plan_images: torch.Tensor,
        anchors: torch.Tensor,
        *,
        generator: torch.Generator,
        start_step: int | None = None,
        sampling_steps: int | None = None,
        guidance_strength: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_images(plan_images)
        if anchors.shape != (len(plan_images), 1024):
            raise ValueError("V51 visual anchors must have shape [B,1024]")
        if not bool(torch.isfinite(anchors).all()):
            raise ValueError("V51 visual anchors must be finite")
        strength = (
            self.config.guidance_strength
            if guidance_strength is None
            else guidance_strength
        )
        if strength < 0.0:
            raise ValueError("V51 guidance strength must be non-negative")
        schedule = self.sampling_schedule(
            start_step=start_step,
            sampling_steps=sampling_steps,
        )
        start = schedule[0]
        noise = torch.randn(
            plan_images.shape,
            device=plan_images.device,
            dtype=plan_images.dtype,
            generator=generator,
        )
        start_times = torch.full(
            (len(plan_images),),
            start,
            device=plan_images.device,
            dtype=torch.long,
        )
        noisy = self.add_noise(plan_images, start_times, noise)
        final_clean = plan_images.float().mul(2.0).sub(1.0)
        for index, step in enumerate(schedule[:-1]):
            previous = schedule[index + 1]
            times = torch.full(
                (len(plan_images),),
                step,
                device=plan_images.device,
                dtype=torch.long,
            )
            with torch.enable_grad():
                predicted_noise, final_clean = self._guided_noise(
                    noisy,
                    times,
                    F.normalize(anchors.float(), dim=-1),
                    guidance_strength=strength,
                )
            previous_alpha = (
                self.alpha_bars[previous]
                if previous > 0
                else noisy.new_tensor(1.0)
            )
            noisy = (
                previous_alpha.sqrt() * final_clean
                + (1.0 - previous_alpha).sqrt() * predicted_noise
            ).detach()
        binary = (final_clean >= 0.0).to(plan_images.dtype)
        return binary, {
            "continuous": final_clean,
            "visual_fields": self.soft_visual_field(final_clean),
            "start_step": torch.tensor(start, device=plan_images.device),
            "sampling_steps": torch.tensor(
                len(schedule), device=plan_images.device
            ),
        }


def visual_posterior_language_config_payload(
    config: VisualPosteriorLanguageConfig,
) -> dict[str, Any]:
    return asdict(config)


def visual_posterior_language_config_from_payload(
    payload: Mapping[str, Any],
) -> VisualPosteriorLanguageConfig:
    return VisualPosteriorLanguageConfig(**dict(payload))


def visual_posterior_language_boundary_receipt(
    model: VisualGlyphScorePrior,
) -> dict[str, Any]:
    return {
        "architecture": V51_ARCHITECTURE,
        "primary_input": "continuous or binary visible writing raster",
        "primary_output": "posterior-guided visible writing raster",
        "condition": "continuous image-derived causal visual field",
        "text_tokenizer": False,
        "unicode_or_character_ids_in_model": False,
        "ocr_in_model": False,
        "glyph_vocabulary_or_candidate_bank_in_model": False,
        "vector_quantizer_or_codebook": False,
        "external_model_at_runtime": False,
        "prior_receives_sequence_or_character_labels": False,
        "total_prior_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "config": visual_posterior_language_config_payload(model.config),
    }

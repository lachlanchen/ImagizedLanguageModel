from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.diffusion.unet2d import ResBlock, SinusoidalTimeEmbed

from .hierarchical_raster_language import CausalRasterReader
from .visual_posterior_language_v51 import VisualGlyphScorePrior


V52_ARCHITECTURE = "noisy-visual-energy-guidance-v52"


@dataclass(frozen=True)
class NoisyVisualEnergyConfig:
    cell_size: int = 32
    context_dimension: int = 384
    anchor_dimension: int = 1024
    energy_dimension: int = 256
    time_dimension: int = 128
    channels: tuple[int, int, int, int] = (32, 64, 128, 192)
    initial_temperature: float = 0.10
    diffusion_steps: int = 100
    sampling_steps: int = 25
    guidance_strength: float = 1.0
    samples_per_context: int = 4

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V52 requires 32x32 visible writing cells")
        if self.context_dimension != 384 or self.anchor_dimension != 1024:
            raise ValueError("V52 requires the frozen V42 state geometry")
        if self.energy_dimension < 64:
            raise ValueError("V52 energy dimension is underspecified")
        if self.time_dimension < 32 or self.time_dimension % 2:
            raise ValueError("V52 time dimension must be positive and even")
        if len(self.channels) != 4 or any(channel % 8 for channel in self.channels):
            raise ValueError("V52 image channels must contain four multiples of 8")
        if not all(
            following > previous
            for previous, following in zip(self.channels, self.channels[1:])
        ):
            raise ValueError("V52 image channels must increase by scale")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V52 initial temperature is invalid")
        if self.diffusion_steps != 100:
            raise ValueError("V52 is tied to the 100-level V51 prior")
        if not 2 <= self.sampling_steps <= self.diffusion_steps:
            raise ValueError("V52 sampling-step count is invalid")
        if self.guidance_strength <= 0.0 or self.samples_per_context < 1:
            raise ValueError("V52 guidance settings are invalid")


class NoisyGlyphImageEncoder(nn.Module):
    """Encode one noisy glyph while preserving gradients to its pixels."""

    def __init__(self, config: NoisyVisualEnergyConfig) -> None:
        super().__init__()
        self.config = config
        self.time = SinusoidalTimeEmbed(config.time_dimension)
        channels = config.channels
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        for index, channel in enumerate(channels):
            entry = nn.ModuleDict(
                {"residual": ResBlock(channel, config.time_dimension)}
            )
            if index + 1 < len(channels):
                entry["down"] = nn.Conv2d(
                    channel,
                    channels[index + 1],
                    4,
                    stride=2,
                    padding=1,
                )
            self.blocks.append(entry)
        self.output_norm = nn.GroupNorm(8, channels[-1])
        self.output = nn.Linear(channels[-1] * 4 * 4, config.energy_dimension)

    def forward(
        self,
        noisy_images: torch.Tensor,
        normalized_time: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_images.ndim != 4 or tuple(noisy_images.shape[1:]) != (
            1,
            32,
            32,
        ):
            raise ValueError("V52 noisy images must have shape [N,1,32,32]")
        if normalized_time.shape != (len(noisy_images),):
            raise ValueError("V52 image time must align with noisy images")
        if not bool(torch.isfinite(noisy_images).all()):
            raise ValueError("V52 noisy images must be finite")
        time = self.time(normalized_time)
        hidden = self.stem(noisy_images)
        for block in self.blocks:
            hidden = block["residual"](hidden, time)
            if "down" in block:
                hidden = block["down"](hidden)
        hidden = F.silu(self.output_norm(hidden)).flatten(1)
        return F.normalize(self.output(hidden).float(), dim=-1, eps=1e-8)


class NoisyVisualEnergyCritic(nn.Module):
    """A continuous image-history/image density-ratio critic."""

    def __init__(self, config: NoisyVisualEnergyConfig) -> None:
        super().__init__()
        self.config = config
        condition_dimension = config.context_dimension + config.anchor_dimension
        self.context = nn.Sequential(
            nn.LayerNorm(condition_dimension),
            nn.Linear(condition_dimension, 512),
            nn.SiLU(),
            nn.Linear(512, config.energy_dimension),
        )
        self.context_time = SinusoidalTimeEmbed(config.time_dimension)
        self.context_time_projection = nn.Linear(
            config.time_dimension,
            config.energy_dimension,
            bias=False,
        )
        self.image = NoisyGlyphImageEncoder(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )
        self._initialize()

    def _initialize(self) -> None:
        for root in (self.context, self.context_time_projection):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.dtype != torch.long:
            raise ValueError("V52 timesteps must be a long vector")
        if not bool(
            ((timesteps >= 0) & (timesteps < self.config.diffusion_steps)).all()
        ):
            raise ValueError("V52 timestep is outside the schedule")
        return timesteps.float() / max(1, self.config.diffusion_steps - 1)

    def context_queries(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape != (len(hidden), self.config.context_dimension):
            raise ValueError("V52 hidden states must be [N,384]")
        if anchors.shape != (len(hidden), self.config.anchor_dimension):
            raise ValueError("V52 anchors must be [N,1024]")
        if timesteps.shape != (len(hidden),):
            raise ValueError("V52 context time must align with hidden states")
        normalized = self.normalized_time(timesteps)
        condition = torch.cat((hidden.float(), anchors.float()), dim=-1)
        query = self.context(condition.to(self.context[1].weight.dtype)).float()
        time = self.context_time(normalized)
        query = query + self.context_time_projection(time).float()
        return F.normalize(query, dim=-1, eps=1e-8)

    def image_keys(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.shape != (len(noisy_images),):
            raise ValueError("V52 image time must align with images")
        return self.image(noisy_images, self.normalized_time(timesteps))

    def score_matrix(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if len(hidden) != len(noisy_images):
            raise ValueError("V52 square score matrix requires matching batches")
        query = self.context_queries(hidden, anchors, timesteps)
        key = self.image_keys(noisy_images, timesteps)
        return self.contrastive_scale.float() * query @ key.transpose(0, 1)

    def score_pairs(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if not len(hidden) == len(noisy_images) == len(timesteps):
            raise ValueError("V52 paired scores require aligned batches")
        query = self.context_queries(hidden, anchors, timesteps)
        key = self.image_keys(noisy_images, timesteps)
        return self.contrastive_scale.float() * (query * key).sum(dim=-1)


class GuidedVisualLanguageSampler(nn.Module):
    """Compose the frozen reader, marginal prior, and learned visual energy."""

    def __init__(
        self,
        reader: CausalRasterReader,
        prior: VisualGlyphScorePrior,
        critic: NoisyVisualEnergyCritic,
    ) -> None:
        super().__init__()
        if prior.config.diffusion_steps != critic.config.diffusion_steps:
            raise ValueError("V52 critic and V51 prior schedules differ")
        self.reader = reader.requires_grad_(False).eval()
        self.prior = prior.requires_grad_(False).eval()
        self.critic = critic

    def train(self, mode: bool = True) -> GuidedVisualLanguageSampler:
        super().train(mode)
        self.reader.eval()
        self.prior.eval()
        return self

    def sampling_schedule(self) -> tuple[int, ...]:
        values = (
            torch.linspace(
                self.critic.config.diffusion_steps - 1,
                0,
                self.critic.config.sampling_steps,
            )
            .round()
            .to(torch.long)
            .tolist()
        )
        schedule: list[int] = []
        for value in values:
            if not schedule or int(value) != schedule[-1]:
                schedule.append(int(value))
        if schedule[-1] != 0:
            schedule.append(0)
        return tuple(schedule)

    def _guided_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        *,
        guidance_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_leaf = noisy.detach().requires_grad_(guidance_strength > 0.0)
        predicted_noise = self.prior.predict_noise(noisy_leaf, timesteps)
        if guidance_strength <= 0.0:
            predicted_clean = self.prior.predict_clean_signed(
                noisy_leaf,
                timesteps,
                predicted_noise,
            )
            zero = torch.zeros(len(noisy_leaf), device=noisy_leaf.device)
            return predicted_noise.detach(), predicted_clean.detach(), zero
        score = self.critic.score_pairs(
            hidden,
            anchors,
            noisy_leaf,
            timesteps,
        )
        gradient = torch.autograd.grad(score.sum(), noisy_leaf)[0]
        alpha = self.prior.alpha_bars[timesteps].reshape(-1, 1, 1, 1)
        guided_noise = predicted_noise.float() - (
            guidance_strength * (1.0 - alpha).sqrt() * gradient.float()
        )
        predicted_clean = self.prior.predict_clean_signed(
            noisy_leaf,
            timesteps,
            guided_noise,
        )
        gradient_rms = gradient.float().square().flatten(1).mean(dim=-1).sqrt()
        return guided_noise.detach(), predicted_clean.detach(), gradient_rms.detach()

    def sample_from_state(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        *,
        generator: torch.Generator,
        samples: int | None = None,
        guidance_strength: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 2 or hidden.shape[1] != 384:
            raise ValueError("V52 sampling hidden states must be [B,384]")
        if anchors.shape != (len(hidden), 1024):
            raise ValueError("V52 sampling anchors must be [B,1024]")
        count = (
            self.critic.config.samples_per_context if samples is None else samples
        )
        strength = (
            self.critic.config.guidance_strength
            if guidance_strength is None
            else guidance_strength
        )
        if count < 1 or strength < 0.0:
            raise ValueError("V52 sampling settings are invalid")
        repeated_hidden = hidden[:, None].expand(-1, count, -1).flatten(0, 1)
        repeated_anchor = anchors[:, None].expand(-1, count, -1).flatten(0, 1)
        noisy = torch.randn(
            (len(repeated_hidden), 1, 32, 32),
            device=hidden.device,
            dtype=hidden.dtype,
            generator=generator,
        )
        schedule = self.sampling_schedule()
        final_clean = noisy
        gradient_rms: list[torch.Tensor] = []
        for index, step in enumerate(schedule[:-1]):
            previous = schedule[index + 1]
            timesteps = torch.full(
                (len(noisy),),
                step,
                device=noisy.device,
                dtype=torch.long,
            )
            with torch.enable_grad():
                predicted_noise, final_clean, rms = self._guided_noise(
                    noisy,
                    timesteps,
                    repeated_hidden,
                    repeated_anchor,
                    guidance_strength=strength,
                )
            gradient_rms.append(rms)
            previous_alpha = (
                self.prior.alpha_bars[previous]
                if previous > 0
                else noisy.new_tensor(1.0)
            )
            noisy = (
                previous_alpha.sqrt() * final_clean
                + (1.0 - previous_alpha).sqrt() * predicted_noise
            ).detach()
        binary = (final_clean >= 0.0).to(hidden.dtype).reshape(
            len(hidden), count, 1, 32, 32
        )
        clean_times = torch.zeros(
            len(repeated_hidden), device=hidden.device, dtype=torch.long
        )
        clean_signed = binary.flatten(0, 1).float().mul(2.0).sub(1.0)
        clean_scores = self.critic.score_pairs(
            repeated_hidden,
            repeated_anchor,
            clean_signed,
            clean_times,
        ).reshape(len(hidden), count)
        selected = clean_scores.argmax(dim=-1)
        rows = torch.arange(len(hidden), device=hidden.device)
        selected_images = binary[rows, selected]
        rms_trace = (
            torch.stack(gradient_rms, dim=1).reshape(len(hidden), count, -1)
            if gradient_rms
            else hidden.new_zeros((len(hidden), count, 0))
        )
        return selected_images, {
            "samples": binary,
            "first_samples": binary[:, 0],
            "selection_scores": clean_scores,
            "selected_indices": selected,
            "gradient_rms": rms_trace,
            "schedule": torch.tensor(schedule, device=hidden.device),
        }

    def sample_next(
        self,
        context: torch.Tensor,
        *,
        generator: torch.Generator,
        samples: int | None = None,
        guidance_strength: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            language = self.reader.language(context)
            hidden = language["hidden_states"][:, -1].float()
            anchors = language["anchor_fields"][:, -1].float()
        return self.sample_from_state(
            hidden,
            anchors,
            generator=generator,
            samples=samples,
            guidance_strength=guidance_strength,
        )


def noisy_visual_energy_config_payload(
    config: NoisyVisualEnergyConfig,
) -> dict[str, Any]:
    return asdict(config)


def noisy_visual_energy_config_from_payload(
    payload: Mapping[str, Any],
) -> NoisyVisualEnergyConfig:
    values = dict(payload)
    if "channels" in values:
        values["channels"] = tuple(values["channels"])
    return NoisyVisualEnergyConfig(**values)


def noisy_visual_energy_boundary_receipt(
    critic: NoisyVisualEnergyCritic,
) -> dict[str, Any]:
    prohibited = (
        "token",
        "vocab",
        "unicode",
        "character_id",
        "ocr",
        "codebook",
        "quantizer",
        "candidate_bank",
    )
    suspicious = sorted(
        name
        for name, _ in critic.named_parameters()
        if any(fragment in name.lower() for fragment in prohibited)
    )
    return {
        "architecture": V52_ARCHITECTURE,
        "config": noisy_visual_energy_config_payload(critic.config),
        "critic_parameters": sum(
            parameter.numel() for parameter in critic.parameters()
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "score_matrix_parameters": list(
            inspect.signature(critic.score_matrix).parameters
        ),
        "score_pairs_parameters": list(inspect.signature(critic.score_pairs).parameters),
        "primary_input": "ordered floating writing rasters",
        "primary_output": "guided visible writing raster",
        "text_tokenizer": False,
        "unicode_or_character_ids_in_model": False,
        "ocr_in_model": False,
        "glyph_vocabulary_or_candidate_bank_in_model": False,
        "vector_quantizer_or_codebook": False,
        "external_model_at_runtime": False,
        "critic_receives_strings_or_labels": False,
    }


__all__ = [
    "GuidedVisualLanguageSampler",
    "NoisyGlyphImageEncoder",
    "NoisyVisualEnergyConfig",
    "NoisyVisualEnergyCritic",
    "V52_ARCHITECTURE",
    "noisy_visual_energy_boundary_receipt",
    "noisy_visual_energy_config_from_payload",
    "noisy_visual_energy_config_payload",
]

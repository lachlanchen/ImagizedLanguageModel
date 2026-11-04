from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.diffusion.unet2d import ResBlock, SinusoidalTimeEmbed

from .hierarchical_raster_language import CausalRasterReader
from .visual_posterior_language_v51 import (
    VisualGlyphScorePrior,
    cosine_diffusion_schedule,
)


V53_ARCHITECTURE = "causal-visual-block-diffusion-v53"


@dataclass(frozen=True)
class CausalVisualBlockDiffusionConfig:
    cell_size: int = 32
    context_cells: int = 60
    future_cells: int = 4
    context_dimension: int = 384
    anchor_dimension: int = 1024
    base_channels: int = 64
    time_dimension: int = 128
    diffusion_steps: int = 100
    sampling_steps: int = 32
    classifier_free_scale: float = 2.0
    condition_dropout: float = 0.10
    cosine_offset: float = 0.008

    def __post_init__(self) -> None:
        if (self.cell_size, self.context_cells, self.future_cells) != (32, 60, 4):
            raise ValueError("V53 requires 60 visible and four future 32px cells")
        if (self.context_dimension, self.anchor_dimension) != (384, 1024):
            raise ValueError("V53 requires the frozen V42 state geometry")
        if self.base_channels != 64 or self.time_dimension != 128:
            raise ValueError("V53 initialization requires the V51 U-Net geometry")
        if self.diffusion_steps != 100 or not 2 <= self.sampling_steps <= 100:
            raise ValueError("V53 diffusion geometry is invalid")
        if self.classifier_free_scale < 1.0:
            raise ValueError("V53 classifier-free scale must be at least one")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("V53 condition dropout must lie in [0,1)")


class FourCellMixer(nn.Module):
    """Exchange information among four pooled future-cell representations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads=4,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        self.output_scale = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[-1] % 4:
            raise ValueError("V53 bottleneck must divide into four spatial cells")
        batch, channels, height, width = features.shape
        cell_width = width // 4
        cells = features.reshape(batch, channels, height, 4, cell_width)
        pooled = cells.mean(dim=(2, 4)).transpose(1, 2)
        normalized = self.norm1(pooled)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        mixed = pooled + attended
        mixed = mixed + self.mlp(self.norm2(mixed))
        delta = (mixed - pooled).transpose(1, 2)[:, :, None, :, None]
        delta = delta.expand(-1, -1, height, -1, cell_width).reshape_as(features)
        return features + self.output_scale * delta, mixed


class ConditionalStripDenoiser(nn.Module):
    """Predict velocity for one jointly generated four-glyph strip."""

    def __init__(self, config: CausalVisualBlockDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        self.time_embed = SinusoidalTimeEmbed(config.time_dimension)
        self.condition = nn.Sequential(
            nn.LayerNorm(config.context_dimension + config.anchor_dimension),
            nn.Linear(
                config.context_dimension + config.anchor_dimension,
                config.time_dimension * 2,
            ),
            nn.SiLU(),
            nn.Linear(config.time_dimension * 2, config.time_dimension),
        )
        self.in_conv = nn.Conv2d(4, base, 3, padding=1)
        self.downs = nn.ModuleList(
            (
                nn.ModuleDict(
                    {
                        "res": ResBlock(base, config.time_dimension),
                        "down": nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
                    }
                ),
                nn.ModuleDict(
                    {
                        "res": ResBlock(base * 2, config.time_dimension),
                        "down": nn.Conv2d(
                            base * 2, base * 4, 4, stride=2, padding=1
                        ),
                    }
                ),
            )
        )
        self.mid1 = ResBlock(base * 4, config.time_dimension)
        self.mixer = FourCellMixer(base * 4)
        self.mid2 = ResBlock(base * 4, config.time_dimension)
        self.representation = nn.Linear(base * 4, config.anchor_dimension)
        self.ups = nn.ModuleList(
            (
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(
                            base * 4, base * 2, 4, stride=2, padding=1
                        ),
                        "res": ResBlock(base * 2, config.time_dimension),
                    }
                ),
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(
                            base * 2, base, 4, stride=2, padding=1
                        ),
                        "res": ResBlock(base, config.time_dimension),
                    }
                ),
            )
        )
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)
        self._initialize_new_paths()

    def _initialize_new_paths(self) -> None:
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        nn.init.zeros_(self.mid2.conv2.weight)
        nn.init.zeros_(self.mid2.conv2.bias)
        nn.init.normal_(self.representation.weight, std=0.02)
        nn.init.zeros_(self.representation.bias)

    @staticmethod
    def cells_to_strip(cells: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 5 or tuple(cells.shape[1:]) != (4, 1, 32, 32):
            raise ValueError("V53 cells must have shape [B,4,1,32,32]")
        return cells[:, :, 0].permute(0, 2, 1, 3).reshape(len(cells), 1, 32, 128)

    @staticmethod
    def strip_to_cells(strip: torch.Tensor) -> torch.Tensor:
        if strip.ndim != 4 or tuple(strip.shape[1:]) != (1, 32, 128):
            raise ValueError("V53 strip must have shape [B,1,32,128]")
        return strip[:, 0].reshape(len(strip), 32, 4, 32).permute(0, 2, 1, 3)[
            :, :, None
        ]

    def _spatial_inputs(
        self,
        noisy_cells: torch.Tensor,
        first_plan: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        strip = self.cells_to_strip(noisy_cells)
        if first_plan.shape != (len(strip), 1, 32, 32):
            raise ValueError("V53 first-cell plan must be [B,1,32,32]")
        if timesteps.shape != (len(strip), 4) or timesteps.dtype != torch.long:
            raise ValueError("V53 timesteps must have shape [B,4]")
        plan_strip = strip.new_zeros(strip.shape)
        plan_strip[:, :, :, :32] = first_plan
        coordinate = torch.linspace(-1.0, 1.0, 128, device=strip.device).reshape(
            1, 1, 1, 128
        )
        coordinate = coordinate.expand(len(strip), -1, 32, -1)
        time_map = (
            timesteps.float()
            .div(self.config.diffusion_steps - 1)
            .reshape(len(strip), 1, 1, 4, 1)
            .expand(-1, -1, 32, -1, 32)
            .reshape(len(strip), 1, 32, 128)
        )
        return torch.cat((strip, plan_strip, coordinate, time_map), dim=1)

    def forward(
        self,
        noisy_cells: torch.Tensor,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        first_plan: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape != (len(noisy_cells), self.config.context_dimension):
            raise ValueError("V53 hidden state must be [B,384]")
        if anchors.shape != (len(noisy_cells), self.config.anchor_dimension):
            raise ValueError("V53 visual anchor must be [B,1024]")
        inputs = self._spatial_inputs(noisy_cells, first_plan, timesteps)
        mean_time = timesteps.float().mean(dim=1) / (self.config.diffusion_steps - 1)
        condition = torch.cat((hidden.float(), anchors.float()), dim=-1)
        modulation = self.time_embed(mean_time) + self.condition(condition)
        features = self.in_conv(inputs)
        skips: list[torch.Tensor] = []
        for block in self.downs:
            features = block["res"](features, modulation)
            skips.append(features)
            features = block["down"](features)
        features = self.mid1(features, modulation)
        features, pooled = self.mixer(features)
        features = self.mid2(features, modulation)
        representation = F.normalize(
            self.representation(pooled).float(), dim=-1, eps=1e-8
        )
        for block in self.ups:
            features = block["up"](features)
            skip = skips.pop()
            features = features + skip
            features = block["res"](features, modulation)
        velocity = self.strip_to_cells(self.out_conv(features))
        return velocity, representation

    @torch.no_grad()
    def initialize_from_prior(self, prior: VisualGlyphScorePrior) -> None:
        source = prior.denoiser
        if not (
            prior.config.base_channels == self.config.base_channels
            and prior.config.unet_depth == 2
            and prior.config.time_dimension == self.config.time_dimension
        ):
            raise ValueError("V53 prior initialization geometry differs")
        self.time_embed.load_state_dict(source.time_embed.state_dict(), strict=True)
        self.in_conv.weight.zero_()
        self.in_conv.weight[:, :1].copy_(source.in_conv.weight)
        self.in_conv.bias.copy_(source.in_conv.bias)
        self.downs.load_state_dict(source.downs.state_dict(), strict=True)
        self.mid1.load_state_dict(source.mid.state_dict(), strict=True)
        self.ups.load_state_dict(source.ups.state_dict(), strict=True)
        self.out_conv.load_state_dict(source.out_conv.state_dict(), strict=True)


class CausalVisualBlockDiffusion(nn.Module):
    """Frozen raster reader plus a directly conditional visible-block writer."""

    def __init__(
        self,
        reader: CausalRasterReader,
        denoiser: ConditionalStripDenoiser,
    ) -> None:
        super().__init__()
        self.reader = reader.requires_grad_(False).eval()
        self.denoiser = denoiser
        betas, alpha_bars = cosine_diffusion_schedule(
            denoiser.config.diffusion_steps,
            offset=denoiser.config.cosine_offset,
        )
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alpha_bars", alpha_bars, persistent=True)

    @property
    def config(self) -> CausalVisualBlockDiffusionConfig:
        return self.denoiser.config

    def train(self, mode: bool = True) -> CausalVisualBlockDiffusion:
        super().train(mode)
        self.reader.eval()
        return self

    @torch.no_grad()
    def read_condition(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 5 or tuple(context.shape[1:]) != (60, 1, 32, 32):
            raise ValueError("V53 context must have shape [B,60,1,32,32]")
        language = self.reader.language(context)
        hidden = language["hidden_states"][:, -1].float()
        anchors = language["anchor_fields"][:, -1].float()
        plan = self.reader.field.binary(anchors)
        return hidden, anchors, plan

    def add_noise(
        self,
        clean_cells: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if clean_cells.shape != noise.shape or timesteps.shape != clean_cells.shape[:2]:
            raise ValueError("V53 corruption tensors do not align")
        signed = clean_cells.float().mul(2.0).sub(1.0)
        alpha = self.alpha_bars[timesteps].reshape(*timesteps.shape, 1, 1, 1)
        return alpha.sqrt() * signed + (1.0 - alpha).sqrt() * noise.float()

    def velocity_target(
        self,
        clean_cells: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        signed = clean_cells.float().mul(2.0).sub(1.0)
        alpha = self.alpha_bars[timesteps].reshape(*timesteps.shape, 1, 1, 1)
        return alpha.sqrt() * noise.float() - (1.0 - alpha).sqrt() * signed

    def velocity_to_clean_noise(
        self,
        noisy: torch.Tensor,
        velocity: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alpha_bars[timesteps].reshape(*timesteps.shape, 1, 1, 1)
        clean = alpha.sqrt() * noisy.float() - (1.0 - alpha).sqrt() * velocity.float()
        noise = (1.0 - alpha).sqrt() * noisy.float() + alpha.sqrt() * velocity.float()
        return clean.clamp(-1.0, 1.0), noise

    def sampling_schedule(self) -> tuple[int, ...]:
        values = (
            torch.linspace(
                self.config.diffusion_steps - 1,
                0,
                self.config.sampling_steps,
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

    @torch.no_grad()
    def sample_next_block(
        self,
        context: torch.Tensor,
        *,
        generator: torch.Generator,
        conditioned: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden, anchors, plan = self.read_condition(context)
        zeros_hidden = torch.zeros_like(hidden)
        zeros_anchor = torch.zeros_like(anchors)
        zeros_plan = torch.zeros_like(plan)
        noisy = torch.randn(
            (len(context), 4, 1, 32, 32),
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        schedule = self.sampling_schedule()
        clean = noisy
        condition_displacement: list[torch.Tensor] = []
        for index, step in enumerate(schedule[:-1]):
            previous = schedule[index + 1]
            times = torch.full(
                (len(context), 4), step, device=context.device, dtype=torch.long
            )
            unconditioned_velocity, _ = self.denoiser(
                noisy,
                zeros_hidden,
                zeros_anchor,
                zeros_plan,
                times,
            )
            if conditioned:
                conditional_velocity, _ = self.denoiser(
                    noisy,
                    hidden,
                    anchors,
                    plan,
                    times,
                )
                velocity = unconditioned_velocity + self.config.classifier_free_scale * (
                    conditional_velocity - unconditioned_velocity
                )
                condition_displacement.append(
                    (conditional_velocity - unconditioned_velocity)
                    .square()
                    .flatten(2)
                    .mean(dim=-1)
                    .sqrt()
                )
            else:
                velocity = unconditioned_velocity
                condition_displacement.append(
                    torch.zeros(
                        len(context), 4, device=context.device, dtype=torch.float32
                    )
                )
            clean, predicted_noise = self.velocity_to_clean_noise(
                noisy, velocity, times
            )
            previous_alpha = (
                self.alpha_bars[previous]
                if previous > 0
                else noisy.new_tensor(1.0)
            )
            noisy = (
                previous_alpha.sqrt() * clean
                + (1.0 - previous_alpha).sqrt() * predicted_noise
            )
        binary = (clean >= 0.0).to(context.dtype)
        return binary, {
            "continuous": clean,
            "condition_displacement_rms": torch.stack(
                condition_displacement, dim=-1
            ),
            "schedule": torch.tensor(schedule, device=context.device),
        }


def causal_visual_block_config_payload(
    config: CausalVisualBlockDiffusionConfig,
) -> dict[str, Any]:
    return asdict(config)


def causal_visual_block_config_from_payload(
    payload: Mapping[str, Any],
) -> CausalVisualBlockDiffusionConfig:
    return CausalVisualBlockDiffusionConfig(**dict(payload))


def causal_visual_block_boundary_receipt(
    model: CausalVisualBlockDiffusion,
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
        for name, _ in model.denoiser.named_parameters()
        if any(fragment in name.lower() for fragment in prohibited)
    )
    denoiser_parameters = sum(
        parameter.numel() for parameter in model.denoiser.parameters()
    )
    reader_parameters = sum(parameter.numel() for parameter in model.reader.parameters())
    return {
        "architecture": V53_ARCHITECTURE,
        "config": causal_visual_block_config_payload(model.config),
        "denoiser_parameters": denoiser_parameters,
        "reader_parameters": reader_parameters,
        "combined_parameters": denoiser_parameters + reader_parameters,
        "parameter_names_with_forbidden_fragments": suspicious,
        "sample_parameters": list(
            inspect.signature(model.sample_next_block).parameters
        ),
        "primary_input": "ordered floating writing rasters",
        "primary_output": "four visible writing rasters",
        "text_tokenizer": False,
        "unicode_or_character_ids_in_model": False,
        "ocr_in_model": False,
        "glyph_vocabulary_or_candidate_bank_in_model": False,
        "vector_quantizer_or_codebook": False,
        "external_model_at_runtime": False,
    }


__all__ = [
    "CausalVisualBlockDiffusion",
    "CausalVisualBlockDiffusionConfig",
    "ConditionalStripDenoiser",
    "FourCellMixer",
    "V53_ARCHITECTURE",
    "causal_visual_block_boundary_receipt",
    "causal_visual_block_config_from_payload",
    "causal_visual_block_config_payload",
]

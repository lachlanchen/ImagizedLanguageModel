from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise_limited_retinal_field_v45 import (
    NoiseLimitedRetinalFieldV45,
    noise_limited_retinal_field_v45_state_sha256,
)
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V46_ARCHITECTURE = "scaled-retinal-glyph-language-v46"
V46_PROTOCOL = "references/scaled_retinal_glyph_language_v46_protocol.md"
V46_REFERENCE_RADIUS = 19.622622215774165
V46_REQUIRED_V45_CHECKPOINT_SHA256 = (
    "0e5947d85a8baeff99d92996ee8434d3aceab39e64042c9b6ec1a142aa174534"
)
V46_REQUIRED_V45_FIELD_STATE_SHA256 = (
    "08b57734ac3ded0c1438cc4bf963d92357ce1f1d31ae49ee6548c56c19db019d"
)
V46_REQUIRED_V45_FIT_SEQUENCE_SHA256 = (
    "eb0a4ede44a062eb4107fa857c0cda488734b9ddf5ad56e30a5f375516e61fd8"
)


@dataclass(frozen=True)
class ScaledRetinalGlyphLanguageV46Config:
    cell_size: int = 32
    maximum_cells: int = 64
    field_dim: int = 1024
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    noise_dim: int = 128
    generator_layers: int = 4
    generator_residual_scale: float = 1.0
    binary_threshold: float = 0.5
    decoder_sharpness: float = 6.0
    initial_temperature: float = 0.07
    radius_epsilon: float = 1e-8
    reference_radius: float = V46_REFERENCE_RADIUS
    anchor_output_std: float = 0.008
    generator_output_std: float = 0.0005

    def __post_init__(self) -> None:
        if self.cell_size != 32 or self.field_dim != self.cell_size**2:
            raise ValueError("V46 requires a full 32x32 continuous glyph field")
        if self.maximum_cells != 64:
            raise ValueError("V46 fixes a 64-cell visual context")
        if self.model_dim < 128 or self.layers < 1:
            raise ValueError("V46 causal model is underspecified")
        if self.heads < 1 or self.model_dim % self.heads:
            raise ValueError("V46 model width must divide into attention heads")
        if self.mlp_ratio < 2.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("V46 causal block settings are invalid")
        if self.noise_dim < 16 or self.generator_layers < 1:
            raise ValueError("V46 field sampler is underspecified")
        if self.generator_residual_scale <= 0.0:
            raise ValueError("V46 generator residual scale must be positive")
        if not 0.0 < self.binary_threshold < 1.0:
            raise ValueError("V46 binary threshold must lie in (0,1)")
        if self.decoder_sharpness <= 0.0:
            raise ValueError("V46 decoder sharpness must be positive")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V46 initial temperature is invalid")
        if not 0.0 < self.radius_epsilon < 1e-4:
            raise ValueError("V46 radius epsilon is invalid")
        if not math.isclose(
            self.reference_radius,
            V46_REFERENCE_RADIUS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V46 reference radius differs from the frozen protocol")
        if not math.isclose(self.anchor_output_std, 0.008, abs_tol=1e-12):
            raise ValueError("V46 anchor initialization differs from the protocol")
        if not math.isclose(self.generator_output_std, 0.0005, abs_tol=1e-12):
            raise ValueError("V46 generator initialization differs from the protocol")


class ScaledNoiseLimitedRetinalFieldV46(nn.Module):
    """Exact V45 field carried as one RMS-scaled 1,024-vector."""

    def __init__(
        self,
        retinal: NoiseLimitedRetinalFieldV45,
        *,
        reference_radius: float = V46_REFERENCE_RADIUS,
    ) -> None:
        super().__init__()
        if retinal.config.size != 32 or retinal.config.field_dim != 1024:
            raise ValueError("V46 requires a 32x32 V45 retinal field")
        if not math.isclose(
            reference_radius,
            V46_REFERENCE_RADIUS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V46 field scale differs from the frozen protocol")
        self.retinal = retinal
        self.register_buffer(
            "reference_radius",
            torch.tensor(reference_radius, dtype=torch.float64),
            persistent=True,
        )
        self.requires_grad_(False)

    @property
    def field_dim(self) -> int:
        return 1024

    @property
    def binary_threshold(self) -> float:
        return self.retinal.config.binary_threshold

    def _validate_fields(self, fields: torch.Tensor) -> None:
        if not fields.is_floating_point():
            raise TypeError("V46 fields must be floating point")
        if fields.ndim < 2 or fields.shape[-1] != self.field_dim:
            raise ValueError("V46 fields must end in 1024 coefficients")
        if not bool(torch.isfinite(fields).all()):
            raise ValueError("V46 fields must be finite")

    def encode(self, pixels: torch.Tensor, *, exact: bool = False) -> torch.Tensor:
        transformed = self.retinal.encode_pixels(pixels, exact=exact).field
        scale = self.reference_radius.to(
            device=transformed.device,
            dtype=transformed.dtype,
        )
        return transformed / scale

    def directions(self, fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(fields)
        work = fields.float()
        return work / work.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def decode_dct(self, fields: torch.Tensor, *, exact: bool = False) -> torch.Tensor:
        self._validate_fields(fields)
        if exact:
            work = fields.to(torch.float64) * self.reference_radius.to(
                device=fields.device,
                dtype=torch.float64,
            )
            vectors = self.retinal.eigenvectors.to(
                device=fields.device,
                dtype=torch.float64,
            )
            values = self.retinal.eigenvalues.to(
                device=fields.device,
                dtype=torch.float64,
            )
            mean = self.retinal.mean_dct.to(
                device=fields.device,
                dtype=torch.float64,
            )
            mean_variance = values.clamp_min(0.0).sum() / self.field_dim
            regularized = (
                values.clamp_min(0.0)
                + self.retinal.config.ridge_ratio * mean_variance
            )
            inverse_scale = regularized.pow(self.retinal.config.whitening_power)
            return (work @ vectors * inverse_scale) @ vectors.transpose(0, 1) + mean
        work = fields.float() * self.reference_radius.to(
            device=fields.device,
            dtype=torch.float32,
        )
        inverse = self.retinal.inverse_matrix.to(
            device=fields.device,
            dtype=torch.float32,
        )
        mean = self.retinal.mean_dct.to(
            device=fields.device,
            dtype=torch.float32,
        )
        return work @ inverse + mean

    def signed_spatial(self, fields: torch.Tensor, *, exact: bool = False) -> torch.Tensor:
        return self.retinal.dct.signed_spatial(
            self.decode_dct(fields, exact=exact).float()
        )

    def probabilities(
        self,
        fields: torch.Tensor,
        *,
        sharpness: float,
    ) -> torch.Tensor:
        if sharpness <= 0.0:
            raise ValueError("V46 decoder sharpness must be positive")
        return torch.sigmoid(sharpness * self.signed_spatial(fields))

    def binary(self, fields: torch.Tensor, *, exact: bool = False) -> torch.Tensor:
        return (self.signed_spatial(fields, exact=exact) >= 0.0).to(fields.dtype)


class AdaptiveScaledFieldGeneratorBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension, eps=1e-6, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dimension, 3 * dimension),
        )
        self.gate = nn.Linear(dimension, 3 * dimension, bias=False)
        self.value = nn.Linear(dimension, 3 * dimension, bias=False)
        self.output = nn.Linear(3 * dimension, dimension, bias=False)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, residual_gate = self.modulation(condition).chunk(3, dim=-1)
        hidden = self.norm(state) * (1.0 + scale) + shift
        update = self.output(F.silu(self.gate(hidden)) * self.value(hidden))
        return state + torch.tanh(residual_gate) * update


class ConditionalScaledRetinalFieldGeneratorV46(nn.Module):
    """Sample full scaled retinal fields without unit normalization."""

    def __init__(self, config: ScaledRetinalGlyphLanguageV46Config) -> None:
        super().__init__()
        self.config = config
        self.noise_projection = nn.Linear(config.noise_dim, config.model_dim)
        self.context_projection = nn.Linear(config.model_dim, config.model_dim)
        self.anchor_projection = nn.Linear(config.field_dim, config.model_dim)
        self.condition_norm = nn.LayerNorm(config.model_dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                AdaptiveScaledFieldGeneratorBlock(config.model_dim)
                for _ in range(config.generator_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.model_dim, eps=1e-6)
        self.output = nn.Linear(config.model_dim, config.field_dim)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.generator_layers)
        for block in self.blocks:
            block.output.weight.data.mul_(residual_scale)
        nn.init.normal_(
            self.output.weight,
            std=self.config.generator_output_std,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        *,
        samples: int,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.config.model_dim:
            raise ValueError("V46 generator hidden state must be [N,model_dim]")
        if anchor.shape != (hidden.shape[0], self.config.field_dim):
            raise ValueError("V46 generator anchor does not align")
        if samples < 1 or noise_scale < 0.0:
            raise ValueError("V46 generator sampling settings are invalid")
        count = hidden.shape[0]
        noise = torch.randn(
            (count, samples, self.config.noise_dim),
            device=hidden.device,
            dtype=hidden.dtype,
            generator=generator,
        ).mul_(noise_scale)
        repeated_hidden = hidden[:, None].expand(-1, samples, -1)
        repeated_anchor = anchor[:, None].expand(-1, samples, -1)
        condition = self.condition_norm(
            self.context_projection(repeated_hidden)
            + self.anchor_projection(repeated_anchor.to(hidden.dtype))
        )
        state = self.noise_projection(noise)
        for block in self.blocks:
            state = block(state, condition)
        residual = self.output(self.output_norm(state)).float()
        return (
            repeated_anchor.float()
            + self.config.generator_residual_scale * residual
        )


class ScaledRetinalGlyphLanguageModelV46(nn.Module):
    """Causal image-stream model over exact scaled V45 retinal fields."""

    def __init__(
        self,
        config: ScaledRetinalGlyphLanguageV46Config,
        retinal_field: NoiseLimitedRetinalFieldV45,
        *,
        v45_checkpoint_sha256: str,
    ) -> None:
        super().__init__()
        if len(v45_checkpoint_sha256) != 64:
            raise ValueError("V46 requires a V45 checkpoint SHA-256")
        self.config = config
        self.v45_checkpoint_sha256 = v45_checkpoint_sha256
        self.field = ScaledNoiseLimitedRetinalFieldV46(
            retinal_field,
            reference_radius=config.reference_radius,
        )
        self.input_projection = nn.Linear(
            config.field_dim,
            config.model_dim,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.output_norm = RMSNorm(config.model_dim)
        self.anchor_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.field_dim),
        )
        self.generator = ConditionalScaledRetinalFieldGeneratorV46(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )
        self._initialize_language()

    def _initialize_language(self) -> None:
        for root in (self.input_projection, self.blocks, self.anchor_head):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.layers)
        for block in self.blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)
        nn.init.normal_(
            self.anchor_head[-1].weight,
            std=self.config.anchor_output_std,
        )

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def _validate_context(self, context: torch.Tensor, *, maximum: bool = True) -> None:
        self.field.retinal.dct._validate_pixels(context)
        if context.ndim != 5:
            raise ValueError("V46 context must be [B,T,1,32,32]")
        if context.shape[1] < 1:
            raise ValueError("V46 context cannot be empty")
        if maximum and context.shape[1] > self.config.maximum_cells:
            raise ValueError("V46 context exceeds 64 visual cells")

    def encode_cells(self, cells: torch.Tensor) -> torch.Tensor:
        self._validate_context(cells, maximum=False)
        return self.field.encode(cells)

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_context(context)
        visual = self.encode_cells(context)
        state = self.input_projection(visual.to(self.input_projection.weight.dtype))
        for block in self.blocks:
            state = block(state)
        hidden = self.output_norm(state)
        anchor = self.anchor_head(hidden).float()
        return {
            "context_visual": visual,
            "hidden_states": hidden,
            "anchor_fields": anchor,
        }

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.language(context)

    def sample_fields(
        self,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        *,
        samples: int,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.generator(
            hidden,
            anchor,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )

    @torch.no_grad()
    def score_image_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 4 or tuple(candidates.shape[1:]) != (1, 32, 32):
            raise ValueError("V46 evaluator candidates must be [N,1,32,32]")
        anchor = self.language(context)["anchor_fields"][:, -1]
        candidate_fields = self.field.encode(candidates)
        return self.contrastive_scale.float() * (
            self.field.directions(anchor)
            @ self.field.directions(candidate_fields).transpose(0, 1)
        )

    @torch.no_grad()
    def sample_next(
        self,
        context: torch.Tensor,
        *,
        samples: int = 4,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.language(context)
        hidden = output["hidden_states"][:, -1]
        anchor = output["anchor_fields"][:, -1]
        fields = self.sample_fields(
            hidden,
            anchor,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )
        squared_distance = (fields.float() - anchor[:, None].float()).square().sum(-1)
        selected_indices = squared_distance.argmin(dim=1)
        selected_fields = fields[
            torch.arange(len(fields), device=fields.device),
            selected_indices,
        ]
        pixels = self.field.binary(selected_fields)
        return pixels, {
            "sample_fields": fields,
            "sample_squared_distances": squared_distance,
            "selected_indices": selected_indices,
            "selected_fields": selected_fields,
            "anchor_fields": anchor,
        }

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        *,
        new_cells: int,
        samples: int = 4,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_context(prefix, maximum=False)
        if new_cells < 1:
            raise ValueError("V46 generation requires at least one new cell")
        sequence = prefix
        generated: list[torch.Tensor] = []
        generated_fields: list[torch.Tensor] = []
        for _ in range(new_cells):
            context = sequence[:, -self.config.maximum_cells :]
            pixels, trace = self.sample_next(
                context,
                samples=samples,
                generator=generator,
                noise_scale=noise_scale,
            )
            generated.append(pixels)
            generated_fields.append(trace["selected_fields"])
            sequence = torch.cat((sequence, pixels[:, None]), dim=1)
        return sequence, {
            "generated_cells": torch.stack(generated, dim=1),
            "generated_fields": torch.stack(generated_fields, dim=1),
            "rereads_generated_pixels": torch.tensor(True, device=sequence.device),
        }


def scaled_retinal_glyph_language_v46_config_payload(
    config: ScaledRetinalGlyphLanguageV46Config,
) -> dict[str, Any]:
    return asdict(config)


def scaled_retinal_glyph_language_v46_config_from_payload(
    payload: Mapping[str, Any],
) -> ScaledRetinalGlyphLanguageV46Config:
    return ScaledRetinalGlyphLanguageV46Config(**dict(payload))


def scaled_retinal_glyph_language_v46_boundary_receipt(
    model: ScaledRetinalGlyphLanguageModelV46,
) -> dict[str, Any]:
    forbidden = (
        "token",
        "vocab",
        "unicode",
        "character_id",
        "codebook",
        "quant",
        "ocr",
        "lookup",
    )
    suspicious = sorted(
        name
        for name, _ in model.named_parameters()
        if any(fragment in name.lower() for fragment in forbidden)
    )
    field_parameters = sum(parameter.numel() for parameter in model.field.parameters())
    return {
        "architecture": V46_ARCHITECTURE,
        "config": scaled_retinal_glyph_language_v46_config_payload(model.config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "field_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.field.parameters()
            if parameter.requires_grad
        ),
        "field_parameters": field_parameters,
        "v45_checkpoint_sha256": model.v45_checkpoint_sha256,
        "v45_field_state_sha256": noise_limited_retinal_field_v45_state_sha256(
            model.field.retinal
        ),
        "reference_radius": float(model.field.reference_radius),
        "parameter_names_with_forbidden_fragments": suspicious,
        "language_parameters": list(inspect.signature(model.language).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "input_is_continuous_image_stream": True,
        "output_is_full_continuous_image_field": True,
        "output_is_direct_raster": True,
        "field_transform_is_fixed_and_invertible": True,
        "field_preserves_direction_and_radius": True,
        "causal_over_visual_time": True,
        "rereads_generated_pixels": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_embedding": False,
        "uses_vocabulary_output": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_quantization": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }

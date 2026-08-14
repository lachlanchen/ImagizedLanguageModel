from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_glyph_language_data import V42_ARCHITECTURE
from .visual_cell_stream import CausalVisualBlock, RMSNorm


@dataclass(frozen=True)
class CanonicalGlyphLanguageConfig:
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

    def __post_init__(self) -> None:
        if self.cell_size != 32 or self.field_dim != self.cell_size**2:
            raise ValueError("V42 requires a full 32x32 continuous glyph field")
        if self.maximum_cells != 64:
            raise ValueError("V42 fixes a 64-cell visual context")
        if self.model_dim < 128 or self.layers < 1:
            raise ValueError("V42 causal model is underspecified")
        if self.heads < 1 or self.model_dim % self.heads:
            raise ValueError("V42 model width must divide into attention heads")
        if self.mlp_ratio < 2.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("V42 causal block settings are invalid")
        if self.noise_dim < 16 or self.generator_layers < 1:
            raise ValueError("V42 field sampler is underspecified")
        if self.generator_residual_scale <= 0.0:
            raise ValueError("V42 generator residual scale must be positive")
        if not 0.0 < self.binary_threshold < 1.0:
            raise ValueError("V42 binary threshold must lie in (0,1)")
        if self.decoder_sharpness <= 0.0:
            raise ValueError("V42 decoder sharpness must be positive")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V42 initial temperature is invalid")


def orthonormal_dct_matrix(
    size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if size < 2:
        raise ValueError("DCT size must be at least two")
    frequency = torch.arange(size, dtype=torch.float64)[:, None]
    position = torch.arange(size, dtype=torch.float64)[None, :] + 0.5
    basis = torch.cos(math.pi * frequency * position / size)
    basis[0].mul_(math.sqrt(1.0 / size))
    basis[1:].mul_(math.sqrt(2.0 / size))
    return basis.to(dtype=dtype)


class OrthonormalGlyphField(nn.Module):
    """Exact fixed transform between binary rasters and continuous DCT fields."""

    def __init__(
        self,
        *,
        size: int = 32,
        binary_threshold: float = 0.5,
        decoder_sharpness: float = 6.0,
    ) -> None:
        super().__init__()
        if size != 32:
            raise ValueError("V42 field transform is fixed to 32x32")
        self.size = int(size)
        self.field_dim = size**2
        self.binary_threshold = float(binary_threshold)
        self.decoder_sharpness = float(decoder_sharpness)
        self.register_buffer(
            "basis",
            orthonormal_dct_matrix(size),
            persistent=True,
        )

    def _validate_pixels(self, pixels: torch.Tensor) -> None:
        if not pixels.is_floating_point():
            raise TypeError("V42 field input must be floating point")
        if pixels.ndim < 4 or tuple(pixels.shape[-3:]) != (1, 32, 32):
            raise ValueError("V42 field input must end in [1,32,32]")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("V42 field input must be finite")
        if not bool(((pixels >= 0.0) & (pixels <= 1.0)).all()):
            raise ValueError("V42 field input must lie in [0,1]")

    def _validate_fields(self, fields: torch.Tensor) -> None:
        if not fields.is_floating_point():
            raise TypeError("V42 fields must be floating point")
        if fields.ndim < 2 or fields.shape[-1] != self.field_dim:
            raise ValueError("V42 fields must end in 1024 coefficients")
        if not bool(torch.isfinite(fields).all()):
            raise ValueError("V42 fields must be finite")

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self._validate_pixels(pixels)
        leading = pixels.shape[:-3]
        signed = (
            (pixels.reshape(-1, 32, 32) >= self.binary_threshold)
            .to(dtype=self.basis.dtype)
            .mul_(2.0)
            .sub_(1.0)
        )
        basis = self.basis.to(device=pixels.device)
        spectral = torch.matmul(torch.matmul(basis, signed), basis.transpose(0, 1))
        return spectral.reshape(*leading, self.field_dim)

    def normalize(self, fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(fields)
        return F.normalize(fields.float(), dim=-1)

    def encode_unit(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.encode(pixels))

    def signed_spatial(self, fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(fields)
        leading = fields.shape[:-1]
        spectral = fields.float().reshape(-1, 32, 32)
        basis = self.basis.float().to(device=fields.device)
        signed = torch.matmul(
            torch.matmul(basis.transpose(0, 1), spectral),
            basis,
        )
        return signed.reshape(*leading, 1, 32, 32)

    def probabilities(self, unit_fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(unit_fields)
        full_fields = unit_fields.float() * math.sqrt(self.field_dim)
        return torch.sigmoid(
            self.decoder_sharpness * self.signed_spatial(full_fields)
        )

    def binary(self, unit_fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(unit_fields)
        full_fields = unit_fields.float() * math.sqrt(self.field_dim)
        return (self.signed_spatial(full_fields) >= 0.0).to(unit_fields.dtype)


class AdaptiveFieldGeneratorBlock(nn.Module):
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


class ConditionalGlyphFieldGenerator(nn.Module):
    """Sample a continuous glyph field from visual context and continuous noise."""

    def __init__(self, config: CanonicalGlyphLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.noise_projection = nn.Linear(config.noise_dim, config.model_dim)
        self.context_projection = nn.Linear(config.model_dim, config.model_dim)
        self.anchor_projection = nn.Linear(config.field_dim, config.model_dim)
        self.condition_norm = nn.LayerNorm(config.model_dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                AdaptiveFieldGeneratorBlock(config.model_dim)
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
            raise ValueError("V42 generator hidden state must be [N,model_dim]")
        if anchor.shape != (hidden.shape[0], self.config.field_dim):
            raise ValueError("V42 generator anchor does not align")
        if samples < 1 or noise_scale < 0.0:
            raise ValueError("V42 generator sampling settings are invalid")
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
        fields = repeated_anchor.float() + self.config.generator_residual_scale * residual
        return F.normalize(fields, dim=-1)


class CanonicalGlyphLanguageModel(nn.Module):
    """Causal image-stream language model with direct continuous image output."""

    def __init__(self, config: CanonicalGlyphLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.field = OrthonormalGlyphField(
            size=config.cell_size,
            binary_threshold=config.binary_threshold,
            decoder_sharpness=config.decoder_sharpness,
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
        self.generator = ConditionalGlyphFieldGenerator(config)
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

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def _validate_context(self, context: torch.Tensor, *, maximum: bool = True) -> None:
        self.field._validate_pixels(context)
        if context.ndim != 5:
            raise ValueError("V42 context must be [B,T,1,32,32]")
        if context.shape[1] < 1:
            raise ValueError("V42 context cannot be empty")
        if maximum and context.shape[1] > self.config.maximum_cells:
            raise ValueError("V42 context exceeds 64 visual cells")

    def encode_cells(self, cells: torch.Tensor) -> torch.Tensor:
        self._validate_context(cells, maximum=False)
        return self.field.encode_unit(cells)

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_context(context)
        visual = self.encode_cells(context)
        state = self.input_projection(visual.to(self.input_projection.weight.dtype))
        for block in self.blocks:
            state = block(state)
        hidden = self.output_norm(state)
        anchor = F.normalize(self.anchor_head(hidden).float(), dim=-1)
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
            raise ValueError("V42 evaluator candidates must be [N,1,32,32]")
        anchor = self.language(context)["anchor_fields"][:, -1]
        candidate_fields = self.field.encode_unit(candidates)
        return self.contrastive_scale.float() * anchor @ candidate_fields.transpose(0, 1)

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
        scores = torch.einsum("bsd,bd->bs", fields, anchor)
        selected_indices = scores.argmax(dim=1)
        selected_fields = fields[
            torch.arange(len(fields), device=fields.device), selected_indices
        ]
        pixels = self.field.binary(selected_fields)
        return pixels, {
            "sample_fields": fields,
            "sample_scores": scores,
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
            raise ValueError("V42 generation requires at least one new cell")
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


def canonical_glyph_language_config_payload(
    config: CanonicalGlyphLanguageConfig,
) -> dict[str, Any]:
    return asdict(config)


def canonical_glyph_language_config_from_payload(
    payload: dict[str, Any],
) -> CanonicalGlyphLanguageConfig:
    return CanonicalGlyphLanguageConfig(**payload)


def canonical_glyph_language_boundary_receipt(
    model: CanonicalGlyphLanguageModel,
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
    return {
        "architecture": V42_ARCHITECTURE,
        "config": canonical_glyph_language_config_payload(model.config),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "language_parameters": list(inspect.signature(model.language).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "input_is_continuous_image_stream": True,
        "output_is_continuous_image_field": True,
        "output_is_direct_raster": True,
        "field_transform_is_fixed_and_invertible": True,
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

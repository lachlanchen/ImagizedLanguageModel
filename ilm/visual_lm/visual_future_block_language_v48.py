from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_glyph_language import OrthonormalGlyphField
from .visual_cell_stream import CausalVisualBlock, RMSNorm
from .visual_future_block_language_v48_data import V48_ARCHITECTURE


@dataclass(frozen=True)
class VisualFutureBlockLanguageConfigV48:
    cell_size: int = 32
    maximum_cells: int = 64
    future_horizons: int = 4
    field_dim: int = 1_024
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    binary_threshold: float = 0.5
    decoder_sharpness: float = 6.0
    initial_temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.cell_size != 32 or self.field_dim != self.cell_size**2:
            raise ValueError("V48 requires a full 32x32 continuous image field")
        if self.maximum_cells != 64 or self.future_horizons != 4:
            raise ValueError("V48 fixes 64 visible cells and four futures")
        if self.model_dim < 128 or self.layers < 1 or self.heads < 1:
            raise ValueError("V48 causal model is underspecified")
        if self.model_dim % self.heads:
            raise ValueError("V48 model width must divide into attention heads")
        if self.mlp_ratio < 2.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("V48 causal block settings are invalid")
        if self.binary_threshold != 0.5 or self.decoder_sharpness != 6.0:
            raise ValueError("V48 fixes rasterization and training sharpness")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V48 initial contrastive temperature is invalid")


class VisualFutureBlockLanguageModelV48(nn.Module):
    """Image-only causal reader with dense four-future visual prediction."""

    def __init__(self, config: VisualFutureBlockLanguageConfigV48) -> None:
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
        self.future_offsets = nn.Parameter(
            torch.empty(config.future_horizons - 1, config.model_dim)
        )
        self.visual_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.field_dim),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )
        self._initialize()

    def _initialize(self) -> None:
        for root in (self.input_projection, self.blocks, self.visual_head):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.normal_(self.future_offsets, std=0.02)
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
            raise ValueError("V48 context must be [B,T,1,32,32]")
        if context.shape[1] < 1:
            raise ValueError("V48 context cannot be empty")
        if maximum and context.shape[1] > self.config.maximum_cells:
            raise ValueError("V48 context exceeds 64 visible cells")

    def encode_cells(self, cells: torch.Tensor) -> torch.Tensor:
        self._validate_context(cells, maximum=False)
        return self.field.encode_unit(cells)

    def _all_offsets(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        zero = torch.zeros(
            1,
            self.config.model_dim,
            device=device,
            dtype=dtype,
        )
        return torch.cat((zero, self.future_offsets.to(dtype=dtype)), dim=0)

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_context(context)
        visual = self.encode_cells(context)
        state = self.input_projection(visual.to(self.input_projection.weight.dtype))
        for block in self.blocks:
            state = block(state)
        hidden = self.output_norm(state)
        offsets = self._all_offsets(device=hidden.device, dtype=hidden.dtype)
        conditioned = hidden[:, :, None, :] + offsets[None, None, :, :]
        future = F.normalize(self.visual_head(conditioned).float(), dim=-1)
        return {
            "context_visual": visual,
            "hidden_states": hidden,
            "anchor_fields": future[:, :, 0],
            "future_anchor_fields": future,
        }

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.language(context)

    @torch.no_grad()
    def forecast(self, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.language(context)
        fields = output["future_anchor_fields"][:, -1]
        pixels = self.field.binary(fields)
        reread = self.field.encode_unit(pixels)
        return pixels, {
            "future_fields": fields,
            "visible_reread_fields": reread,
            "anchor_fields": fields[:, 0],
            "rereads_visible_pixels": torch.tensor(True, device=context.device),
        }

    @torch.no_grad()
    def next_image(self, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pixels, trace = self.forecast(context)
        return pixels[:, 0], {
            **trace,
            "selected_fields": trace["future_fields"][:, 0],
            "selected_reread_fields": trace["visible_reread_fields"][:, 0],
        }

    @torch.no_grad()
    def score_image_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 4 or tuple(candidates.shape[1:]) != (1, 32, 32):
            raise ValueError("V48 evaluator candidates must be [N,1,32,32]")
        anchor = self.language(context)["anchor_fields"][:, -1]
        candidate_fields = self.field.encode_unit(candidates)
        return (
            self.contrastive_scale.float()
            * anchor.float()
            @ candidate_fields.float().transpose(0, 1)
        )

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        *,
        new_cells: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_context(prefix, maximum=False)
        if new_cells < 1:
            raise ValueError("V48 generation requires at least one new cell")
        sequence = prefix
        generated: list[torch.Tensor] = []
        proposed: list[torch.Tensor] = []
        reread: list[torch.Tensor] = []
        for _ in range(new_cells):
            context = sequence[:, -self.config.maximum_cells :]
            pixels, trace = self.next_image(context)
            generated.append(pixels)
            proposed.append(trace["selected_fields"])
            reread.append(trace["selected_reread_fields"])
            sequence = torch.cat((sequence, pixels[:, None]), dim=1)
        return sequence, {
            "generated_cells": torch.stack(generated, dim=1),
            "generated_fields": torch.stack(proposed, dim=1),
            "reread_fields": torch.stack(reread, dim=1),
            "rereads_generated_pixels": torch.tensor(True, device=sequence.device),
        }


def visual_future_block_language_config_payload_v48(
    config: VisualFutureBlockLanguageConfigV48,
) -> dict[str, Any]:
    return asdict(config)


def visual_future_block_language_config_from_payload_v48(
    payload: dict[str, Any],
) -> VisualFutureBlockLanguageConfigV48:
    return VisualFutureBlockLanguageConfigV48(**payload)


def visual_future_block_language_boundary_receipt_v48(
    model: VisualFutureBlockLanguageModelV48,
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
        "architecture": V48_ARCHITECTURE,
        "config": visual_future_block_language_config_payload_v48(model.config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "language_parameters": list(inspect.signature(model.language).parameters),
        "forecast_parameters": list(inspect.signature(model.forecast).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "input_is_continuous_image_stream": True,
        "output_is_continuous_image_block": True,
        "output_is_direct_raster": True,
        "field_transform_is_fixed_and_invertible": True,
        "inverse_dct_threshold_is_fixed_zero": True,
        "causal_over_visual_time": True,
        "predicts_four_future_images_densely": True,
        "rereads_generated_pixels": True,
        "uses_stochastic_generator": False,
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


__all__ = [
    "VisualFutureBlockLanguageConfigV48",
    "VisualFutureBlockLanguageModelV48",
    "visual_future_block_language_boundary_receipt_v48",
    "visual_future_block_language_config_from_payload_v48",
    "visual_future_block_language_config_payload_v48",
]

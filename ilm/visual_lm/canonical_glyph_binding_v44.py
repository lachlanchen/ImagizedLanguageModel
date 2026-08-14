from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_boundary_receipt,
)


V44_ARCHITECTURE = "canonical-glyph-binding-v44"


@dataclass(frozen=True)
class CanonicalGlyphBindingV44Config:
    suffix_cells: int = 4
    residual_dim: int = 192
    residual_layers: int = 2
    residual_heads: int = 6
    residual_dropout: float = 0.10
    residual_scale: float = 0.50

    def __post_init__(self) -> None:
        if self.suffix_cells != 4:
            raise ValueError("V44 fixes the shared suffix to four glyph images")
        if self.residual_dim != 192 or self.residual_layers != 2:
            raise ValueError("V44 fixes a two-layer width-192 residual")
        if self.residual_heads != 6 or self.residual_dim % self.residual_heads:
            raise ValueError("V44 fixes six residual attention heads")
        if self.residual_dropout != 0.10:
            raise ValueError("V44 fixes residual dropout at 0.10")
        if self.residual_scale != 0.50:
            raise ValueError("V44 fixes tangent residual scale at 0.50")


class HistoryCrossAttentionBlock(nn.Module):
    def __init__(self, config: CanonicalGlyphBindingV44Config) -> None:
        super().__init__()
        dimension = config.residual_dim
        self.query_norm = nn.LayerNorm(dimension, eps=1e-6)
        self.memory_norm = nn.LayerNorm(dimension, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dimension,
            config.residual_heads,
            dropout=config.residual_dropout,
            bias=False,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dimension, eps=1e-6)
        self.gate = nn.Linear(dimension, 3 * dimension, bias=False)
        self.value = nn.Linear(dimension, 3 * dimension, bias=False)
        self.output = nn.Linear(3 * dimension, dimension, bias=False)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3 or query.shape[1] != 1:
            raise ValueError("V44 residual query must be [B,1,D]")
        if memory.ndim != 3 or memory.shape[0] != query.shape[0]:
            raise ValueError("V44 residual memory must be [B,T,D]")
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        query = query + self.dropout(attended)
        normalized = self.mlp_norm(query)
        update = self.output(F.silu(self.gate(normalized)) * self.value(normalized))
        return query + self.dropout(update)


class LongHistoryTangentResidual(nn.Module):
    """Read only the pre-suffix visual history and perturb a frozen V42 field."""

    def __init__(
        self,
        language_config: CanonicalGlyphLanguageConfig,
        config: CanonicalGlyphBindingV44Config,
    ) -> None:
        super().__init__()
        maximum_prefix = language_config.maximum_cells - config.suffix_cells
        self.language_config = language_config
        self.config = config
        self.hidden_projection = nn.Linear(
            language_config.model_dim,
            config.residual_dim,
            bias=False,
        )
        self.visual_projection = nn.Linear(
            language_config.field_dim,
            config.residual_dim,
            bias=False,
        )
        self.query_projection = nn.Linear(
            3 * language_config.model_dim,
            config.residual_dim,
            bias=False,
        )
        self.innovation_projection = nn.Linear(
            language_config.model_dim,
            config.residual_dim,
            bias=False,
        )
        self.relative_positions = nn.Parameter(
            torch.empty(maximum_prefix, config.residual_dim)
        )
        self.memory_norm = nn.LayerNorm(config.residual_dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [HistoryCrossAttentionBlock(config) for _ in range(config.residual_layers)]
        )
        self.output_norm = nn.LayerNorm(config.residual_dim, eps=1e-6)
        self.output = nn.Linear(
            config.residual_dim,
            language_config.field_dim,
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.relative_positions, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.residual_layers)
        for block in self.blocks:
            block.attention.out_proj.weight.data.mul_(residual_scale)
            block.output.weight.data.mul_(residual_scale)

    def forward(
        self,
        full_hidden: torch.Tensor,
        full_visual: torch.Tensor,
        suffix_hidden: torch.Tensor,
        base_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix = self.config.suffix_cells
        if full_hidden.ndim != 3 or full_visual.ndim != 3:
            raise ValueError("V44 frozen states must be [B,T,D]")
        if full_hidden.shape[:2] != full_visual.shape[:2]:
            raise ValueError("V44 frozen hidden and visual histories do not align")
        if full_hidden.shape[1] <= suffix:
            raise ValueError("V44 residual requires history before the suffix")
        if suffix_hidden.shape[:2] != (len(full_hidden), suffix):
            raise ValueError("V44 suffix hidden state has the wrong shape")
        if base_anchor.shape != (len(full_hidden), self.language_config.field_dim):
            raise ValueError("V44 base anchor has the wrong shape")

        prefix_hidden = full_hidden[:, :-suffix]
        prefix_visual = full_visual[:, :-suffix]
        prefix_length = prefix_hidden.shape[1]
        if prefix_length > len(self.relative_positions):
            raise ValueError("V44 prefix exceeds the preregistered memory length")
        positions = self.relative_positions[-prefix_length:][None]
        memory = self.memory_norm(
            self.hidden_projection(prefix_hidden)
            + self.visual_projection(prefix_visual)
            + positions.to(prefix_hidden.dtype)
        )
        full_last = full_hidden[:, -1]
        suffix_last = suffix_hidden[:, -1]
        innovation = full_last - suffix_last
        query = self.query_projection(
            torch.cat((full_last, suffix_last, innovation), dim=-1)
        )
        query = query + self.innovation_projection(innovation)
        query = query[:, None]
        for block in self.blocks:
            query = block(query, memory)
        residual = self.output(self.output_norm(query[:, 0])).float()
        base_anchor = F.normalize(base_anchor.float(), dim=-1)
        tangent = residual - (residual * base_anchor).sum(
            dim=-1, keepdim=True
        ) * base_anchor
        adapted = F.normalize(
            base_anchor + self.config.residual_scale * tangent,
            dim=-1,
        )
        return adapted, tangent


class CanonicalGlyphBindingV44(nn.Module):
    """Frozen V42 reader plus a candidate-independent long-history residual."""

    def __init__(
        self,
        language_config: CanonicalGlyphLanguageConfig,
        config: CanonicalGlyphBindingV44Config = CanonicalGlyphBindingV44Config(),
    ) -> None:
        super().__init__()
        self.config = config
        self.base = CanonicalGlyphLanguageModel(language_config)
        self.adapter = LongHistoryTangentResidual(language_config, config)
        self.freeze_base()

    @property
    def language_config(self) -> CanonicalGlyphLanguageConfig:
        return self.base.config

    @property
    def field(self):
        return self.base.field

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.base.contrastive_scale

    def freeze_base(self) -> None:
        self.base.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> CanonicalGlyphBindingV44:
        super().train(mode)
        self.base.eval()
        return self

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        self.base._validate_context(context)
        with torch.no_grad():
            base_output = self.base.language(context)
        base_anchors = base_output["anchor_fields"].detach()
        zeros = torch.zeros_like(base_anchors)
        if context.shape[1] <= self.config.suffix_cells:
            return {
                **base_output,
                "base_anchor_fields": base_anchors,
                "residual_fields": zeros,
            }
        with torch.no_grad():
            suffix_output = self.base.language(context[:, -self.config.suffix_cells :])
        adapted, tangent = self.adapter(
            base_output["hidden_states"].detach(),
            base_output["context_visual"].detach(),
            suffix_output["hidden_states"].detach(),
            base_anchors[:, -1],
        )
        anchor_fields = torch.cat((base_anchors[:, :-1], adapted[:, None]), dim=1)
        residual_fields = torch.cat((zeros[:, :-1], tangent[:, None]), dim=1)
        return {
            **base_output,
            "anchor_fields": anchor_fields,
            "base_anchor_fields": base_anchors,
            "residual_fields": residual_fields,
        }

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.language(context)

    def pair_outputs(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if contexts.ndim != 6 or tuple(contexts.shape[1:]) != (
            2,
            64,
            1,
            32,
            32,
        ):
            raise ValueError("V44 pair contexts must be [B,2,64,1,32,32]")
        if candidates.ndim != 5 or tuple(candidates.shape[1:]) != (2, 1, 32, 32):
            raise ValueError("V44 pair candidates must be [B,2,1,32,32]")
        batch = len(contexts)
        output = self.language(contexts.flatten(0, 1))
        anchors = output["anchor_fields"][:, -1].reshape(batch, 2, -1)
        base_anchors = output["base_anchor_fields"][:, -1].reshape(batch, 2, -1)
        candidate_fields = self.field.encode_unit(candidates.flatten(0, 1)).reshape(
            batch, 2, -1
        )
        logits = self.contrastive_scale.float() * torch.einsum(
            "bid,bjd->bij",
            anchors.float(),
            candidate_fields.float(),
        )
        return {
            "anchors": anchors,
            "base_anchors": base_anchors,
            "candidate_fields": candidate_fields,
            "logits": logits,
        }

    def pair_logits(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return self.pair_outputs(contexts, candidates)["logits"]

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
        fields = self.base.sample_fields(
            hidden,
            anchor,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )
        scores = torch.einsum("bsd,bd->bs", fields, anchor)
        selected_indices = scores.argmax(dim=1)
        rows = torch.arange(len(fields), device=fields.device)
        selected_fields = fields[rows, selected_indices]
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
        self.base._validate_context(prefix, maximum=False)
        if new_cells < 1:
            raise ValueError("V44 generation requires at least one new cell")
        sequence = prefix
        generated: list[torch.Tensor] = []
        generated_fields: list[torch.Tensor] = []
        for _ in range(new_cells):
            context = sequence[:, -self.language_config.maximum_cells :]
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


def canonical_glyph_binding_v44_config_payload(
    config: CanonicalGlyphBindingV44Config,
) -> dict[str, Any]:
    return asdict(config)


def canonical_glyph_binding_v44_config_from_payload(
    payload: dict[str, Any],
) -> CanonicalGlyphBindingV44Config:
    return CanonicalGlyphBindingV44Config(**payload)


def canonical_glyph_binding_v44_boundary_receipt(
    model: CanonicalGlyphBindingV44,
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
    adapter_parameters = sum(parameter.numel() for parameter in model.adapter.parameters())
    return {
        "architecture": V44_ARCHITECTURE,
        "config": canonical_glyph_binding_v44_config_payload(model.config),
        "base": canonical_glyph_language_boundary_receipt(model.base),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_parameters": adapter_parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "base_parameters_frozen": not any(
            parameter.requires_grad for parameter in model.base.parameters()
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "language_parameters": list(inspect.signature(model.language).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "input_is_continuous_image_stream": True,
        "output_is_continuous_image_field": True,
        "output_is_direct_raster": True,
        "candidate_independent_residual": True,
        "long_history_excludes_shared_suffix": True,
        "tangent_field_update": True,
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

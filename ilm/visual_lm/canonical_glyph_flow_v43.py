from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .canonical_glyph_flow_v43_data import V43_ARCHITECTURE
from .canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_boundary_receipt,
)
from .ink_writer import FovealInkFlow, FovealWriterConfig, sample_foveal_ink


@dataclass(frozen=True)
class CanonicalGlyphFlowV43Config:
    writer_base_channels: int = 64
    writer_context_dim: int = 256
    condition_dropout: float = 0.10
    flow_steps: int = 16
    generated_candidates: int = 4
    guidance_scale: float = 1.25

    def __post_init__(self) -> None:
        if self.writer_base_channels != 64 or self.writer_context_dim != 256:
            raise ValueError("V43 fixes the spatial writer topology")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("V43 condition dropout must lie in [0,1)")
        if self.flow_steps != 16 or self.generated_candidates != 4:
            raise ValueError("V43 fixes inference integration and sample count")
        if self.guidance_scale != 1.25:
            raise ValueError("V43 fixes classifier-free guidance at 1.25")


class CanonicalGlyphFlowV43(nn.Module):
    """V42 visual reader with a bank-free conditional spatial flow writer."""

    def __init__(
        self,
        language_config: CanonicalGlyphLanguageConfig,
        config: CanonicalGlyphFlowV43Config = CanonicalGlyphFlowV43Config(),
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = CanonicalGlyphLanguageModel(language_config)
        self.writer = FovealInkFlow(
            FovealWriterConfig(
                fovea_size=language_config.cell_size,
                condition_dim=language_config.model_dim,
                base_channels=config.writer_base_channels,
                context_dim=config.writer_context_dim,
                condition_dropout=config.condition_dropout,
            )
        )

    def freeze_writer(self) -> None:
        self.writer.requires_grad_(False).eval()

    def unfreeze_writer(self) -> None:
        self.writer.requires_grad_(True).train()

    def freeze_language(self) -> None:
        self.language_model.requires_grad_(False).eval()

    def unfreeze_language_core(self) -> None:
        self.language_model.requires_grad_(True).train()
        self.language_model.generator.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> CanonicalGlyphFlowV43:
        super().train(mode)
        if not any(
            parameter.requires_grad for parameter in self.language_model.parameters()
        ):
            self.language_model.eval()
        if not any(
            parameter.requires_grad
            for parameter in self.language_model.generator.parameters()
        ):
            self.language_model.generator.eval()
        return self

    def anchor_ink_plan(self, anchor_fields: torch.Tensor) -> torch.Tensor:
        probabilities = self.language_model.field.probabilities(anchor_fields)
        return probabilities.mul(2.0).sub(1.0)

    def pair_logits(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or tuple(contexts.shape[1:]) != (
            2,
            64,
            1,
            32,
            32,
        ):
            raise ValueError("V43 pair contexts must be [B,2,64,1,32,32]")
        if candidates.ndim != 5 or tuple(candidates.shape[1:]) != (2, 1, 32, 32):
            raise ValueError("V43 pair candidates must be [B,2,1,32,32]")
        batch = contexts.shape[0]
        output = self.language_model(contexts.flatten(0, 1))
        anchors = output["anchor_fields"][:, -1].reshape(batch, 2, -1)
        candidate_fields = self.language_model.field.encode_unit(
            candidates.flatten(0, 1)
        ).reshape(batch, 2, -1)
        return self.language_model.contrastive_scale.float() * torch.einsum(
            "bid,bjd->bij",
            anchors.float(),
            candidate_fields.float(),
        )

    def flow_inputs(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.language_model(context)
        hidden = output["hidden_states"][:, -1]
        anchor = output["anchor_fields"][:, -1]
        plan = self.anchor_ink_plan(anchor)
        return hidden, anchor, plan

    @torch.no_grad()
    def sample_next(
        self,
        context: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        samples: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = self.config.generated_candidates if samples is None else int(samples)
        integration_steps = self.config.flow_steps if steps is None else int(steps)
        guidance = (
            self.config.guidance_scale
            if guidance_scale is None
            else float(guidance_scale)
        )
        if count < 1 or integration_steps < 1 or guidance < 0.0:
            raise ValueError("V43 sampling settings are invalid")
        hidden, anchor, plan = self.flow_inputs(context)
        repeated_hidden = hidden[:, None].expand(-1, count, -1).flatten(0, 1)
        repeated_plan = plan[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        signed = sample_foveal_ink(
            self.writer,
            repeated_hidden,
            repeated_plan,
            steps=integration_steps,
            guidance_scale=guidance,
            generator=generator,
        )
        candidate_pixels = (
            signed.add(1.0).div(2.0).reshape(len(context), count, 1, 32, 32)
        )
        visible_candidates = (candidate_pixels >= 0.5).to(candidate_pixels.dtype)
        reread_fields = self.language_model.field.encode_unit(
            visible_candidates.flatten(0, 1)
        ).reshape(len(context), count, -1)
        scores = torch.einsum("bsd,bd->bs", reread_fields, anchor.float())
        selected_indices = scores.argmax(dim=1)
        rows = torch.arange(len(context), device=context.device)
        selected_pixels = visible_candidates[rows, selected_indices]
        selected_fields = reread_fields[rows, selected_indices]
        return selected_pixels, {
            "candidate_pixels": visible_candidates,
            "candidate_fields": reread_fields,
            "sample_scores": scores,
            "selected_indices": selected_indices,
            "selected_fields": selected_fields,
            "anchor_fields": anchor,
            "ink_plan": plan,
        }

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        *,
        new_cells: int,
        generator: torch.Generator | None = None,
        samples: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.language_model._validate_context(prefix, maximum=False)
        if new_cells < 1:
            raise ValueError("V43 generation requires at least one new cell")
        sequence = prefix
        generated: list[torch.Tensor] = []
        generated_fields: list[torch.Tensor] = []
        for _ in range(new_cells):
            context = sequence[:, -self.language_model.config.maximum_cells :]
            pixels, trace = self.sample_next(
                context,
                generator=generator,
                samples=samples,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            generated.append(pixels)
            generated_fields.append(trace["selected_fields"])
            sequence = torch.cat((sequence, pixels[:, None]), dim=1)
        return sequence, {
            "generated_cells": torch.stack(generated, dim=1),
            "generated_fields": torch.stack(generated_fields, dim=1),
            "rereads_generated_pixels": torch.tensor(True, device=sequence.device),
        }


def canonical_glyph_flow_v43_config_payload(
    config: CanonicalGlyphFlowV43Config,
) -> dict[str, Any]:
    return asdict(config)


def canonical_glyph_flow_v43_config_from_payload(
    payload: dict[str, Any],
) -> CanonicalGlyphFlowV43Config:
    return CanonicalGlyphFlowV43Config(**payload)


def canonical_glyph_flow_v43_boundary_receipt(
    model: CanonicalGlyphFlowV43,
) -> dict[str, Any]:
    language = canonical_glyph_language_boundary_receipt(model.language_model)
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
        "architecture": V43_ARCHITECTURE,
        "config": canonical_glyph_flow_v43_config_payload(model.config),
        "language": language,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "writer_parameters": sum(
            parameter.numel() for parameter in model.writer.parameters()
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "sample_next_parameters": list(inspect.signature(model.sample_next).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "input_is_continuous_image_stream": True,
        "output_is_direct_raster": True,
        "conditional_spatial_flow": True,
        "candidate_selection_uses_reader_reread": True,
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

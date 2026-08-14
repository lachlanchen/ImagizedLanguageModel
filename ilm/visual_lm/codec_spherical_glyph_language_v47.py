from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_glyph_language import AdaptiveFieldGeneratorBlock
from .continuous_glyph_codec import (
    V34_ARCHITECTURE,
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V47_ARCHITECTURE = "codec-spherical-glyph-language-v47"
V47_PROTOCOL = "references/codec_spherical_glyph_language_v47_protocol.md"
V47_REQUIRED_CODEC_CHECKPOINT_SHA256 = (
    "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"
)
V47_REQUIRED_CODEC_STATE_SHA256 = (
    "140e6d68d2be3bcbcdb6fb74b27a8f1258caba54bc0f0888ba8acecc48c22edb"
)
V47_CODEC_UPDATE = 6_000


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_verified_v34_codec(
    checkpoint_path: str | Path,
    *,
    strict_digest: bool = True,
) -> tuple[ContinuousGlyphCodec, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V47 V34 checkpoint does not exist: {path}")
    checkpoint_sha256 = _file_sha256(path)
    if (
        strict_digest
        and checkpoint_sha256 != V47_REQUIRED_CODEC_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "V47 V34 checkpoint digest differs from the frozen protocol: "
            f"{checkpoint_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("V47 V34 checkpoint must be a mapping")
    if payload.get("architecture") != V34_ARCHITECTURE:
        raise ValueError("V47 received a checkpoint with the wrong V34 architecture")
    if int(payload.get("update", -1)) != V47_CODEC_UPDATE:
        raise ValueError("V47 requires the V34 EMA state at update 6,000")
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("V47 V34 checkpoint lacks a complete EMA shadow")
    shadow = ema["shadow"]
    if not all(isinstance(value, torch.Tensor) for value in shadow.values()):
        raise TypeError("V47 V34 EMA shadow must contain only tensors")
    state_sha256 = tensor_state_sha256(shadow)
    if strict_digest and state_sha256 != V47_REQUIRED_CODEC_STATE_SHA256:
        raise ValueError("V47 V34 EMA tensor state differs from the frozen protocol")
    config_payload = payload.get("model_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("V47 V34 checkpoint lacks its model configuration")
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig(**config_payload))
    codec.load_state_dict(shadow, strict=True)
    codec.requires_grad_(False).eval()
    receipt = {
        "path": str(path),
        "checkpoint_sha256": checkpoint_sha256,
        "ema_tensor_state_sha256": state_sha256,
        "architecture": V34_ARCHITECTURE,
        "update": V47_CODEC_UPDATE,
        "selection": "ema-shadow",
        "parameters": sum(parameter.numel() for parameter in codec.parameters()),
        "trainable_parameters": 0,
    }
    return codec, receipt


@dataclass(frozen=True)
class CodecSphericalGlyphLanguageV47Config:
    cell_size: int = 32
    maximum_cells: int = 64
    field_dim: int = 768
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    noise_dim: int = 128
    generator_layers: int = 4
    generator_residual_scale: float = 1.0
    binary_threshold: float = 0.5
    initial_temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.cell_size != 32 or self.field_dim != 768:
            raise ValueError("V47 fixes 32x32 glyphs and a 768-wide codec sphere")
        if self.maximum_cells != 64:
            raise ValueError("V47 fixes a 64-cell visual context")
        if self.model_dim < 128 or self.layers < 1:
            raise ValueError("V47 causal model is underspecified")
        if self.heads < 1 or self.model_dim % self.heads:
            raise ValueError("V47 model width must divide into attention heads")
        if self.mlp_ratio < 2.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("V47 causal block settings are invalid")
        if self.noise_dim < 16 or self.generator_layers < 1:
            raise ValueError("V47 spherical generator is underspecified")
        if self.generator_residual_scale <= 0.0:
            raise ValueError("V47 generator residual scale must be positive")
        if not 0.0 < self.binary_threshold < 1.0:
            raise ValueError("V47 binary threshold must lie in (0,1)")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V47 initial temperature is invalid")


class CodecSphericalGlyphFieldV47(nn.Module):
    """Frozen V34 visual retina/actuator constrained to the unit sphere."""

    def __init__(
        self,
        codec: ContinuousGlyphCodec,
        *,
        binary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if codec.config.patch_size != 32 or codec.config.latent_width != 768:
            raise ValueError("V47 requires the frozen 32x32, 768-wide V34 codec")
        if not 0.0 < binary_threshold < 1.0:
            raise ValueError("V47 field threshold must lie in (0,1)")
        self.codec = codec.requires_grad_(False).eval()
        self.field_dim = 768
        self.binary_threshold = float(binary_threshold)
        self.decoder_radius = math.sqrt(self.field_dim)

    def train(self, mode: bool = True) -> CodecSphericalGlyphFieldV47:
        super().train(mode)
        self.codec.eval()
        return self

    def _validate_pixels(self, pixels: torch.Tensor) -> None:
        if not pixels.is_floating_point():
            raise TypeError("V47 field input must be floating point")
        if pixels.ndim < 4 or tuple(pixels.shape[-3:]) != (1, 32, 32):
            raise ValueError("V47 field input must end in [1,32,32]")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("V47 field input must be finite")
        if not bool(((pixels >= 0.0) & (pixels <= 1.0)).all()):
            raise ValueError("V47 field input must lie in [0,1]")

    def _validate_fields(self, fields: torch.Tensor) -> None:
        if not fields.is_floating_point():
            raise TypeError("V47 fields must be floating point")
        if fields.ndim < 2 or fields.shape[-1] != self.field_dim:
            raise ValueError("V47 fields must end in 768 coefficients")
        if not bool(torch.isfinite(fields).all()):
            raise ValueError("V47 fields must be finite")

    def normalize(self, fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(fields)
        return F.normalize(fields.float(), dim=-1, eps=1e-8)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self._validate_pixels(pixels)
        leading = pixels.shape[:-3]
        ink = (pixels.reshape(-1, 1, 32, 32) >= self.binary_threshold).float()
        white_positive = 1.0 - ink
        fields = self.codec.encode(white_positive)
        return F.normalize(fields.float(), dim=-1, eps=1e-8).reshape(
            *leading,
            self.field_dim,
        )

    def encode_unit(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)

    def ink_logits(self, unit_fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(unit_fields)
        leading = unit_fields.shape[:-1]
        normalized = F.normalize(
            unit_fields.reshape(-1, self.field_dim).float(),
            dim=-1,
            eps=1e-8,
        )
        white_logits = self.codec.decode(normalized * self.decoder_radius)
        return (-white_logits).reshape(*leading, 1, 32, 32)

    def probabilities(self, unit_fields: torch.Tensor) -> torch.Tensor:
        return self.ink_logits(unit_fields).sigmoid()

    def binary(self, unit_fields: torch.Tensor) -> torch.Tensor:
        probability = self.probabilities(unit_fields)
        return (probability >= self.binary_threshold).to(probability.dtype)

    def soft_reread(self, unit_fields: torch.Tensor) -> torch.Tensor:
        self._validate_fields(unit_fields)
        leading = unit_fields.shape[:-1]
        white_probability = 1.0 - self.probabilities(unit_fields)
        reread = self.codec.encode(white_probability.reshape(-1, 1, 32, 32))
        return F.normalize(reread.float(), dim=-1, eps=1e-8).reshape(
            *leading,
            self.field_dim,
        )

    def visible_reread(
        self,
        unit_fields: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.binary(unit_fields)
        return pixels, self.encode(pixels)


class ConditionalCodecSphericalGeneratorV47(nn.Module):
    def __init__(self, config: CodecSphericalGlyphLanguageV47Config) -> None:
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
            raise ValueError("V47 generator hidden state must be [N,model_dim]")
        if anchor.shape != (hidden.shape[0], self.config.field_dim):
            raise ValueError("V47 generator anchor does not align")
        if samples < 1 or noise_scale < 0.0:
            raise ValueError("V47 generator sampling settings are invalid")
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
        return F.normalize(
            repeated_anchor.float()
            + self.config.generator_residual_scale * residual,
            dim=-1,
            eps=1e-8,
        )


class CodecSphericalGlyphLanguageModelV47(nn.Module):
    """Causal raster language over a frozen, codebook-free glyph manifold."""

    def __init__(
        self,
        config: CodecSphericalGlyphLanguageV47Config,
        codec: ContinuousGlyphCodec,
        *,
        codec_checkpoint_sha256: str,
        codec_state_sha256: str,
    ) -> None:
        super().__init__()
        if len(codec_checkpoint_sha256) != 64 or len(codec_state_sha256) != 64:
            raise ValueError("V47 requires complete V34 codec digests")
        self.config = config
        self.codec_checkpoint_sha256 = codec_checkpoint_sha256
        self.codec_state_sha256 = codec_state_sha256
        self.field = CodecSphericalGlyphFieldV47(
            codec,
            binary_threshold=config.binary_threshold,
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
        self.generator = ConditionalCodecSphericalGeneratorV47(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )
        self._initialize_language()

    def train(self, mode: bool = True) -> CodecSphericalGlyphLanguageModelV47:
        super().train(mode)
        self.field.codec.eval()
        return self

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
            raise ValueError("V47 context must be [B,T,1,32,32]")
        if context.shape[1] < 1:
            raise ValueError("V47 context cannot be empty")
        if maximum and context.shape[1] > self.config.maximum_cells:
            raise ValueError("V47 context exceeds 64 visual cells")

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
        anchor = F.normalize(self.anchor_head(hidden).float(), dim=-1, eps=1e-8)
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
            raise ValueError("V47 pair contexts must be [B,2,64,1,32,32]")
        if candidates.ndim != 5 or tuple(candidates.shape[1:]) != (2, 1, 32, 32):
            raise ValueError("V47 pair candidates must be [B,2,1,32,32]")
        batch = contexts.shape[0]
        anchors = self.language(contexts.flatten(0, 1))["anchor_fields"][:, -1]
        anchors = anchors.reshape(batch, 2, self.config.field_dim)
        fields = self.field.encode(candidates.flatten(0, 1)).reshape(
            batch,
            2,
            self.config.field_dim,
        )
        return self.contrastive_scale.float() * torch.einsum(
            "bqd,bkd->bqk",
            anchors.float(),
            fields.float(),
        )

    @torch.no_grad()
    def score_image_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 4 or tuple(candidates.shape[1:]) != (1, 32, 32):
            raise ValueError("V47 evaluator candidates must be [N,1,32,32]")
        anchor = self.language(context)["anchor_fields"][:, -1]
        fields = self.field.encode(candidates)
        return self.contrastive_scale.float() * anchor @ fields.transpose(0, 1)

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
        proposals = self.sample_fields(
            hidden,
            anchor,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )
        candidate_pixels, reread_fields = self.field.visible_reread(proposals)
        scores = torch.einsum("bsd,bd->bs", reread_fields, anchor)
        proposal_reread_cosine = (proposals * reread_fields).sum(dim=-1)
        selected_indices = scores.argmax(dim=1)
        rows = torch.arange(len(proposals), device=proposals.device)
        pixels = candidate_pixels[rows, selected_indices]
        selected_proposals = proposals[rows, selected_indices]
        selected_rereads = reread_fields[rows, selected_indices]
        return pixels, {
            "sample_fields": proposals,
            "candidate_pixels": candidate_pixels,
            "reread_fields": reread_fields,
            "sample_scores": scores,
            "proposal_reread_cosine": proposal_reread_cosine,
            "selected_indices": selected_indices,
            "selected_proposal_fields": selected_proposals,
            "selected_fields": selected_rereads,
            "selected_reread_fields": selected_rereads,
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
            raise ValueError("V47 generation requires at least one new cell")
        sequence = prefix
        generated: list[torch.Tensor] = []
        reread_fields: list[torch.Tensor] = []
        proposal_fields: list[torch.Tensor] = []
        for _ in range(new_cells):
            context = sequence[:, -self.config.maximum_cells :]
            pixels, trace = self.sample_next(
                context,
                samples=samples,
                generator=generator,
                noise_scale=noise_scale,
            )
            generated.append(pixels)
            reread_fields.append(trace["selected_reread_fields"])
            proposal_fields.append(trace["selected_proposal_fields"])
            sequence = torch.cat((sequence, pixels[:, None]), dim=1)
        return sequence, {
            "generated_cells": torch.stack(generated, dim=1),
            "generated_fields": torch.stack(reread_fields, dim=1),
            "proposal_fields": torch.stack(proposal_fields, dim=1),
            "rereads_generated_pixels": torch.tensor(True, device=sequence.device),
        }


def codec_spherical_glyph_language_v47_config_payload(
    config: CodecSphericalGlyphLanguageV47Config,
) -> dict[str, Any]:
    return asdict(config)


def codec_spherical_glyph_language_v47_config_from_payload(
    payload: Mapping[str, Any],
) -> CodecSphericalGlyphLanguageV47Config:
    return CodecSphericalGlyphLanguageV47Config(**dict(payload))


def codec_spherical_glyph_language_v47_boundary_receipt(
    model: CodecSphericalGlyphLanguageModelV47,
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
        "architecture": V47_ARCHITECTURE,
        "config": codec_spherical_glyph_language_v47_config_payload(model.config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "field_parameters": sum(
            parameter.numel() for parameter in model.field.parameters()
        ),
        "field_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.field.parameters()
            if parameter.requires_grad
        ),
        "codec_checkpoint_sha256": model.codec_checkpoint_sha256,
        "codec_state_sha256": model.codec_state_sha256,
        "parameter_names_with_forbidden_fragments": suspicious,
        "language_parameters": list(inspect.signature(model.language).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "input_is_continuous_image_stream": True,
        "output_is_continuous_image_field": True,
        "output_is_direct_raster": True,
        "field_is_fixed_continuous_codec_sphere": True,
        "causal_over_visual_time": True,
        "selection_uses_visible_reread": True,
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


__all__ = [
    "CodecSphericalGlyphFieldV47",
    "CodecSphericalGlyphLanguageModelV47",
    "CodecSphericalGlyphLanguageV47Config",
    "ConditionalCodecSphericalGeneratorV47",
    "V47_ARCHITECTURE",
    "V47_CODEC_UPDATE",
    "V47_PROTOCOL",
    "V47_REQUIRED_CODEC_CHECKPOINT_SHA256",
    "V47_REQUIRED_CODEC_STATE_SHA256",
    "codec_spherical_glyph_language_v47_boundary_receipt",
    "codec_spherical_glyph_language_v47_config_from_payload",
    "codec_spherical_glyph_language_v47_config_payload",
    "load_verified_v34_codec",
    "tensor_state_sha256",
]

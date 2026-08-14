from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_glyph_codec import (
    V34_ARCHITECTURE,
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from .continuous_glyph_codec_data import file_sha256


V40_ARCHITECTURE = "continuous-glyph-content-form-v40"
V40_CODEC_CHECKPOINT_SHA256 = (
    "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"
)


@dataclass(frozen=True)
class GlyphContentFormConfig:
    patch_size: int = 32
    surface_width: int = 768
    content_width: int = 192
    form_width: int = 64
    encoder_width: int = 512
    form_encoder_width: int = 256
    synthesis_width: int = 512
    synthesis_depth: int = 3
    content_temperature: float = 0.07
    codec_channels: tuple[int, int, int, int] = (32, 64, 96, 192)
    codec_group_norm_groups: int = 8

    def __post_init__(self) -> None:
        if self.patch_size != 32 or self.surface_width != 768:
            raise ValueError("V40 requires the qualified V34 raster geometry")
        if not 16 <= self.content_width < self.surface_width:
            raise ValueError("V40 content width is invalid")
        if not 8 <= self.form_width < self.content_width:
            raise ValueError("V40 form width is invalid")
        if min(
            self.encoder_width,
            self.form_encoder_width,
            self.synthesis_width,
        ) < 32:
            raise ValueError("V40 hidden widths must be at least 32")
        if self.synthesis_depth < 1:
            raise ValueError("V40 synthesis depth must be positive")
        if not 0.01 <= self.content_temperature <= 1.0:
            raise ValueError("V40 contrastive temperature is invalid")


@dataclass
class GlyphContentFormOutput:
    anchor_surface: torch.Tensor
    positive_surface: torch.Tensor
    anchor_content: torch.Tensor
    positive_content: torch.Tensor
    anchor_form: torch.Tensor
    positive_form: torch.Tensor
    anchor_reference_form: torch.Tensor
    positive_reference_form: torch.Tensor
    anchor_self_surface: torch.Tensor
    positive_self_surface: torch.Tensor
    anchor_cross_surface: torch.Tensor
    positive_cross_surface: torch.Tensor
    anchor_reference_surface: torch.Tensor
    positive_reference_surface: torch.Tensor


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width, elementwise_affine=False)
        self.layers = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(self.normalization(inputs))


class NormalizedProjection(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_width, elementwise_affine=False),
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(inputs).float(), dim=-1).to(inputs.dtype)


class ContentFormSynthesizer(nn.Module):
    def __init__(self, config: GlyphContentFormConfig) -> None:
        super().__init__()
        self.surface_width = config.surface_width
        self.input_projection = nn.Linear(
            config.content_width + config.form_width,
            config.synthesis_width,
        )
        self.blocks = nn.Sequential(
            *(ResidualMLPBlock(config.synthesis_width) for _ in range(config.synthesis_depth))
        )
        self.output_normalization = nn.LayerNorm(
            config.synthesis_width,
            elementwise_affine=False,
        )
        self.output_projection = nn.Linear(
            config.synthesis_width,
            config.surface_width,
        )

    def forward(
        self,
        content_states: torch.Tensor,
        form_states: torch.Tensor,
    ) -> torch.Tensor:
        if content_states.shape[:-1] != form_states.shape[:-1]:
            raise ValueError("V40 content and form states must align")
        hidden = self.input_projection(torch.cat((content_states, form_states), dim=-1))
        hidden = self.blocks(hidden)
        surface = self.output_projection(self.output_normalization(hidden))
        return F.layer_norm(surface.float(), (self.surface_width,)).to(surface.dtype)


class GlyphContentFormModel(nn.Module):
    def __init__(self, config: GlyphContentFormConfig) -> None:
        super().__init__()
        self.config = config
        self.codec = ContinuousGlyphCodec(
            ContinuousGlyphCodecConfig(
                patch_size=config.patch_size,
                latent_width=config.surface_width,
                channels=config.codec_channels,
                group_norm_groups=config.codec_group_norm_groups,
            )
        )
        self.codec.requires_grad_(False)
        self.codec.eval()
        self.content_encoder = NormalizedProjection(
            config.surface_width,
            config.encoder_width,
            config.content_width,
        )
        self.form_encoder = NormalizedProjection(
            config.surface_width,
            config.form_encoder_width,
            config.form_width,
        )
        self.synthesizer = ContentFormSynthesizer(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.content_temperature))
        )

    def train(self, mode: bool = True) -> GlyphContentFormModel:
        super().train(mode)
        self.codec.eval()
        return self

    def _validate_pixels(self, pixels: torch.Tensor) -> None:
        expected = (1, self.config.patch_size, self.config.patch_size)
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != expected:
            raise ValueError("V40 pixels must have shape [B,1,32,32]")
        if not pixels.is_floating_point():
            raise TypeError("V40 pixels must be floating point")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("V40 pixels must be finite")
        if not bool(((pixels >= 0.0) & (pixels <= 1.0)).all()):
            raise ValueError("V40 pixels must be in [0,1]")

    def encode_surface(self, pixels: torch.Tensor) -> torch.Tensor:
        self._validate_pixels(pixels)
        with torch.no_grad():
            return self.codec.encode(pixels)

    def encode_content_from_surface(self, surfaces: torch.Tensor) -> torch.Tensor:
        if surfaces.ndim != 2 or surfaces.shape[-1] != self.config.surface_width:
            raise ValueError("V40 surface states must have shape [B,768]")
        return self.content_encoder(surfaces)

    def encode_form_from_surface(self, surfaces: torch.Tensor) -> torch.Tensor:
        if surfaces.ndim != 2 or surfaces.shape[-1] != self.config.surface_width:
            raise ValueError("V40 surface states must have shape [B,768]")
        return self.form_encoder(surfaces)

    def encode_content(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode_content_from_surface(self.encode_surface(pixels))

    def encode_form(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode_form_from_surface(self.encode_surface(pixels))

    def synthesize_surface(
        self,
        content_states: torch.Tensor,
        form_states: torch.Tensor,
    ) -> torch.Tensor:
        if content_states.shape[-1] != self.config.content_width:
            raise ValueError("V40 content states have the wrong width")
        if form_states.shape[-1] != self.config.form_width:
            raise ValueError("V40 form states have the wrong width")
        return self.synthesizer(content_states, form_states)

    def decode_surface(self, surfaces: torch.Tensor) -> torch.Tensor:
        if surfaces.ndim != 2 or surfaces.shape[-1] != self.config.surface_width:
            raise ValueError("V40 decoded surfaces must have shape [B,768]")
        return self.codec.decode(surfaces)

    def render(
        self,
        content_states: torch.Tensor,
        form_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_surface(self.synthesize_surface(content_states, form_states))

    def content_similarity_logits(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("V40 content pairs must be aligned matrices")
        if left.shape[-1] != self.config.content_width:
            raise ValueError("V40 content pairs have the wrong width")
        scale = self.logit_scale.float().clamp(max=math.log(100.0)).exp()
        return scale * (F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T)

    def forward(
        self,
        anchor_pixels: torch.Tensor,
        positive_pixels: torch.Tensor,
        anchor_style_pixels: torch.Tensor,
        positive_style_pixels: torch.Tensor,
    ) -> GlyphContentFormOutput:
        inputs = (
            anchor_pixels,
            positive_pixels,
            anchor_style_pixels,
            positive_style_pixels,
        )
        if len({tuple(value.shape) for value in inputs}) != 1:
            raise ValueError("V40 paired raster batches must align")
        for value in inputs:
            self._validate_pixels(value)
        batch = anchor_pixels.shape[0]
        surfaces = self.encode_surface(torch.cat(inputs, dim=0))
        anchor_surface, positive_surface, anchor_style_surface, positive_style_surface = (
            surfaces.split(batch)
        )
        anchor_content = self.encode_content_from_surface(anchor_surface)
        positive_content = self.encode_content_from_surface(positive_surface)
        anchor_form = self.encode_form_from_surface(anchor_surface)
        positive_form = self.encode_form_from_surface(positive_surface)
        anchor_reference_form = self.encode_form_from_surface(anchor_style_surface)
        positive_reference_form = self.encode_form_from_surface(positive_style_surface)
        return GlyphContentFormOutput(
            anchor_surface=anchor_surface,
            positive_surface=positive_surface,
            anchor_content=anchor_content,
            positive_content=positive_content,
            anchor_form=anchor_form,
            positive_form=positive_form,
            anchor_reference_form=anchor_reference_form,
            positive_reference_form=positive_reference_form,
            anchor_self_surface=self.synthesize_surface(anchor_content, anchor_form),
            positive_self_surface=self.synthesize_surface(positive_content, positive_form),
            anchor_cross_surface=self.synthesize_surface(positive_content, anchor_form),
            positive_cross_surface=self.synthesize_surface(anchor_content, positive_form),
            anchor_reference_surface=self.synthesize_surface(
                anchor_content,
                anchor_reference_form,
            ),
            positive_reference_surface=self.synthesize_surface(
                positive_content,
                positive_reference_form,
            ),
        )


def load_v40_v34_codec(
    model: GlyphContentFormModel,
    checkpoint_path: str | Path,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V40 V34 checkpoint does not exist: {path}")
    checkpoint_hash = file_sha256(path)
    if verify_hash and checkpoint_hash != V40_CODEC_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected V34 checkpoint SHA-256: {checkpoint_hash}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V40 V34 checkpoint must be a mapping")
    if checkpoint.get("architecture") != V34_ARCHITECTURE:
        raise ValueError("V40 received a checkpoint with the wrong codec architecture")
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("V40 V34 checkpoint lacks a complete EMA state")
    model.codec.load_state_dict(ema["shadow"], strict=True)
    model.codec.requires_grad_(False).eval()
    return {
        "path": str(path),
        "sha256": checkpoint_hash,
        "architecture": checkpoint.get("architecture"),
        "update": int(checkpoint.get("update", 0)),
        "selection": "ema-shadow",
    }


def glyph_content_form_boundary_receipt(
    model: GlyphContentFormModel,
) -> dict[str, Any]:
    forbidden = (
        "token",
        "vocab",
        "unicode",
        "character_id",
        "codebook",
        "quant",
        "ocr",
        "retrieval",
    )
    suspicious = sorted(
        name.lower()
        for name, _ in model.named_parameters()
        if any(fragment in name.lower() for fragment in forbidden)
    )
    return {
        "architecture": V40_ARCHITECTURE,
        "config": asdict(model.config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "codec_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.codec.parameters()
            if parameter.requires_grad
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "forward_parameters": list(inspect.signature(model.forward).parameters),
        "render_parameters": list(inspect.signature(model.render).parameters),
        "primary_input": "paired binary writing raster patches",
        "content_latent": "continuous cross-form geometry",
        "form_latent": "continuous visible rendering condition",
        "primary_output": "direct writing raster logits through frozen V34",
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_embedding_table": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_quantization": False,
        "uses_runtime_teacher": False,
    }


__all__ = [
    "GlyphContentFormConfig",
    "GlyphContentFormModel",
    "GlyphContentFormOutput",
    "V40_ARCHITECTURE",
    "V40_CODEC_CHECKPOINT_SHA256",
    "glyph_content_form_boundary_receipt",
    "load_v40_v34_codec",
]

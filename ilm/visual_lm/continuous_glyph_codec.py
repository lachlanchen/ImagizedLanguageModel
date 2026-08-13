from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


V34_ARCHITECTURE = "continuous-glyph-representation-codec-v34"


@dataclass(frozen=True)
class ContinuousGlyphCodecConfig:
    patch_size: int = 32
    latent_width: int = 768
    channels: tuple[int, int, int, int] = (32, 64, 96, 192)
    group_norm_groups: int = 8
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.patch_size != 32:
            raise ValueError("V34 fixes 32-pixel glyph patches")
        if self.latent_width != 768:
            raise ValueError("V34 fixes a 768-dimensional continuous latent")
        if len(self.channels) != 4 or any(channel < 8 for channel in self.channels):
            raise ValueError("V34 requires four positive convolutional scales")
        if any(channel % self.group_norm_groups for channel in self.channels):
            raise ValueError("V34 channels must divide into GroupNorm groups")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("V34 layer-normalization epsilon must be positive")


@dataclass
class ContinuousGlyphCodecOutput:
    logits: torch.Tensor
    latents: torch.Tensor
    decoder_latents: torch.Tensor

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.float().sigmoid()


class ResidualGlyphBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class ContinuousGlyphEncoder(nn.Module):
    def __init__(self, config: ContinuousGlyphCodecConfig) -> None:
        super().__init__()
        c1, c2, c3, c4 = config.channels
        groups = config.group_norm_groups
        self.stem = nn.Conv2d(1, c1, 3, padding=1)
        self.scale1 = ResidualGlyphBlock(c1, groups)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.scale2 = ResidualGlyphBlock(c2, groups)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.scale3 = ResidualGlyphBlock(c3, groups)
        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)
        self.scale4 = ResidualGlyphBlock(c4, groups)
        self.projection = nn.Linear(c4 * 4 * 4, config.latent_width)
        self.normalization = nn.LayerNorm(
            config.latent_width,
            eps=config.layer_norm_epsilon,
            elementwise_affine=False,
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.scale1(self.stem(pixels))
        hidden = self.scale2(self.down1(hidden))
        hidden = self.scale3(self.down2(hidden))
        hidden = self.scale4(self.down3(hidden))
        hidden = hidden.flatten(1)
        return self.normalization(self.projection(hidden))


class ContinuousGlyphDecoder(nn.Module):
    def __init__(self, config: ContinuousGlyphCodecConfig) -> None:
        super().__init__()
        c1, c2, c3, c4 = config.channels
        groups = config.group_norm_groups
        self.channels = c4
        self.projection = nn.Linear(config.latent_width, c4 * 4 * 4)
        self.scale4 = ResidualGlyphBlock(c4, groups)
        self.up3 = nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1)
        self.scale3 = ResidualGlyphBlock(c3, groups)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.scale2 = ResidualGlyphBlock(c2, groups)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.scale1 = ResidualGlyphBlock(c1, groups)
        self.output = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(latents).reshape(
            latents.shape[0],
            self.channels,
            4,
            4,
        )
        hidden = self.scale4(hidden)
        hidden = self.scale3(self.up3(hidden))
        hidden = self.scale2(self.up2(hidden))
        hidden = self.scale1(self.up1(hidden))
        return self.output(F.silu(hidden))


class ContinuousGlyphCodec(nn.Module):
    def __init__(self, config: ContinuousGlyphCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ContinuousGlyphEncoder(config)
        self.decoder = ContinuousGlyphDecoder(config)

    def _validate_pixels(self, pixels: torch.Tensor) -> None:
        expected = (1, self.config.patch_size, self.config.patch_size)
        if pixels.ndim != 4 or pixels.shape[1:] != expected:
            raise ValueError("V34 pixels must have shape [B,1,32,32]")
        if not pixels.is_floating_point():
            raise TypeError("V34 pixels must be floating point")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("V34 pixels must be finite")
        if not bool(((pixels >= 0.0) & (pixels <= 1.0)).all()):
            raise ValueError("V34 pixels must be in [0,1]")

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 2 or latents.shape[1] != self.config.latent_width:
            raise ValueError("V34 latents must have shape [B,768]")
        if not latents.is_floating_point() or not bool(torch.isfinite(latents).all()):
            raise ValueError("V34 latents must be finite floating-point values")

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self._validate_pixels(pixels)
        normalized = pixels.clamp(0, 1).mul(2.0).sub(1.0)
        return self.encoder(normalized)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_latents(latents)
        return self.decoder(latents).float()

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        latent_noise: torch.Tensor | None = None,
    ) -> ContinuousGlyphCodecOutput:
        latents = self.encode(pixels)
        decoder_latents = latents
        if latent_noise is not None:
            if latent_noise.shape != latents.shape:
                raise ValueError("V34 latent noise must align with encoded patches")
            if not latent_noise.is_floating_point() or not bool(
                torch.isfinite(latent_noise).all()
            ):
                raise ValueError("V34 latent noise must be finite floating point")
            decoder_latents = latents + latent_noise
        logits = self.decode(decoder_latents)
        return ContinuousGlyphCodecOutput(
            logits=logits,
            latents=latents,
            decoder_latents=decoder_latents,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        pixels: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("V34 reconstruction threshold must be in [0,1]")
        was_training = self.training
        self.eval()
        try:
            return (self(pixels).probabilities >= threshold).to(pixels.dtype)
        finally:
            self.train(was_training)


def continuous_glyph_codec_boundary_receipt(
    model: ContinuousGlyphCodec,
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
    parameter_names = [name.lower() for name, _ in model.named_parameters()]
    suspicious = sorted(
        name
        for name in parameter_names
        if any(fragment in name for fragment in forbidden)
    )
    return {
        "architecture": V34_ARCHITECTURE,
        "config": asdict(model.config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "encode_parameters": list(inspect.signature(model.encode).parameters),
        "decode_parameters": list(inspect.signature(model.decode).parameters),
        "reconstruct_parameters": list(
            inspect.signature(model.reconstruct).parameters
        ),
        "primary_input": "binary writing raster patches",
        "latent": "normalized continuous vectors",
        "primary_output": "direct writing raster logits",
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_quantization": False,
        "uses_runtime_teacher": False,
    }

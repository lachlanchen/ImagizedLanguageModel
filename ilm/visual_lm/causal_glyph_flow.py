from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel

from .continuous_glyph_codec import (
    V34_ARCHITECTURE,
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from .direct_visual_patch_lm import (
    DirectVisualPatchConfig,
    DirectVisualPatchLM,
    file_sha256,
    load_pixar_initialization,
)


V35_ARCHITECTURE = "causal-glyph-flow-v35"
V35_CODEC_CHECKPOINT_SHA256 = (
    "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"
)


@dataclass(frozen=True)
class CausalGlyphFlowConfig:
    patch_size: int = 32
    maximum_patches: int = 96
    latent_width: int = 768
    hidden_size: int = 768
    layers: int = 12
    attention_heads: int = 12
    key_value_heads: int = 12
    intermediate_size: int = 3072
    flow_width: int = 512
    flow_depth: int = 3
    rms_norm_epsilon: float = 1e-5
    rope_theta: float = 10_000.0
    attention_dropout: float = 0.0
    codec_channels: tuple[int, int, int, int] = (32, 64, 96, 192)
    codec_group_norm_groups: int = 8

    def __post_init__(self) -> None:
        if self.patch_size != 32:
            raise ValueError("V35 fixes 32-pixel retinal patches")
        if not 4 <= self.maximum_patches <= 1024:
            raise ValueError("V35 maximum patch count is invalid")
        if self.latent_width != 768:
            raise ValueError("V35 fixes the qualified 768-dimensional V34 latent")
        if self.hidden_size < 32 or self.hidden_size % self.attention_heads:
            raise ValueError("V35 hidden size must divide into attention heads")
        if not 1 <= self.key_value_heads <= self.attention_heads:
            raise ValueError("V35 key/value head count is invalid")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("V35 attention heads must divide into key/value heads")
        if self.layers < 1 or self.intermediate_size < self.hidden_size:
            raise ValueError("V35 causal-field dimensions are invalid")
        if self.flow_width < 32 or self.flow_depth < 1:
            raise ValueError("V35 flow-head dimensions are invalid")
        if self.rms_norm_epsilon <= 0 or self.rope_theta <= 0:
            raise ValueError("V35 normalization and RoPE values must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("V35 attention dropout must be in [0,1)")

    @property
    def production_shape(self) -> bool:
        return (
            self.patch_size == 32
            and self.maximum_patches == 96
            and self.latent_width == 768
            and self.hidden_size == 768
            and self.layers == 12
            and self.attention_heads == 12
            and self.key_value_heads == 12
            and self.intermediate_size == 3072
            and self.flow_width == 512
            and self.flow_depth == 3
            and self.codec_channels == (32, 64, 96, 192)
            and self.codec_group_norm_groups == 8
        )


@dataclass
class CausalGlyphFlowOutput:
    latents: torch.Tensor
    input_states: torch.Tensor
    hidden_states: torch.Tensor
    anchor_latents: torch.Tensor
    stop_logits: torch.Tensor


@dataclass
class CausalGlyphFlowGeneration:
    patches: torch.Tensor
    patch_mask: torch.Tensor
    feedback_latents: torch.Tensor
    lengths: torch.Tensor
    stop_probabilities: torch.Tensor

    def strips(self) -> torch.Tensor:
        if self.patches.ndim != 5:
            raise ValueError("V35 generated patches must be [B,L,1,32,32]")
        batch, count, channels, height, width = self.patches.shape
        return self.patches.permute(0, 2, 3, 1, 4).reshape(
            batch,
            channels,
            height,
            count * width,
        )


class ResidualCoordinateAdapter(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.skip = (
            nn.Identity()
            if input_width == output_width
            else nn.Linear(input_width, output_width, bias=False)
        )
        self.residual = nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.SiLU(),
            nn.Linear(output_width, output_width),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.skip(inputs) + self.residual(inputs)


class LatentAnchorHead(nn.Module):
    def __init__(self, hidden_size: int, latent_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, latent_width),
            nn.SiLU(),
            nn.Linear(latent_width, latent_width),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latents = self.layers(hidden_states)
        return F.layer_norm(latents.float(), (latents.shape[-1],)).to(latents.dtype)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, width: int, frequency_width: int = 256) -> None:
        super().__init__()
        if frequency_width < 2:
            raise ValueError("V35 time embedding requires at least two frequencies")
        self.frequency_width = frequency_width
        self.mlp = nn.Sequential(
            nn.Linear(frequency_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    @staticmethod
    def fourier_features(times: torch.Tensor, width: int) -> torch.Tensor:
        if times.ndim != 1:
            raise ValueError("V35 flow times must be a vector")
        half = width // 2
        frequencies = torch.exp(
            math.log(1_000.0)
            * torch.arange(half, device=times.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        angles = 2.0 * math.pi * times.float()[:, None] * frequencies[None]
        features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if width % 2:
            features = torch.cat((features, torch.zeros_like(features[:, :1])), dim=-1)
        return features

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier_features(times, self.frequency_width))


def _modulate(inputs: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return inputs * (1.0 + scale) + shift


class FlowResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 3 * width),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        hidden = _modulate(self.normalization(inputs), shift, scale)
        return inputs + gate * self.mlp(hidden)


class ConditionalGlyphFlowHead(nn.Module):
    def __init__(
        self,
        *,
        latent_width: int,
        condition_width: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.latent_width = latent_width
        self.input_projection = nn.Linear(latent_width, width)
        self.condition_projection = nn.Linear(condition_width, width)
        self.time_embedding = ContinuousTimeEmbedding(width)
        self.blocks = nn.ModuleList(FlowResidualBlock(width) for _ in range(depth))
        self.final_normalization = nn.LayerNorm(
            width,
            eps=1e-6,
            elementwise_affine=False,
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )
        self.output_projection = nn.Linear(width, latent_width)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latents.ndim != 2 or noisy_latents.shape[1] != self.latent_width:
            raise ValueError("V35 flow inputs must have shape [N,768]")
        if conditions.ndim != 2 or conditions.shape[0] != noisy_latents.shape[0]:
            raise ValueError("V35 flow conditions must align with flow inputs")
        if times.shape != (noisy_latents.shape[0],):
            raise ValueError("V35 flow times must align with flow inputs")
        condition = self.condition_projection(conditions) + self.time_embedding(times)
        hidden = self.input_projection(noisy_latents)
        for block in self.blocks:
            hidden = block(hidden, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        hidden = _modulate(self.final_normalization(hidden), shift, scale)
        return self.output_projection(hidden)


class CausalGlyphFlowLM(nn.Module):
    def __init__(self, config: CausalGlyphFlowConfig) -> None:
        super().__init__()
        self.config = config
        codec_config = ContinuousGlyphCodecConfig(
            patch_size=config.patch_size,
            latent_width=config.latent_width,
            channels=config.codec_channels,
            group_norm_groups=config.codec_group_norm_groups,
        )
        self.codec = ContinuousGlyphCodec(codec_config)
        self.codec.requires_grad_(False)
        self.input_adapter = ResidualCoordinateAdapter(
            config.latent_width,
            config.hidden_size,
        )
        backbone_config = LlamaConfig(
            vocab_size=1,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.layers,
            num_attention_heads=config.attention_heads,
            num_key_value_heads=config.key_value_heads,
            hidden_act="silu",
            max_position_embeddings=config.maximum_patches,
            initializer_range=0.02,
            rms_norm_eps=config.rms_norm_epsilon,
            rope_theta=config.rope_theta,
            attention_bias=False,
            attention_dropout=config.attention_dropout,
            mlp_bias=False,
            tie_word_embeddings=False,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
        )
        self.backbone = LlamaModel(backbone_config)
        self.backbone.embed_tokens = None
        self.anchor_head = LatentAnchorHead(config.hidden_size, config.latent_width)
        self.flow_head = ConditionalGlyphFlowHead(
            latent_width=config.latent_width,
            condition_width=config.hidden_size,
            width=config.flow_width,
            depth=config.flow_depth,
        )
        self.stop_head = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -4.0)

    def train(self, mode: bool = True) -> CausalGlyphFlowLM:
        super().train(mode)
        self.codec.eval()
        return self

    def _validate_inputs(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> int:
        if not pixels.is_floating_point() or not patch_mask.is_floating_point():
            raise TypeError("V35 pixels and patch mask must be floating point")
        if pixels.ndim != 4 or pixels.shape[1:3] != (
            1,
            self.config.patch_size,
        ):
            raise ValueError("V35 pixels must have shape [B,1,32,32*L]")
        if pixels.shape[-1] % self.config.patch_size:
            raise ValueError("V35 raster width must divide into complete patches")
        count = pixels.shape[-1] // self.config.patch_size
        if not 1 <= count <= self.config.maximum_patches:
            raise ValueError("V35 raster has an invalid patch count")
        if patch_mask.shape != (pixels.shape[0], count):
            raise ValueError("V35 patch mask does not align with raster patches")
        if not bool(torch.isfinite(pixels).all() and torch.isfinite(patch_mask).all()):
            raise ValueError("V35 inputs must be finite")
        if not bool(((pixels >= 0) & (pixels <= 1)).all()):
            raise ValueError("V35 pixels must be in [0,1]")
        if not bool(((patch_mask >= 0) & (patch_mask <= 1)).all()):
            raise ValueError("V35 patch mask must be in [0,1]")
        lengths = patch_mask.round().long().sum(dim=1)
        if not bool((lengths >= 1).all()):
            raise ValueError("V35 each sample requires at least one active patch")
        expected = (
            torch.arange(count, device=patch_mask.device)[None, :] < lengths[:, None]
        )
        if not bool(patch_mask.bool().eq(expected).all()):
            raise ValueError("V35 patch mask must be a contiguous active prefix")
        return count

    def patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.unfold(
            -1,
            self.config.patch_size,
            self.config.patch_size,
        ).permute(0, 3, 1, 2, 4).contiguous()

    def encode_patches(self, pixels: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(pixels)
        batch, count = patches.shape[:2]
        with torch.no_grad():
            latents = self.codec.encode(
                patches.reshape(-1, 1, self.config.patch_size, self.config.patch_size)
            )
        return latents.reshape(batch, count, self.config.latent_width)

    def causal_hidden(
        self,
        latents: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 3 or latents.shape[2] != self.config.latent_width:
            raise ValueError("V35 latent sequence must have shape [B,L,768]")
        if patch_mask.shape != latents.shape[:2]:
            raise ValueError("V35 latent mask does not align with latent sequence")
        input_states = self.input_adapter(latents)
        hidden = self.backbone(
            inputs_embeds=input_states,
            attention_mask=patch_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return input_states, hidden

    def forward(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> CausalGlyphFlowOutput:
        self._validate_inputs(pixels, patch_mask)
        latents = self.encode_patches(pixels)
        input_states, hidden = self.causal_hidden(latents, patch_mask)
        return CausalGlyphFlowOutput(
            latents=latents,
            input_states=input_states,
            hidden_states=hidden,
            anchor_latents=self.anchor_head(hidden),
            stop_logits=self.stop_head(hidden).squeeze(-1).float(),
        )

    def flow_velocity(
        self,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow_head(noisy_latents, times, conditions)

    def normalized_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] != self.config.latent_width:
            raise ValueError("V35 normalized latents require width 768")
        return F.layer_norm(latents.float(), (self.config.latent_width,)).to(
            latents.dtype
        )

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        shape = latents.shape[:-1]
        normalized = self.normalized_latents(latents).reshape(
            -1,
            self.config.latent_width,
        )
        logits = self.codec.decode(normalized)
        return logits.reshape(*shape, 1, self.config.patch_size, self.config.patch_size)

    def visible_feedback(
        self,
        latents: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("V35 raster threshold must be in [0,1]")
        shape = latents.shape[:-1]
        logits = self.decode_latents(latents)
        patches = (logits.float().sigmoid() >= threshold).to(latents.dtype)
        flat = patches.reshape(-1, 1, self.config.patch_size, self.config.patch_size)
        with torch.no_grad():
            feedback = self.codec.encode(flat)
        return patches, feedback.reshape(*shape, self.config.latent_width)

    def sample_flow(
        self,
        conditions: torch.Tensor,
        noise: torch.Tensor,
        *,
        steps: int = 8,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("V35 flow sampling requires at least one step")
        if noise.ndim != 2 or noise.shape != (
            conditions.shape[0],
            self.config.latent_width,
        ):
            raise ValueError("V35 flow noise does not align with conditions")
        state = noise
        step_size = 1.0 / steps
        for index in range(steps):
            time = state.new_full((state.shape[0],), index / steps)
            first = self.flow_velocity(state, time, conditions)
            proposal = state + step_size * first
            if index + 1 == steps:
                state = proposal
            else:
                next_time = state.new_full((state.shape[0],), (index + 1) / steps)
                second = self.flow_velocity(proposal, next_time, conditions)
                state = state + 0.5 * step_size * (first + second)
        return self.normalized_latents(state)

    @torch.no_grad()
    def generate(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
        *,
        maximum_new_patches: int = 31,
        minimum_new_patches: int = 1,
        stop_threshold: float = 0.5,
        raster_threshold: float = 0.5,
        flow_steps: int = 8,
        seed: int = 20_263_500,
        use_flow: bool = False,
    ) -> CausalGlyphFlowGeneration:
        self._validate_inputs(pixels, patch_mask)
        if not 1 <= maximum_new_patches <= self.config.maximum_patches:
            raise ValueError("V35 maximum generation length is invalid")
        if not 0 <= minimum_new_patches <= maximum_new_patches:
            raise ValueError("V35 minimum generation length is invalid")
        if not 0.0 <= stop_threshold <= 1.0:
            raise ValueError("V35 stop threshold must be in [0,1]")
        if not 0.0 <= raster_threshold <= 1.0:
            raise ValueError("V35 raster threshold must be in [0,1]")
        if flow_steps < 1:
            raise ValueError("V35 flow sampling requires at least one step")

        batch = pixels.shape[0]
        prompt_lengths = patch_mask.round().long().sum(dim=1)
        if bool((prompt_lengths + maximum_new_patches > self.config.maximum_patches).any()):
            raise ValueError("V35 prompt plus generation exceeds the causal context")
        maximum_canvas = int((prompt_lengths.max() + maximum_new_patches).item())
        source_latents = self.encode_patches(pixels)
        latent_canvas = source_latents.new_zeros(
            batch,
            maximum_canvas,
            self.config.latent_width,
        )
        canvas_mask = patch_mask.new_zeros(batch, maximum_canvas)
        for sample in range(batch):
            length = int(prompt_lengths[sample])
            latent_canvas[sample, :length] = source_latents[sample, :length]
            canvas_mask[sample, :length] = 1.0

        generated = pixels.new_ones(
            batch,
            maximum_new_patches,
            1,
            self.config.patch_size,
            self.config.patch_size,
        )
        generated_mask = patch_mask.new_zeros(batch, maximum_new_patches)
        feedback_latents = source_latents.new_zeros(
            batch,
            maximum_new_patches,
            self.config.latent_width,
        )
        stop_probabilities = pixels.new_zeros(batch, maximum_new_patches + 1)
        generated_lengths = torch.zeros(batch, dtype=torch.long, device=pixels.device)
        current_lengths = prompt_lengths.clone()
        active = torch.ones(batch, dtype=torch.bool, device=pixels.device)
        generator = torch.Generator(device=pixels.device)
        generator.manual_seed(int(seed))
        was_training = self.training
        self.eval()

        try:
            for _ in range(maximum_new_patches + 1):
                visible = int(current_lengths.max())
                _, hidden = self.causal_hidden(
                    latent_canvas[:, :visible],
                    canvas_mask[:, :visible],
                )
                rows = torch.arange(batch, device=pixels.device)
                last = current_lengths - 1
                condition = hidden[rows, last]
                stop = self.stop_head(condition).squeeze(-1).sigmoid()
                stop_probabilities[
                    rows,
                    generated_lengths.clamp_max(maximum_new_patches),
                ] = stop
                may_stop = generated_lengths >= minimum_new_patches
                active = active & ~(may_stop & (stop >= stop_threshold))
                active = active & (generated_lengths < maximum_new_patches)
                if not bool(active.any()):
                    break

                if use_flow:
                    noise = torch.randn(
                        batch,
                        self.config.latent_width,
                        device=pixels.device,
                        dtype=condition.dtype,
                        generator=generator,
                    )
                    predicted = self.sample_flow(condition, noise, steps=flow_steps)
                else:
                    predicted = self.anchor_head(condition)
                next_patches, next_feedback = self.visible_feedback(
                    predicted,
                    threshold=raster_threshold,
                )
                for sample in active.nonzero(as_tuple=False).flatten().tolist():
                    generation_index = int(generated_lengths[sample])
                    canvas_index = int(current_lengths[sample])
                    generated[sample, generation_index] = next_patches[sample]
                    generated_mask[sample, generation_index] = 1.0
                    feedback_latents[sample, generation_index] = next_feedback[sample]
                    latent_canvas[sample, canvas_index] = next_feedback[sample]
                    canvas_mask[sample, canvas_index] = 1.0
                    generated_lengths[sample] += 1
                    current_lengths[sample] += 1
        finally:
            self.train(was_training)

        return CausalGlyphFlowGeneration(
            patches=generated,
            patch_mask=generated_mask,
            feedback_latents=feedback_latents,
            lengths=generated_lengths,
            stop_probabilities=stop_probabilities,
        )


def load_v34_ema_codec(
    codec: ContinuousGlyphCodec,
    checkpoint_path: str | Path,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V35 V34 checkpoint does not exist: {path}")
    checkpoint_hash = file_sha256(path)
    if verify_hash and checkpoint_hash != V35_CODEC_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected V34 checkpoint SHA-256: {checkpoint_hash}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V35 V34 checkpoint must be a mapping")
    if checkpoint.get("architecture") != V34_ARCHITECTURE:
        raise ValueError("V35 received a checkpoint with the wrong codec architecture")
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("V35 V34 checkpoint lacks a complete EMA state")
    codec.load_state_dict(ema["shadow"], strict=True)
    codec.requires_grad_(False).eval()
    return {
        "path": str(path),
        "sha256": checkpoint_hash,
        "architecture": checkpoint.get("architecture"),
        "update": int(checkpoint.get("update", 0)),
        "selection": "ema-shadow",
        "tensors": len(ema["shadow"]),
    }


def load_pixar_causal_foundation(
    model: CausalGlyphFlowLM,
    checkpoint_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], nn.Conv2d]:
    if not model.config.production_shape:
        raise ValueError("V35 PIXAR initialization requires the production shape")
    source = DirectVisualPatchLM(DirectVisualPatchConfig())
    receipt = load_pixar_initialization(
        source,
        checkpoint_path,
        verify_hashes=verify_hashes,
    )
    model.backbone.load_state_dict(source.backbone.state_dict(), strict=True)
    projection = nn.Conv2d(
        1,
        model.config.hidden_size,
        kernel_size=model.config.patch_size,
        stride=model.config.patch_size,
        bias=False,
    )
    projection.load_state_dict(source.input_projection.state_dict(), strict=True)
    projection.requires_grad_(False).eval()
    del source
    return receipt, projection


def causal_glyph_flow_boundary_receipt(model: CausalGlyphFlowLM) -> dict[str, Any]:
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
        "architecture": V35_ARCHITECTURE,
        "config": asdict(model.config),
        "production_shape": model.config.production_shape,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "codec_parameters": sum(parameter.numel() for parameter in model.codec.parameters()),
        "codec_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.codec.parameters()
            if parameter.requires_grad
        ),
        "parameter_names_with_forbidden_fragments": suspicious,
        "forward_parameters": list(inspect.signature(model.forward).parameters),
        "generate_parameters": list(inspect.signature(model.generate).parameters),
        "primary_input": "binary writing raster strip and visual-position mask",
        "primary_output": "generated binary writing raster patches",
        "feedback_boundary": "decode-threshold-reencode-visible-raster",
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_embedding_table": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_quantization": False,
        "uses_retrieval": False,
        "uses_runtime_teacher": False,
    }


from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel


V33_ARCHITECTURE = "direct-visual-patch-lm-v33"
PIXAR_REPOSITORY = "https://github.com/april-tools/pixar"
PIXAR_REVISION = "810a423336d5fdeb33e4c1695381e357ff32c4bb"
PIXAR_CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    "1ngfKBmCL_nEa2om9ifJHOaf4SDKQ-eMP/view"
)
PIXAR_WEIGHT_SHA256 = (
    "ae4f899bbbb0bfaa90ee033c6d1dc5aeb3f50b323f726a0df241be8104682eb9"
)
PIXAR_CONFIG_SHA256 = (
    "1bc49b2e8cc59a4865d0f737739e73a55d14a6b849a5cae516de3dbb3f8e2ace"
)


@dataclass(frozen=True)
class DirectVisualPatchConfig:
    patch_size: int = 32
    maximum_patches: int = 96
    hidden_size: int = 768
    layers: int = 12
    attention_heads: int = 12
    key_value_heads: int = 12
    intermediate_size: int = 3072
    rms_norm_epsilon: float = 1e-5
    rope_theta: float = 10_000.0
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        if not 8 <= self.patch_size <= 64:
            raise ValueError("V33 patch size must be in [8,64]")
        if not 4 <= self.maximum_patches <= 1024:
            raise ValueError("V33 maximum patch count must be in [4,1024]")
        if self.hidden_size < 32 or self.hidden_size % self.attention_heads:
            raise ValueError("V33 hidden size must divide into attention heads")
        if not 1 <= self.key_value_heads <= self.attention_heads:
            raise ValueError("V33 key/value head count is invalid")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("V33 attention heads must divide into key/value heads")
        if self.layers < 1 or self.intermediate_size < self.hidden_size:
            raise ValueError("V33 transformer dimensions are invalid")
        if self.rms_norm_epsilon <= 0 or self.rope_theta <= 0:
            raise ValueError("V33 normalization and RoPE values must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("V33 attention dropout must be in [0,1)")

    @property
    def pixels_per_patch(self) -> int:
        return self.patch_size * self.patch_size

    @property
    def production_shape(self) -> bool:
        return (
            self.patch_size == 32
            and self.hidden_size == 768
            and self.layers == 12
            and self.attention_heads == 12
            and self.key_value_heads == 12
            and self.intermediate_size == 3072
        )


@dataclass
class DirectVisualPatchOutput:
    patch_logits: torch.Tensor
    stop_logits: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class DirectVisualPatchGeneration:
    patches: torch.Tensor
    patch_mask: torch.Tensor
    lengths: torch.Tensor
    stop_probabilities: torch.Tensor

    def strips(self) -> torch.Tensor:
        if self.patches.ndim != 5:
            raise ValueError("V33 generated patches must be [B,L,1,H,W]")
        batch, count, channels, height, width = self.patches.shape
        return self.patches.permute(0, 2, 3, 1, 4).reshape(
            batch,
            channels,
            height,
            count * width,
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_projection_kernel(
    weight: torch.Tensor,
    *,
    size: int,
    preserve_input_variance: bool,
) -> torch.Tensor:
    if weight.ndim != 4 or weight.shape[1] != 1:
        raise ValueError("PIXAR projection must have shape [hidden,1,H,W]")
    old_area = int(weight.shape[-2] * weight.shape[-1])
    resized = F.interpolate(
        weight.float(),
        size=(size, size),
        mode="bicubic",
        align_corners=False,
    )
    if preserve_input_variance:
        resized = resized * math.sqrt(old_area / float(size * size))
    return resized


class DirectVisualPatchLM(nn.Module):
    def __init__(self, config: DirectVisualPatchConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv2d(
            1,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
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
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.pixels_per_patch,
            bias=False,
        )
        self.stop_head = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -4.0)

    def _validate_inputs(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> None:
        if not pixels.is_floating_point() or not patch_mask.is_floating_point():
            raise TypeError("V33 pixels and patch mask must be floating point")
        if pixels.ndim != 4 or pixels.shape[1:3] != (
            1,
            self.config.patch_size,
        ):
            raise ValueError("V33 pixels must have shape [B,1,patch,patch*L]")
        if pixels.shape[-1] % self.config.patch_size:
            raise ValueError("V33 raster width must divide into complete patches")
        count = pixels.shape[-1] // self.config.patch_size
        if not 1 <= count <= self.config.maximum_patches:
            raise ValueError("V33 raster has an invalid patch count")
        if patch_mask.shape != (pixels.shape[0], count):
            raise ValueError("V33 patch mask does not align with raster patches")
        if not bool(torch.isfinite(pixels).all() and torch.isfinite(patch_mask).all()):
            raise ValueError("V33 inputs must be finite")
        if not bool(((patch_mask >= 0) & (patch_mask <= 1)).all()):
            raise ValueError("V33 patch mask must be in [0,1]")
        if not bool((patch_mask.sum(dim=1) >= 1).all()):
            raise ValueError("V33 each sample requires at least one active patch")

    def encode_patches(self, pixels: torch.Tensor) -> torch.Tensor:
        normalized = pixels.clamp(0, 1).mul(2.0).sub(1.0)
        embedded = self.input_projection(normalized)
        if embedded.shape[2] != 1:
            raise RuntimeError("V33 patch projection produced multiple raster rows")
        return embedded.squeeze(2).transpose(1, 2)

    def forward(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> DirectVisualPatchOutput:
        self._validate_inputs(pixels, patch_mask)
        embedded = self.encode_patches(pixels)
        hidden = self.backbone(
            inputs_embeds=embedded,
            attention_mask=patch_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        patch_logits = self.output_projection(hidden).reshape(
            pixels.shape[0],
            embedded.shape[1],
            1,
            self.config.patch_size,
            self.config.patch_size,
        )
        stop_logits = self.stop_head(hidden).squeeze(-1)
        return DirectVisualPatchOutput(
            patch_logits=patch_logits.float(),
            stop_logits=stop_logits.float(),
            hidden_states=hidden,
        )

    @torch.no_grad()
    def generate(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
        *,
        maximum_new_patches: int = 32,
        minimum_new_patches: int = 1,
        pixel_threshold: float = 0.5,
        stop_threshold: float = 0.5,
    ) -> DirectVisualPatchGeneration:
        self._validate_inputs(pixels, patch_mask)
        if not 1 <= maximum_new_patches <= self.config.maximum_patches:
            raise ValueError("V33 maximum generation length is invalid")
        if not 0 <= minimum_new_patches <= maximum_new_patches:
            raise ValueError("V33 minimum generation length is invalid")
        if not 0.0 <= pixel_threshold <= 1.0:
            raise ValueError("V33 pixel threshold must be in [0,1]")
        if not 0.0 <= stop_threshold <= 1.0:
            raise ValueError("V33 stop threshold must be in [0,1]")

        batch = pixels.shape[0]
        prompt_lengths = patch_mask.round().long().sum(dim=1)
        if bool((prompt_lengths + maximum_new_patches > self.config.maximum_patches).any()):
            raise ValueError("V33 prompt plus requested generation exceeds context")
        maximum_canvas = int((prompt_lengths.max() + maximum_new_patches).item())
        canvas = pixels.new_ones(
            batch,
            1,
            self.config.patch_size,
            maximum_canvas * self.config.patch_size,
        )
        canvas_mask = pixels.new_zeros(batch, maximum_canvas)
        source_patches = pixels.unfold(
            -1,
            self.config.patch_size,
            self.config.patch_size,
        ).permute(0, 3, 1, 2, 4)
        for sample in range(batch):
            length = int(prompt_lengths[sample].item())
            for index in range(length):
                left = index * self.config.patch_size
                canvas[sample, :, :, left : left + self.config.patch_size] = (
                    source_patches[sample, index]
                )
            canvas_mask[sample, :length] = 1.0

        generated = pixels.new_ones(
            batch,
            maximum_new_patches,
            1,
            self.config.patch_size,
            self.config.patch_size,
        )
        generated_mask = pixels.new_zeros(batch, maximum_new_patches)
        stop_probabilities = pixels.new_zeros(batch, maximum_new_patches + 1)
        generated_lengths = torch.zeros(batch, dtype=torch.long, device=pixels.device)
        current_lengths = prompt_lengths.clone()
        active = torch.ones(batch, dtype=torch.bool, device=pixels.device)
        was_training = self.training
        self.eval()

        try:
            for _ in range(maximum_new_patches + 1):
                visible = int(current_lengths.max().item())
                output = self(
                    canvas[..., : visible * self.config.patch_size],
                    canvas_mask[:, :visible],
                )
                row = torch.arange(batch, device=pixels.device)
                last = current_lengths - 1
                next_logits = output.patch_logits[row, last]
                stop = output.stop_logits[row, last].sigmoid()
                stop_probabilities[row, generated_lengths.clamp_max(maximum_new_patches)] = stop
                may_stop = generated_lengths >= minimum_new_patches
                active = active & ~(may_stop & (stop >= stop_threshold))
                active = active & (generated_lengths < maximum_new_patches)
                if not bool(active.any()):
                    break

                next_patch = (next_logits.sigmoid() >= pixel_threshold).to(pixels.dtype)
                for sample in active.nonzero(as_tuple=False).flatten().tolist():
                    generation_index = int(generated_lengths[sample].item())
                    canvas_index = int(current_lengths[sample].item())
                    generated[sample, generation_index] = next_patch[sample]
                    generated_mask[sample, generation_index] = 1.0
                    left = canvas_index * self.config.patch_size
                    canvas[sample, :, :, left : left + self.config.patch_size] = (
                        next_patch[sample]
                    )
                    canvas_mask[sample, canvas_index] = 1.0
                    generated_lengths[sample] += 1
                    current_lengths[sample] += 1
        finally:
            self.train(was_training)

        return DirectVisualPatchGeneration(
            patches=generated,
            patch_mask=generated_mask,
            lengths=generated_lengths,
            stop_probabilities=stop_probabilities,
        )


def resolve_pixar_checkpoint(path: str | Path) -> tuple[Path, Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.name != "pytorch_model.bin":
            raise ValueError("V33 PIXAR file must be named pytorch_model.bin")
        weight_path = root
        config_path = root.with_name("config.json")
    else:
        if (root / "backbone" / "pytorch_model.bin").exists():
            root = root / "backbone"
        weight_path = root / "pytorch_model.bin"
        config_path = root / "config.json"
    if not weight_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"V33 PIXAR checkpoint is incomplete under {root}")
    return weight_path, config_path


def load_pixar_initialization(
    model: DirectVisualPatchLM,
    checkpoint: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if not model.config.production_shape:
        raise ValueError("PIXAR initialization requires the production V33 transformer")
    weight_path, config_path = resolve_pixar_checkpoint(checkpoint)
    weight_hash = file_sha256(weight_path)
    config_hash = file_sha256(config_path)
    if verify_hashes and weight_hash != PIXAR_WEIGHT_SHA256:
        raise ValueError(f"unexpected PIXAR weight SHA-256: {weight_hash}")
    if verify_hashes and config_hash != PIXAR_CONFIG_SHA256:
        raise ValueError(f"unexpected PIXAR config SHA-256: {config_hash}")

    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("PIXAR checkpoint must contain a state dictionary")
    required_projection_keys = {"in_proj.weight", "out_proj.weight"}
    if not required_projection_keys.issubset(state):
        raise ValueError("PIXAR checkpoint lacks direct raster projections")
    core = {
        str(name): value
        for name, value in state.items()
        if name not in required_projection_keys
    }
    expected_core = set(model.backbone.state_dict())
    if set(core) != expected_core:
        missing = sorted(expected_core.difference(core))
        unexpected = sorted(set(core).difference(expected_core))
        raise ValueError(
            f"PIXAR transformer tensors differ: missing={missing}, unexpected={unexpected}"
        )
    model.backbone.load_state_dict(core, strict=True)

    old_input = state["in_proj.weight"].float()
    old_output = state["out_proj.weight"].float()
    resized_input = _resize_projection_kernel(
        old_input,
        size=model.config.patch_size,
        preserve_input_variance=True,
    )
    resized_output = _resize_projection_kernel(
        old_output,
        size=model.config.patch_size,
        preserve_input_variance=False,
    )
    with torch.no_grad():
        model.input_projection.weight.copy_(resized_input)
        model.output_projection.weight.copy_(
            resized_output.flatten(1).transpose(0, 1)
        )

    return {
        "repository": PIXAR_REPOSITORY,
        "revision": PIXAR_REVISION,
        "checkpoint_url": PIXAR_CHECKPOINT_URL,
        "weight_path": str(weight_path),
        "weight_sha256": weight_hash,
        "config_sha256": config_hash,
        "source_license": "MIT at audited revision",
        "weight_license": "not stated in checkpoint archive; local research use only",
        "core_tensors_loaded": len(core),
        "source_patch_shape": list(old_input.shape[-2:]),
        "target_patch_shape": [model.config.patch_size, model.config.patch_size],
        "input_norm_before": float(old_input.norm()),
        "input_norm_after": float(resized_input.norm()),
        "output_norm_before": float(old_output.norm()),
        "output_norm_after": float(resized_output.norm()),
    }


def direct_visual_patch_boundary_receipt(
    model: DirectVisualPatchLM,
) -> dict[str, Any]:
    parameter_names = tuple(name.lower() for name, _ in model.named_parameters())
    forbidden_fragments = (
        "embed_tokens",
        "lm_head",
        "tokenizer",
        "unicode",
        "character_id",
        "vocab",
        "codebook",
    )
    forbidden = sorted(
        name
        for name in parameter_names
        if any(fragment in name for fragment in forbidden_fragments)
    )
    generate_parameters = tuple(inspect.signature(model.generate).parameters)
    return {
        "architecture": V33_ARCHITECTURE,
        "config": asdict(model.config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_names_with_forbidden_fragments": forbidden,
        "generate_parameters": list(generate_parameters),
        "primary_input": "binary writing raster",
        "primary_output": "generated binary writing raster",
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_vae": False,
        "uses_runtime_teacher": False,
        "generation_feedback": "thresholded predicted raster patches",
    }

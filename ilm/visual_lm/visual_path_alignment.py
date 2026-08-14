from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from .visual_semantic_distillation import (
    PIXEL_LINGUIST_CONFIG_SHA256,
    PIXEL_LINGUIST_REPOSITORY,
    PIXEL_LINGUIST_REVISION,
    PIXEL_LINGUIST_WEIGHT_SHA256,
    file_sha256,
)
from .visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_SEMANTIC_DIM,
    V37_WIDTH,
)


V38_ARCHITECTURE = "visual-path-alignment-v38"


@dataclass(frozen=True)
class VisualPathAlignmentConfig:
    patch_size: int = V37_PATCH_SIZE
    patches: int = V37_PATCHES
    reader_hidden_size: int = 768
    reader_layers: int = 12
    reader_heads: int = 12
    reader_intermediate_size: int = 3072
    reader_dropout: float = 0.1
    projection_hidden_size: int = 1536
    semantic_dim: int = V37_SEMANTIC_DIM
    projection_dropout: float = 0.05
    answer_hidden_size: int = 512
    length_hidden_size: int = 256

    def __post_init__(self) -> None:
        if self.patch_size != V37_PATCH_SIZE or self.patches != V37_PATCHES:
            raise ValueError("V38 fixes a 16 by 1024 visual strip")
        if self.reader_hidden_size % self.reader_heads:
            raise ValueError("V38 reader width must divide into reader heads")
        if self.reader_layers < 1:
            raise ValueError("V38 reader must contain at least one layer")
        if self.reader_intermediate_size < self.reader_hidden_size:
            raise ValueError("V38 reader MLP is underspecified")
        if min(
            self.projection_hidden_size,
            self.semantic_dim,
            self.answer_hidden_size,
            self.length_hidden_size,
        ) < 8:
            raise ValueError("V38 heads are underspecified")
        if not 0.0 <= self.reader_dropout < 1.0:
            raise ValueError("V38 reader dropout must be in [0,1)")
        if not 0.0 <= self.projection_dropout < 1.0:
            raise ValueError("V38 projection dropout must be in [0,1)")

    @property
    def production_reader(self) -> bool:
        return (
            self.reader_hidden_size == 768
            and self.reader_layers == 12
            and self.reader_heads == 12
            and self.reader_intermediate_size == 3072
            and self.semantic_dim == V37_SEMANTIC_DIM
        )


@dataclass
class VisualPathAlignmentOutput:
    prompt_state: torch.Tensor
    answer_state: torch.Tensor
    length: torch.Tensor
    prompt_features: torch.Tensor
    answer_base: torch.Tensor
    answer_correction: torch.Tensor
    pooled_visual_state: torch.Tensor


class VisualPathAlignmentModel(nn.Module):
    """One-pass image-only reader and continuous answer-state planner."""

    def __init__(self, config: VisualPathAlignmentConfig) -> None:
        super().__init__()
        self.config = config
        reader_config = ViTConfig(
            hidden_size=config.reader_hidden_size,
            num_hidden_layers=config.reader_layers,
            num_attention_heads=config.reader_heads,
            intermediate_size=config.reader_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=config.reader_dropout,
            attention_probs_dropout_prob=config.reader_dropout,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=(config.patch_size, config.patch_size * config.patches),
            patch_size=config.patch_size,
            num_channels=3,
            qkv_bias=True,
        )
        self.reader = ViTModel(reader_config, add_pooling_layer=False)
        self.prompt_head = nn.Sequential(
            nn.LayerNorm(config.reader_hidden_size),
            nn.Linear(config.reader_hidden_size, config.projection_hidden_size),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
            nn.Linear(config.projection_hidden_size, config.semantic_dim),
        )
        self.answer_transform = nn.Linear(
            config.semantic_dim,
            config.semantic_dim,
            bias=False,
        )
        self.answer_adapter = nn.Sequential(
            nn.LayerNorm(config.semantic_dim),
            nn.Linear(config.semantic_dim, config.answer_hidden_size),
            nn.SiLU(),
            nn.Linear(config.answer_hidden_size, config.semantic_dim),
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(config.reader_hidden_size),
            nn.Linear(config.reader_hidden_size, config.length_hidden_size),
            nn.SiLU(),
            nn.Linear(config.length_hidden_size, 1),
        )
        nn.init.eye_(self.answer_transform.weight)
        nn.init.zeros_(self.answer_adapter[-1].weight)
        nn.init.zeros_(self.answer_adapter[-1].bias)
        self._reader_trainable = True

    @staticmethod
    def _validate_inputs(pixels: torch.Tensor, patch_mask: torch.Tensor) -> None:
        if not torch.is_floating_point(pixels) or not torch.is_floating_point(
            patch_mask
        ):
            raise TypeError("V38 pixels and masks must be floating tensors")
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (
            3,
            V37_PATCH_SIZE,
            V37_WIDTH,
        ):
            raise ValueError("V38 pixels must be [B,3,16,1024]")
        if patch_mask.shape != (pixels.shape[0], V37_PATCHES):
            raise ValueError("V38 patch mask must be [B,64]")
        if bool((patch_mask < 0).any()) or bool((patch_mask > 1).any()):
            raise ValueError("V38 patch mask must stay in [0,1]")

    def initialize_answer_rotation(self, rotation: torch.Tensor) -> None:
        expected = (self.config.semantic_dim, self.config.semantic_dim)
        if rotation.shape != expected or not torch.is_floating_point(rotation):
            raise ValueError("V38 answer rotation has an invalid shape or dtype")
        value = rotation.detach().float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("V38 answer rotation is non-finite")
        identity = torch.eye(value.shape[0], dtype=value.dtype, device=value.device)
        error = (value.T @ value - identity).abs().max()
        if float(error) > 2e-3:
            raise ValueError("V38 answer initialization is not orthogonal")
        # nn.Linear applies x @ weight.T; the fitted row-vector map is x @ rotation.
        self.answer_transform.weight.data.copy_(
            value.T.to(self.answer_transform.weight)
        )

    def freeze_reader(self) -> None:
        self.reader.requires_grad_(False).eval()
        self._reader_trainable = False

    def unfreeze_reader(self) -> None:
        self.reader.requires_grad_(True)
        self._reader_trainable = True

    def train(self, mode: bool = True) -> VisualPathAlignmentModel:
        super().train(mode)
        if mode and not self._reader_trainable:
            self.reader.eval()
        return self

    def encode_visual(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> VisualPathAlignmentOutput:
        self._validate_inputs(pixels, patch_mask)
        normalized_pixels = pixels.clamp(0, 1).mul(2.0).sub(1.0)
        patch_states = self.reader(normalized_pixels).last_hidden_state[:, 1:].float()
        mask = patch_mask.float().unsqueeze(-1)
        pooled = (patch_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        prompt_features = self.prompt_head(pooled).float()
        prompt_state = F.normalize(prompt_features, dim=-1)
        answer_base = self.answer_transform(prompt_state).float()
        answer_correction = self.answer_adapter(prompt_features).float()
        answer_state = F.normalize(answer_base + answer_correction, dim=-1)
        length = V37_PATCHES * torch.sigmoid(
            self.length_head(pooled).squeeze(-1).float()
        )
        return VisualPathAlignmentOutput(
            prompt_state=prompt_state,
            answer_state=answer_state,
            length=length,
            prompt_features=prompt_features,
            answer_base=answer_base,
            answer_correction=answer_correction,
            pooled_visual_state=pooled,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> VisualPathAlignmentOutput:
        return self.encode_visual(pixels, patch_mask)

    @torch.no_grad()
    def generate_plan(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> VisualPathAlignmentOutput:
        return self.encode_visual(prompt_pixels, prompt_mask)


def load_v38_v37_initialization(
    model: VisualPathAlignmentModel,
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("V38 V37 initialization hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("V38 initialization checkpoint must contain a mapping")
    if payload.get("architecture") != "visual-semantic-distillation-v37":
        raise ValueError("V38 initialization is not a V37 student")
    if payload.get("weight_route") != "all-parameter-ema":
        raise ValueError("V38 initialization must use the V37 EMA route")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("V38 initialization has no model state")

    copied: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if str(name).startswith("reader."):
            copied[str(name)] = value
        elif str(name).startswith("semantic_head."):
            copied[str(name).replace("semantic_head.", "prompt_head.", 1)] = value
    expected = {
        name
        for name in model.state_dict()
        if name.startswith("reader.") or name.startswith("prompt_head.")
    }
    if set(copied) != expected:
        missing = sorted(expected.difference(copied))
        unexpected = sorted(set(copied).difference(expected))
        raise ValueError(
            f"V38 cannot map V37 initialization: missing={missing}, "
            f"unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(copied, strict=False)
    if set(incompatible.missing_keys) != set(model.state_dict()).difference(copied):
        raise ValueError("V38 initialization missing-key audit failed")
    if incompatible.unexpected_keys:
        raise ValueError("V38 initialization contains unexpected parameters")
    return {
        "route": "v37-ema-reader-and-prompt-head-plus-train-only-rotation",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest,
        "v37_global_update": int(payload.get("global_update", 0)),
        "copied_tensors": len(copied),
        "copied_elements": sum(value.numel() for value in copied.values()),
        "external_origin": {
            "repository": PIXEL_LINGUIST_REPOSITORY,
            "revision": PIXEL_LINGUIST_REVISION,
            "weight_sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
            "config_sha256": PIXEL_LINGUIST_CONFIG_SHA256,
            "license": "not stated upstream; local research only",
        },
        "evidence_eligible": True,
    }


def visual_path_alignment_boundary_receipt(
    model: VisualPathAlignmentModel,
) -> dict[str, Any]:
    forbidden_fragments = (
        "token_embed",
        "token_embedding",
        "input_id",
        "vocab",
        "unicode",
        "character_id",
        "glyph_id",
        "codebook",
        "candidate",
        "teacher",
        "target_bank",
        "bge",
        "ocr",
        "retrieval",
    )
    forbidden = [
        name
        for name, _ in model.named_parameters()
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    ]
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "architecture": V38_ARCHITECTURE,
        "model_config": asdict(model.config),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_cap": 100_000_000,
        "parameter_cap_pass": total < 100_000_000,
        "primary_input": "prompt raster and clean visual patch mask",
        "primary_output": "continuous prompt state, answer state, and visual length",
        "forbidden_parameter_names": forbidden,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_bge_at_runtime": False,
        "uses_qwen_at_runtime": False,
        "uses_teacher_at_runtime": False,
        "candidate_bank_deployed": False,
        "network_client_deployed": False,
        "answer_map_angle_capped": False,
        "generates_raster": False,
        "external_origin": {
            "repository": PIXEL_LINGUIST_REPOSITORY,
            "revision": PIXEL_LINGUIST_REVISION,
            "weight_sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
            "config_sha256": PIXEL_LINGUIST_CONFIG_SHA256,
        },
    }


__all__ = [
    "V38_ARCHITECTURE",
    "VisualPathAlignmentConfig",
    "VisualPathAlignmentModel",
    "VisualPathAlignmentOutput",
    "file_sha256",
    "load_v38_v37_initialization",
    "visual_path_alignment_boundary_receipt",
]

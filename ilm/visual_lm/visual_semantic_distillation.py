from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from .visual_semantic_distillation_data import (
    V37_ARCHITECTURE,
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_SEMANTIC_DIM,
    V37_WIDTH,
)
from .visual_semantic_plan import (
    PIXEL_LINGUIST_CONFIG_SHA256,
    PIXEL_LINGUIST_REPOSITORY,
    PIXEL_LINGUIST_REVISION,
    PIXEL_LINGUIST_WEIGHT_SHA256,
    file_sha256,
    load_pixel_linguist_reader,
    resolve_pixel_linguist_checkpoint,
)


@dataclass(frozen=True)
class VisualSemanticDistillationConfig:
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
    plan_hidden_size: int = 512
    length_hidden_size: int = 256
    initial_residual_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.patch_size != V37_PATCH_SIZE or self.patches != V37_PATCHES:
            raise ValueError("V37 fixes a 16 by 1024 prompt raster")
        if self.reader_hidden_size % self.reader_heads:
            raise ValueError("V37 reader width must divide into reader heads")
        if self.reader_layers < 1:
            raise ValueError("V37 reader must contain at least one layer")
        if self.reader_intermediate_size < self.reader_hidden_size:
            raise ValueError("V37 reader MLP is underspecified")
        if (
            min(
                self.projection_hidden_size,
                self.semantic_dim,
                self.plan_hidden_size,
                self.length_hidden_size,
            )
            < 8
        ):
            raise ValueError("V37 semantic heads are underspecified")
        if not 0.0 <= self.reader_dropout < 1.0:
            raise ValueError("V37 reader dropout must be in [0,1)")
        if not 0.0 <= self.projection_dropout < 1.0:
            raise ValueError("V37 projection dropout must be in [0,1)")
        if not 0.0 < self.initial_residual_scale < 0.5:
            raise ValueError("V37 initial residual scale must be in (0,0.5)")

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
class VisualSemanticDistillationOutput:
    semantic_state: torch.Tensor
    answer_plan: torch.Tensor
    length: torch.Tensor
    semantic_features: torch.Tensor
    scaled_residual: torch.Tensor
    pooled_visual_state: torch.Tensor


class VisualSemanticDistillationModel(nn.Module):
    def __init__(self, config: VisualSemanticDistillationConfig) -> None:
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
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(config.reader_hidden_size),
            nn.Linear(config.reader_hidden_size, config.projection_hidden_size),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
            nn.Linear(config.projection_hidden_size, config.semantic_dim),
        )
        self.plan_head = nn.Sequential(
            nn.LayerNorm(config.semantic_dim),
            nn.Linear(config.semantic_dim, config.plan_hidden_size),
            nn.SiLU(),
            nn.Linear(config.plan_hidden_size, config.semantic_dim),
        )
        self.length_head = nn.Sequential(
            nn.Linear(config.semantic_dim, config.length_hidden_size),
            nn.SiLU(),
            nn.Linear(config.length_hidden_size, 1),
        )
        residual_probability = config.initial_residual_scale / 0.5
        residual_logit = math.log(residual_probability / (1.0 - residual_probability))
        self.residual_logit = nn.Parameter(torch.tensor(residual_logit))
        self._reader_trainable = True

    @staticmethod
    def _validate_inputs(pixels: torch.Tensor, patch_mask: torch.Tensor) -> None:
        if not torch.is_floating_point(pixels) or not torch.is_floating_point(
            patch_mask
        ):
            raise TypeError("V37 pixels and masks must be floating tensors")
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (
            3,
            V37_PATCH_SIZE,
            V37_WIDTH,
        ):
            raise ValueError("V37 pixels must be [B,3,16,1024]")
        if patch_mask.shape != (pixels.shape[0], V37_PATCHES):
            raise ValueError("V37 patch mask must be [B,64]")
        if bool((patch_mask < 0).any()) or bool((patch_mask > 1).any()):
            raise ValueError("V37 patch mask must stay in [0,1]")

    @property
    def residual_scale(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.residual_logit.float())

    def freeze_reader(self) -> None:
        self.reader.requires_grad_(False).eval()
        self._reader_trainable = False

    def unfreeze_reader(self) -> None:
        self.reader.requires_grad_(True)
        self._reader_trainable = True

    def train(self, mode: bool = True) -> VisualSemanticDistillationModel:
        super().train(mode)
        if mode and not self._reader_trainable:
            self.reader.eval()
        return self

    def encode_visual(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> VisualSemanticDistillationOutput:
        self._validate_inputs(pixels, patch_mask)
        normalized_pixels = pixels.clamp(0, 1).mul(2.0).sub(1.0)
        patch_states = self.reader(normalized_pixels).last_hidden_state[:, 1:].float()
        mask = patch_mask.float().unsqueeze(-1)
        pooled = (patch_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        semantic_features = self.semantic_head(pooled).float()
        semantic_state = F.normalize(semantic_features, dim=-1)
        residual_direction = F.normalize(self.plan_head(semantic_state).float(), dim=-1)
        scaled_residual = self.residual_scale * residual_direction
        answer_plan = F.normalize(semantic_state + scaled_residual, dim=-1)
        length = V37_PATCHES * torch.sigmoid(
            self.length_head(semantic_state).squeeze(-1).float()
        )
        return VisualSemanticDistillationOutput(
            semantic_state=semantic_state,
            answer_plan=answer_plan,
            length=length,
            semantic_features=semantic_features,
            scaled_residual=scaled_residual,
            pooled_visual_state=pooled,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> VisualSemanticDistillationOutput:
        return self.encode_visual(pixels, patch_mask)

    @torch.no_grad()
    def generate_plan(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> VisualSemanticDistillationOutput:
        return self.encode_visual(prompt_pixels, prompt_mask)


def load_v37_pixel_linguist_initialization(
    model: VisualSemanticDistillationModel,
    checkpoint: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> dict[str, Any]:
    resolved = resolve_pixel_linguist_checkpoint(
        checkpoint,
        local_files_only=local_files_only,
    )
    receipt = load_pixel_linguist_reader(model.reader, resolved)
    return receipt | {
        "route": "pixel-linguist-v0-vit-only-v37",
        "config_sha256": PIXEL_LINGUIST_CONFIG_SHA256,
        "upstream_representation_head": "masked active-patch mean; pooler MLP dropped",
        "evidence_eligible": True,
    }


def visual_semantic_distillation_boundary_receipt(
    model: VisualSemanticDistillationModel,
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
        "architecture": V37_ARCHITECTURE,
        "model_config": asdict(model.config),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_cap": 100_000_000,
        "parameter_cap_pass": total < 100_000_000,
        "primary_input": "prompt raster and clean visual patch mask",
        "primary_output": "continuous semantic state, answer plan, and visual length",
        "forbidden_parameter_names": forbidden,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_bge_at_runtime": False,
        "uses_teacher_at_runtime": False,
        "candidate_bank_deployed": False,
        "network_client_deployed": False,
        "generates_raster": False,
        "external_initialization": {
            "repository": PIXEL_LINGUIST_REPOSITORY,
            "revision": PIXEL_LINGUIST_REVISION,
            "weight_sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
            "config_sha256": PIXEL_LINGUIST_CONFIG_SHA256,
        },
    }


__all__ = [
    "PIXEL_LINGUIST_CONFIG_SHA256",
    "PIXEL_LINGUIST_REPOSITORY",
    "PIXEL_LINGUIST_REVISION",
    "PIXEL_LINGUIST_WEIGHT_SHA256",
    "V37_ARCHITECTURE",
    "VisualSemanticDistillationConfig",
    "VisualSemanticDistillationModel",
    "VisualSemanticDistillationOutput",
    "file_sha256",
    "load_v37_pixel_linguist_initialization",
    "resolve_pixel_linguist_checkpoint",
    "visual_semantic_distillation_boundary_receipt",
]

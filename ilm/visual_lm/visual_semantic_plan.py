from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from .visual_semantic_plan_data import (
    V36_ARCHITECTURE,
    V36_PATCHES,
    V36_PATCH_SIZE,
    V36_PLAN_SLOTS,
    V36_WIDTH,
)


PIXEL_LINGUIST_REPOSITORY = "Pixel-Linguist/Pixel-Linguist-v0"
PIXEL_LINGUIST_REVISION = "086b70818b2241e81b0de131aa5debe982af7a54"
PIXEL_LINGUIST_WEIGHT_SHA256 = (
    "84c1bfbeada57e7e70164811a201a116ad18c22df69beb762fdbb853f8e02676"
)
PIXEL_LINGUIST_CONFIG_SHA256 = (
    "b61018a997aa030f41571615641204cf40bd7c623f25dfc129a49ffe1f571b97"
)


@dataclass(frozen=True)
class VisualSemanticPlanConfig:
    patch_size: int = V36_PATCH_SIZE
    patches: int = V36_PATCHES
    reader_hidden_size: int = 768
    reader_layers: int = 12
    reader_heads: int = 12
    reader_intermediate_size: int = 3072
    reader_dropout: float = 0.1
    planner_dim: int = 384
    planner_layers: int = 3
    planner_heads: int = 6
    planner_mlp_dim: int = 1536
    planner_dropout: float = 0.05
    plan_slots: int = V36_PLAN_SLOTS
    plan_dim: int = 768
    length_hidden_size: int = 192

    def __post_init__(self) -> None:
        if self.patch_size != V36_PATCH_SIZE or self.patches != V36_PATCHES:
            raise ValueError("V36 fixes a 16 by 1024 prompt raster")
        if self.reader_hidden_size % self.reader_heads:
            raise ValueError("V36 reader width must divide into reader heads")
        if self.reader_layers < 1 or self.reader_intermediate_size < self.reader_hidden_size:
            raise ValueError("V36 reader configuration is invalid")
        if self.planner_dim % self.planner_heads or self.planner_layers < 1:
            raise ValueError("V36 planner configuration is invalid")
        if self.planner_mlp_dim < self.planner_dim:
            raise ValueError("V36 planner MLP is underspecified")
        if self.plan_slots != V36_PLAN_SLOTS:
            raise ValueError("V36 fixes one global and four spatial plan slots")
        if self.plan_dim < 16 or self.length_hidden_size < 8:
            raise ValueError("V36 output configuration is underspecified")
        for dropout in (self.reader_dropout, self.planner_dropout):
            if not 0.0 <= dropout < 1.0:
                raise ValueError("V36 dropout must be in [0,1)")

    @property
    def production_reader(self) -> bool:
        return (
            self.reader_hidden_size == 768
            and self.reader_layers == 12
            and self.reader_heads == 12
            and self.reader_intermediate_size == 3072
            and self.plan_dim == 768
        )


@dataclass
class VisualSemanticPlanOutput:
    plans: torch.Tensor
    length: torch.Tensor
    planner_hidden: torch.Tensor
    reader_memory: torch.Tensor


class VisualSemanticPlanModel(nn.Module):
    def __init__(self, config: VisualSemanticPlanConfig) -> None:
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
        self.memory_projection = nn.Linear(config.reader_hidden_size, config.planner_dim)
        self.memory_norm = nn.LayerNorm(config.planner_dim)
        self.plan_queries = nn.Parameter(
            torch.empty(1, config.plan_slots, config.planner_dim)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.planner_dim,
            nhead=config.planner_heads,
            dim_feedforward=config.planner_mlp_dim,
            dropout=config.planner_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.planner = nn.TransformerDecoder(
            layer,
            num_layers=config.planner_layers,
            norm=nn.LayerNorm(config.planner_dim),
        )
        self.plan_projection = nn.Linear(config.planner_dim, config.plan_dim)
        self.plan_scale = nn.Parameter(torch.ones(1, config.plan_slots, config.plan_dim))
        self.plan_bias = nn.Parameter(torch.zeros(1, config.plan_slots, config.plan_dim))
        self.length_head = nn.Sequential(
            nn.Linear(config.planner_dim, config.length_hidden_size),
            nn.SiLU(),
            nn.Linear(config.length_hidden_size, 1),
        )
        self._reader_trainable_blocks: int | None = None
        nn.init.normal_(self.plan_queries, std=0.02)

    @staticmethod
    def _validate_inputs(
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> None:
        if not torch.is_floating_point(prompt_pixels) or not torch.is_floating_point(
            prompt_mask
        ):
            raise TypeError("V36 prompt pixels and mask must be floating tensors")
        if prompt_pixels.ndim != 4 or tuple(prompt_pixels.shape[1:]) != (
            3,
            V36_PATCH_SIZE,
            V36_WIDTH,
        ):
            raise ValueError("V36 prompt pixels must be [B,3,16,1024]")
        if prompt_mask.shape != (prompt_pixels.shape[0], V36_PATCHES):
            raise ValueError("V36 prompt mask must be [B,64]")

    def freeze_reader(self) -> None:
        self._reader_trainable_blocks = 0
        self.reader.requires_grad_(False).eval()

    def unfreeze_reader_final_blocks(self, count: int = 2) -> None:
        if not 1 <= count <= len(self.reader.encoder.layer):
            raise ValueError("V36 reader block count is invalid")
        self.reader.requires_grad_(False)
        for block in self.reader.encoder.layer[-count:]:
            block.requires_grad_(True)
        self.reader.layernorm.requires_grad_(True)
        self._reader_trainable_blocks = count

    def train(self, mode: bool = True) -> VisualSemanticPlanModel:
        super().train(mode)
        if mode and self._reader_trainable_blocks is not None:
            self.reader.eval()
            if self._reader_trainable_blocks:
                for block in self.reader.encoder.layer[-self._reader_trainable_blocks :]:
                    block.train()
                self.reader.layernorm.train()
        return self

    def encode_prompt(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(prompt_pixels, prompt_mask)
        normalized = prompt_pixels.clamp(0, 1).mul(2.0).sub(1.0)
        reader_memory = self.reader(normalized).last_hidden_state
        memory = self.memory_norm(self.memory_projection(reader_memory))
        cls_padding = torch.zeros(
            prompt_mask.shape[0],
            1,
            dtype=torch.bool,
            device=prompt_mask.device,
        )
        memory_padding = torch.cat((cls_padding, prompt_mask <= 0.0), dim=1)
        return memory, memory_padding

    def forward(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> VisualSemanticPlanOutput:
        memory, memory_padding = self.encode_prompt(prompt_pixels, prompt_mask)
        queries = self.plan_queries.expand(prompt_pixels.shape[0], -1, -1)
        hidden = self.planner(
            queries,
            memory,
            memory_key_padding_mask=memory_padding,
        )
        projected = self.plan_projection(hidden).float()
        projected = F.layer_norm(projected, (self.config.plan_dim,))
        plans = F.normalize(
            projected * self.plan_scale.float() + self.plan_bias.float(),
            dim=-1,
        )
        length = F.softplus(self.length_head(hidden[:, 0]).squeeze(-1).float())
        return VisualSemanticPlanOutput(
            plans=plans,
            length=length,
            planner_hidden=hidden,
            reader_memory=memory,
        )

    @torch.no_grad()
    def generate_plan(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> VisualSemanticPlanOutput:
        return self.forward(prompt_pixels, prompt_mask)


class VisualSentenceImageTeacher(nn.Module):
    def __init__(self, config: VisualSemanticPlanConfig) -> None:
        super().__init__()
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
        self.config = config

    def forward(self, pixels: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        VisualSemanticPlanModel._validate_inputs(pixels, patch_mask)
        normalized = pixels.clamp(0, 1).mul(2.0).sub(1.0)
        states = self.reader(normalized).last_hidden_state[:, 1:].float()
        mask = patch_mask.float().unsqueeze(-1)
        pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pixel_linguist_checkpoint(
    path: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> Path:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            PIXEL_LINGUIST_REPOSITORY,
            "pytorch_model.bin",
            revision=PIXEL_LINGUIST_REVISION,
            local_files_only=local_files_only,
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = file_sha256(resolved)
    if digest != PIXEL_LINGUIST_WEIGHT_SHA256:
        raise ValueError(
            "V36 Pixel-Linguist checkpoint hash mismatch: "
            f"expected {PIXEL_LINGUIST_WEIGHT_SHA256}, got {digest}"
        )
    return resolved


def load_pixel_linguist_reader(
    reader: ViTModel,
    checkpoint: str | Path,
) -> dict[str, Any]:
    path = resolve_pixel_linguist_checkpoint(checkpoint)
    source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(source, Mapping):
        raise TypeError("V36 Pixel-Linguist checkpoint must contain a state mapping")
    selected = {
        key.removeprefix("vit."): value
        for key, value in source.items()
        if isinstance(key, str) and key.startswith("vit.")
    }
    ignored = sorted(
        key
        for key in source
        if isinstance(key, str) and not key.startswith("vit.")
    )
    if len(selected) != 198 or ignored != [
        "pooler.linear.bias",
        "pooler.linear.weight",
        "pooler.ln.bias",
        "pooler.ln.weight",
    ]:
        raise ValueError("V36 Pixel-Linguist state boundary changed")
    required_positions = V36_PATCHES + 1
    position_key = "embeddings.position_embeddings"
    if selected[position_key].shape != (
        1,
        required_positions,
        reader.config.hidden_size,
    ):
        raise ValueError("V36 Pixel-Linguist positions do not match 64 patches")
    missing, unexpected = reader.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"V36 Pixel-Linguist mapping failed: missing={missing}, unexpected={unexpected}"
        )
    return {
        "repository": PIXEL_LINGUIST_REPOSITORY,
        "revision": PIXEL_LINGUIST_REVISION,
        "checkpoint": str(path),
        "sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
        "selected_state_prefix": "vit.*",
        "selected_tensors": len(selected),
        "ignored_tensors": ignored,
        "selected_positions": required_positions,
        "missing_keys": [],
        "unexpected_keys": [],
        "license": "not stated; local research only; no redistribution",
    }


def visual_semantic_plan_boundary_receipt(
    model: VisualSemanticPlanModel,
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
        "answer",
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
        "architecture": V36_ARCHITECTURE,
        "model_config": asdict(model.config),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_cap": 100_000_000,
        "parameter_cap_pass": total < 100_000_000,
        "primary_input": "prompt raster and visual patch mask",
        "primary_output": "five continuous visual semantic plans and visual length",
        "forbidden_parameter_names": forbidden,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_answer_teacher_at_runtime": False,
        "candidate_bank_deployed": False,
        "generates_raster": False,
    }

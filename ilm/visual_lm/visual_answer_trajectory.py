from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from .visual_semantic_distillation import file_sha256
from .visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_SEMANTIC_DIM,
    V37_WIDTH,
)


V39_ARCHITECTURE = "visual-answer-trajectory-v39"
V39_MAX_SEGMENTS = 16


@dataclass(frozen=True)
class VisualAnswerTrajectoryConfig:
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
    planner_hidden_size: int = 512
    planner_layers: int = 3
    planner_heads: int = 8
    planner_intermediate_size: int = 2048
    planner_dropout: float = 0.1
    maximum_segments: int = V39_MAX_SEGMENTS
    length_hidden_size: int = 256

    def __post_init__(self) -> None:
        if self.patch_size != V37_PATCH_SIZE or self.patches != V37_PATCHES:
            raise ValueError("V39 fixes a 16 by 1024 visual strip")
        if self.reader_hidden_size % self.reader_heads:
            raise ValueError("V39 reader width must divide into reader heads")
        if self.planner_hidden_size % self.planner_heads:
            raise ValueError("V39 planner width must divide into planner heads")
        if min(self.reader_layers, self.planner_layers) < 1:
            raise ValueError("V39 requires positive reader and planner depth")
        if self.reader_intermediate_size < self.reader_hidden_size:
            raise ValueError("V39 reader MLP is underspecified")
        if self.planner_intermediate_size < self.planner_hidden_size:
            raise ValueError("V39 planner MLP is underspecified")
        if self.maximum_segments != V39_MAX_SEGMENTS:
            raise ValueError("V39 fixes sixteen ordered answer segments")
        if min(
            self.projection_hidden_size,
            self.semantic_dim,
            self.answer_hidden_size,
            self.length_hidden_size,
        ) < 8:
            raise ValueError("V39 projection heads are underspecified")
        for name, value in (
            ("reader_dropout", self.reader_dropout),
            ("projection_dropout", self.projection_dropout),
            ("planner_dropout", self.planner_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"V39 {name} must be in [0,1)")


@dataclass
class VisualAnswerEncoding:
    read_state: torch.Tensor
    read_features: torch.Tensor
    patch_states: torch.Tensor
    pooled_visual_state: torch.Tensor


@dataclass
class VisualAnswerTrajectoryOutput:
    read_state: torch.Tensor
    baseline_answer_state: torch.Tensor
    answer_state: torch.Tensor
    stage1_answer_state: torch.Tensor
    segment_states: torch.Tensor
    stage1_segment_states: torch.Tensor
    stop_logits: torch.Tensor
    active_probabilities: torch.Tensor
    lengths: torch.Tensor
    patch_states: torch.Tensor
    pooled_visual_state: torch.Tensor


def _zero_last_linear(module: nn.Sequential) -> None:
    final = module[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("V39 residual head must end in a linear layer")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


class VisualAnswerTrajectoryModel(nn.Module):
    """Image-only reader with an ordered continuous answer trajectory."""

    def __init__(self, config: VisualAnswerTrajectoryConfig) -> None:
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

        self.memory_projection = nn.Sequential(
            nn.LayerNorm(config.reader_hidden_size),
            nn.Linear(config.reader_hidden_size, config.planner_hidden_size),
        )
        self.baseline_projection = nn.Sequential(
            nn.LayerNorm(config.semantic_dim),
            nn.Linear(config.semantic_dim, config.planner_hidden_size),
        )
        self.position_queries = nn.Parameter(
            torch.empty(1 + config.maximum_segments, config.planner_hidden_size)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.planner_hidden_size,
            nhead=config.planner_heads,
            dim_feedforward=config.planner_intermediate_size,
            dropout=config.planner_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.planner = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.planner_layers,
            norm=nn.LayerNorm(config.planner_hidden_size),
        )
        self.feedback_projection = nn.Sequential(
            nn.LayerNorm(config.semantic_dim),
            nn.Linear(config.semantic_dim, config.planner_hidden_size),
        )
        self.stage1_correction = nn.Sequential(
            nn.LayerNorm(config.planner_hidden_size),
            nn.Linear(config.planner_hidden_size, config.answer_hidden_size),
            nn.SiLU(),
            nn.Linear(config.answer_hidden_size, config.semantic_dim),
        )
        self.final_correction = nn.Sequential(
            nn.LayerNorm(config.planner_hidden_size),
            nn.Linear(config.planner_hidden_size, config.answer_hidden_size),
            nn.SiLU(),
            nn.Linear(config.answer_hidden_size, config.semantic_dim),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(config.planner_hidden_size),
            nn.Linear(config.planner_hidden_size, config.length_hidden_size),
            nn.SiLU(),
            nn.Linear(config.length_hidden_size, 1),
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(config.planner_hidden_size),
            nn.Linear(config.planner_hidden_size, config.length_hidden_size),
            nn.SiLU(),
            nn.Linear(config.length_hidden_size, 1),
        )

        nn.init.eye_(self.answer_transform.weight)
        _zero_last_linear(self.answer_adapter)
        _zero_last_linear(self.stage1_correction)
        _zero_last_linear(self.final_correction)
        nn.init.normal_(self.position_queries, std=0.02)
        stop_final = self.stop_head[-1]
        if not isinstance(stop_final, nn.Linear):
            raise TypeError("V39 stop head must end in a linear layer")
        nn.init.zeros_(stop_final.weight)
        nn.init.constant_(stop_final.bias, -2.0)
        self._reader_trainable = True

    @staticmethod
    def _validate_inputs(pixels: torch.Tensor, patch_mask: torch.Tensor) -> None:
        if not torch.is_floating_point(pixels) or not torch.is_floating_point(
            patch_mask
        ):
            raise TypeError("V39 pixels and masks must be floating tensors")
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (
            3,
            V37_PATCH_SIZE,
            V37_WIDTH,
        ):
            raise ValueError("V39 pixels must be [B,3,16,1024]")
        if patch_mask.shape != (pixels.shape[0], V37_PATCHES):
            raise ValueError("V39 patch mask must be [B,64]")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("V39 pixels must be finite")
        if not bool(torch.isfinite(patch_mask).all()):
            raise ValueError("V39 patch mask must be finite")
        if bool((patch_mask < 0).any()) or bool((patch_mask > 1).any()):
            raise ValueError("V39 patch mask must stay in [0,1]")

    @staticmethod
    def _safe_attention_mask(patch_mask: torch.Tensor) -> torch.Tensor:
        safe = patch_mask.float().clone()
        empty = safe.sum(dim=1) <= 0
        if bool(empty.any()):
            safe[empty, 0] = 1.0
        return safe

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1,
        )

    @staticmethod
    def _active_probabilities(stop_logits: torch.Tensor) -> torch.Tensor:
        if stop_logits.ndim != 2 or stop_logits.shape[1] != V39_MAX_SEGMENTS:
            raise ValueError("V39 stop logits must be [B,16]")
        continuation = 1.0 - stop_logits.float().sigmoid()
        first = torch.ones(
            stop_logits.shape[0],
            1,
            dtype=continuation.dtype,
            device=continuation.device,
        )
        return torch.cat((first, continuation[:, :-1].cumprod(dim=1)), dim=1)

    def freeze_reader(self) -> None:
        self.reader.requires_grad_(False).eval()
        self.prompt_head.requires_grad_(False).eval()
        self._reader_trainable = False

    def unfreeze_reader(self) -> None:
        self.reader.requires_grad_(True)
        self.prompt_head.requires_grad_(True)
        self._reader_trainable = True

    def train(self, mode: bool = True) -> VisualAnswerTrajectoryModel:
        super().train(mode)
        if mode and not self._reader_trainable:
            self.reader.eval()
            self.prompt_head.eval()
        return self

    def encode_visual(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> VisualAnswerEncoding:
        self._validate_inputs(pixels, patch_mask)
        normalized_pixels = pixels.clamp(0, 1).mul(2.0).sub(1.0)
        patch_states = self.reader(normalized_pixels).last_hidden_state[:, 1:].float()
        mask = patch_mask.float().unsqueeze(-1)
        pooled = (patch_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        read_features = self.prompt_head(pooled).float()
        return VisualAnswerEncoding(
            read_state=F.normalize(read_features, dim=-1),
            read_features=read_features,
            patch_states=patch_states,
            pooled_visual_state=pooled,
        )

    def plan_from_encoding(
        self,
        encoding: VisualAnswerEncoding,
        patch_mask: torch.Tensor,
    ) -> VisualAnswerTrajectoryOutput:
        batch = encoding.read_state.shape[0]
        if encoding.read_state.shape != (batch, self.config.semantic_dim):
            raise ValueError("V39 visual encoding has an invalid semantic shape")
        if encoding.patch_states.shape != (
            batch,
            self.config.patches,
            self.config.reader_hidden_size,
        ):
            raise ValueError("V39 visual encoding has invalid patch states")
        safe_mask = self._safe_attention_mask(patch_mask)
        memory = self.memory_projection(encoding.patch_states).float()
        baseline_base = self.answer_transform(encoding.read_state).float()
        baseline_correction = self.answer_adapter(encoding.read_features).float()
        baseline = F.normalize(baseline_base + baseline_correction, dim=-1)

        query_count = 1 + self.config.maximum_segments
        queries = self.position_queries.unsqueeze(0).expand(batch, -1, -1)
        first_inputs = queries + self.baseline_projection(baseline).unsqueeze(1)
        causal = self._causal_mask(query_count, first_inputs.device)
        padding = safe_mask <= 0
        stage1_hidden = self.planner(
            first_inputs,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=padding,
        ).float()
        stage1_states = F.normalize(
            baseline.unsqueeze(1) + self.stage1_correction(stage1_hidden).float(),
            dim=-1,
        )

        previous_segments = torch.cat(
            (stage1_states[:, :1], stage1_states[:, 1:-1]),
            dim=1,
        )
        shifted_states = torch.cat(
            (stage1_states[:, :1], previous_segments),
            dim=1,
        )
        if shifted_states.shape[1] != query_count:
            raise RuntimeError("V39 feedback trajectory has the wrong length")
        second_inputs = queries + self.feedback_projection(shifted_states)
        final_hidden = self.planner(
            second_inputs,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=padding,
        ).float()
        final_states = F.normalize(
            stage1_states + self.final_correction(final_hidden).float(),
            dim=-1,
        )
        segment_hidden = final_hidden[:, 1:]
        stop_logits = self.stop_head(segment_hidden).squeeze(-1).float()
        lengths = self.config.patches * torch.sigmoid(
            self.length_head(segment_hidden).squeeze(-1).float()
        )
        return VisualAnswerTrajectoryOutput(
            read_state=encoding.read_state,
            baseline_answer_state=baseline,
            answer_state=final_states[:, 0],
            stage1_answer_state=stage1_states[:, 0],
            segment_states=final_states[:, 1:],
            stage1_segment_states=stage1_states[:, 1:],
            stop_logits=stop_logits,
            active_probabilities=self._active_probabilities(stop_logits),
            lengths=lengths,
            patch_states=encoding.patch_states,
            pooled_visual_state=encoding.pooled_visual_state,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> VisualAnswerTrajectoryOutput:
        encoding = self.encode_visual(pixels, patch_mask)
        return self.plan_from_encoding(encoding, patch_mask)

    @torch.no_grad()
    def generate_plan(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> VisualAnswerTrajectoryOutput:
        return self(prompt_pixels, prompt_mask)


def load_v39_v38_initialization(
    model: VisualAnswerTrajectoryModel,
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("V39 V38 initialization hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("V39 initialization checkpoint must contain a mapping")
    if payload.get("architecture") != "visual-path-alignment-v38":
        raise ValueError("V39 initialization is not a V38 student")
    if payload.get("weight_route") != "all-parameter-ema":
        raise ValueError("V39 initialization must use the V38 EMA artifact")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("V39 initialization has no model state")

    prefixes = ("reader.", "prompt_head.", "answer_transform.", "answer_adapter.")
    copied = {
        str(name): value
        for name, value in state.items()
        if str(name).startswith(prefixes)
    }
    expected = {
        name for name in model.state_dict() if name.startswith(prefixes)
    }
    if set(copied) != expected:
        missing = sorted(expected.difference(copied))
        unexpected = sorted(set(copied).difference(expected))
        raise ValueError(
            f"V39 cannot map V38 initialization: missing={missing}, "
            f"unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(copied, strict=False)
    if set(incompatible.missing_keys) != set(model.state_dict()).difference(copied):
        raise ValueError("V39 initialization missing-key audit failed")
    if incompatible.unexpected_keys:
        raise ValueError("V39 initialization contains unexpected parameters")
    return {
        "route": "v38-ema-reader-prompt-and-global-answer-map",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest,
        "v38_global_update": int(payload.get("global_update", 0)),
        "copied_tensors": len(copied),
        "copied_elements": sum(value.numel() for value in copied.values()),
        "planner_residuals_zero_initialized": True,
    }


def visual_answer_trajectory_boundary_receipt(
    model: VisualAnswerTrajectoryModel,
) -> dict[str, Any]:
    forbidden = (
        "token_embed",
        "token_embedding",
        "input_id",
        "vocab",
        "unicode",
        "character_id",
        "codebook",
        "quant",
        "ocr",
        "retrieval",
        "teacher",
        "candidate",
        "bge",
        "qwen",
        "opencc",
    )
    suspicious = sorted(
        name
        for name, _parameter in model.named_parameters()
        if any(fragment in name.lower() for fragment in forbidden)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "architecture": V39_ARCHITECTURE,
        "config": asdict(model.config),
        "total_parameters": total,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_cap": 120_000_000,
        "parameter_cap_pass": total < 120_000_000,
        "forbidden_parameter_names": suspicious,
        "forward_parameters": list(inspect.signature(model.forward).parameters),
        "generate_parameters": list(
            inspect.signature(model.generate_plan).parameters
        ),
        "deployable_inputs": ["prompt_pixels", "prompt_mask"],
        "primary_output": "global and sixteen ordered continuous answer states",
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_bge_at_runtime": False,
        "uses_qwen_at_runtime": False,
        "uses_opencc_at_runtime": False,
        "uses_candidate_bank_at_runtime": False,
        "uses_target_answer_at_runtime": False,
        "generates_raster": False,
    }


__all__ = [
    "V39_ARCHITECTURE",
    "V39_MAX_SEGMENTS",
    "VisualAnswerEncoding",
    "VisualAnswerTrajectoryConfig",
    "VisualAnswerTrajectoryModel",
    "VisualAnswerTrajectoryOutput",
    "load_v39_v38_initialization",
    "visual_answer_trajectory_boundary_receipt",
]

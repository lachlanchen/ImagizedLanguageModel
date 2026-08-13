from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense_visual_future_energy import ResidualSemanticAdapter
from .joint_visual_compatibility import paired_assignment_loss
from .saccade_lm import FovealRetina, VisualSaccadeConfig
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V29_ARCHITECTURE = "conditional-visual-density-ratio-v29"
V29_SUFFIX_CELLS = 4


@dataclass(frozen=True)
class ConditionalVisualDensityRatioConfig:
    cell_size: int = 32
    maximum_cells: int = 64
    suffix_cells: int = V29_SUFFIX_CELLS
    visual_dim: int = 192
    semantic_dim: int = 192
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    retina_base_channels: int = 64
    semantic_hidden_dim: int = 384
    semantic_residual_scale: float = 0.10
    evidence_layers: int = 2
    evidence_heads: int = 6
    evidence_mlp_ratio: float = 2.0
    evidence_dropout: float = 0.05
    relation_hidden_dim: int = 384
    score_chunk_size: int = 128

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V29 requires 32x32 visual cells")
        if self.maximum_cells != 64 or self.suffix_cells != V29_SUFFIX_CELLS:
            raise ValueError("V29 fixes 64 context cells and a suffix of four")
        if self.visual_dim < 64 or self.semantic_dim < 64:
            raise ValueError("V29 visual dimensions are underspecified")
        if self.semantic_dim != self.visual_dim:
            raise ValueError("V29 requires aligned raw and semantic dimensions")
        if self.model_dim < 128 or self.model_dim % self.heads:
            raise ValueError("V29 causal width must divide into causal heads")
        if self.model_dim % self.evidence_heads:
            raise ValueError("V29 evidence width must divide into evidence heads")
        if self.layers < 1 or self.evidence_layers < 1:
            raise ValueError("V29 requires causal and evidence layers")
        if self.mlp_ratio < 2.0 or self.evidence_mlp_ratio < 1.0:
            raise ValueError("V29 MLP ratios are invalid")
        if self.retina_base_channels < 8 or self.semantic_hidden_dim < 8:
            raise ValueError("V29 perception dimensions are underspecified")
        if not 0.0 < self.semantic_residual_scale <= 1.0:
            raise ValueError("V29 semantic residual scale must be in (0,1]")
        for value in (self.dropout, self.evidence_dropout):
            if not 0.0 <= value < 1.0:
                raise ValueError("V29 dropout must be in [0,1)")
        if self.relation_hidden_dim < 32 or self.score_chunk_size < 1:
            raise ValueError("V29 relation or chunk width is invalid")


class CandidateEvidenceLayer(nn.Module):
    """Update independent candidate queries from retained visual history."""

    def __init__(self, config: ConditionalVisualDensityRatioConfig) -> None:
        super().__init__()
        hidden = int(config.model_dim * config.evidence_mlp_ratio)
        self.query_norm = nn.LayerNorm(config.model_dim)
        self.context_norm = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.evidence_heads,
            dropout=config.evidence_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.evidence_dropout)
        self.mlp_norm = nn.LayerNorm(config.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, hidden),
            nn.SiLU(),
            nn.Dropout(config.evidence_dropout),
            nn.Linear(hidden, config.model_dim),
        )
        self.mlp_dropout = nn.Dropout(config.evidence_dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError("V29 evidence tensors must be [B,N,D] and [B,T,D]")
        if query.shape[0] != context.shape[0] or query.shape[2] != context.shape[2]:
            raise ValueError("V29 evidence query and context do not align")
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + self.attention_dropout(attended)
        return query + self.mlp_dropout(self.mlp(self.mlp_norm(query)))


class ConditionalVisualDensityRatioModel(nn.Module):
    """Score candidate-specific evidence from an image-only writing stream."""

    _BACKBONE_MODULES = (
        "retina",
        "semantic_adapter",
        "target_semantic_adapter",
        "context_input",
        "context_blocks",
        "context_norm",
    )

    def __init__(self, config: ConditionalVisualDensityRatioConfig) -> None:
        super().__init__()
        self.config = config
        retina_config = VisualSaccadeConfig(
            fovea_size=config.cell_size,
            visual_dim=config.visual_dim,
            state_dim=config.model_dim,
            state_layers=1,
            retina_base_channels=config.retina_base_channels,
            ink_base_channels=32,
            dropout=0.0,
        )
        self.retina = FovealRetina(retina_config)
        self.semantic_adapter = ResidualSemanticAdapter(
            config.visual_dim,
            config.semantic_hidden_dim,
            config.semantic_residual_scale,
        )
        self.target_semantic_adapter = copy.deepcopy(self.semantic_adapter)

        self.context_input = nn.Linear(
            config.visual_dim + config.semantic_dim,
            config.model_dim,
            bias=False,
        )
        self.context_blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.context_norm = RMSNorm(config.model_dim)

        self.candidate_projection = nn.Linear(
            config.visual_dim + config.semantic_dim,
            config.model_dim,
            bias=False,
        )
        self.evidence_blocks = nn.ModuleList(
            [CandidateEvidenceLayer(config) for _ in range(config.evidence_layers)]
        )
        relation_input = config.model_dim * 3
        self.relation_norm = nn.LayerNorm(relation_input)
        self.relation_head = nn.Sequential(
            nn.Linear(relation_input, config.relation_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.relation_hidden_dim, 1),
        )
        self._initialize_new_path()
        self._freeze_perception()

    def _initialize_new_path(self) -> None:
        roots: tuple[nn.Module, ...] = (
            self.candidate_projection,
            self.evidence_blocks,
            self.relation_head,
        )
        for root in roots:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.evidence_layers)
        for block in self.evidence_blocks:
            block.attention.out_proj.weight.data.mul_(residual_scale)
            block.mlp[-1].weight.data.mul_(residual_scale)

    def _freeze_perception(self) -> None:
        for module in (
            self.retina,
            self.semantic_adapter,
            self.target_semantic_adapter,
        ):
            module.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> ConditionalVisualDensityRatioModel:
        super().train(mode)
        self._freeze_perception()
        return self

    @staticmethod
    def _validate_images(images: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(images):
            raise TypeError(f"V29 {name} must be a floating image tensor")
        if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V29 {name} must end in [1,32,32]")

    @torch.no_grad()
    def encode_image_parts(
        self,
        images: torch.Tensor,
        *,
        target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(images, name="images")
        leading = images.shape[:-3]
        flat = images.reshape(-1, 1, 32, 32).clamp(0, 1)
        raw = F.normalize(self.retina(flat).float(), dim=-1)
        adapter = (
            self.target_semantic_adapter if target else self.semantic_adapter
        )
        semantic = F.normalize(adapter(raw).float(), dim=-1)
        return (
            raw.reshape(*leading, self.config.visual_dim),
            semantic.reshape(*leading, self.config.semantic_dim),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_images(context, name="context")
        if context.ndim != 5:
            raise ValueError("V29 context must have shape [B,T,1,32,32]")
        if not 1 <= context.shape[1] <= self.config.maximum_cells:
            raise ValueError("V29 context length must be in [1,64]")
        raw, semantic = self.encode_image_parts(context, target=False)
        combined = torch.cat((raw, semantic), dim=-1)
        state = self.context_input(combined.to(self.context_input.weight.dtype))
        for block in self.context_blocks:
            state = block(state)
        return self.context_norm(state)

    def _candidate_queries(
        self,
        raw: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        if raw.shape[:-1] != semantic.shape[:-1]:
            raise ValueError("V29 raw and semantic candidate shapes do not align")
        if raw.shape[-1] != self.config.visual_dim:
            raise ValueError("V29 raw candidate dimension is invalid")
        if semantic.shape[-1] != self.config.semantic_dim:
            raise ValueError("V29 semantic candidate dimension is invalid")
        combined = torch.cat((raw, semantic), dim=-1)
        return self.candidate_projection(
            combined.to(self.candidate_projection.weight.dtype)
        )

    def _score_query_chunk(
        self,
        context: torch.Tensor,
        query0: torch.Tensor,
    ) -> torch.Tensor:
        query = query0
        for block in self.evidence_blocks:
            query = block(query, context)
        relation = torch.cat((query, query - query0, query * query0), dim=-1)
        return self.relation_head(self.relation_norm(relation))[..., 0].float()

    def score_encoded_shared(
        self,
        context_state: torch.Tensor,
        candidate_raw: torch.Tensor,
        candidate_semantic: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if context_state.ndim != 3:
            raise ValueError("V29 context state must be [B,T,D]")
        if candidate_raw.ndim != 2 or candidate_semantic.ndim != 2:
            raise ValueError("V29 shared candidate features must be [N,D]")
        if candidate_raw.shape[0] != candidate_semantic.shape[0]:
            raise ValueError("V29 shared candidate counts do not align")
        width = self.config.score_chunk_size if chunk_size is None else chunk_size
        if width < 1:
            raise ValueError("V29 score chunk must be positive")
        scores: list[torch.Tensor] = []
        for start in range(0, candidate_raw.shape[0], width):
            query = self._candidate_queries(
                candidate_raw[start : start + width],
                candidate_semantic[start : start + width],
            )
            query = query.unsqueeze(0).expand(context_state.shape[0], -1, -1)
            scores.append(self._score_query_chunk(context_state, query))
        return torch.cat(scores, dim=1)

    def score_encoded_paired(
        self,
        context_state: torch.Tensor,
        candidate_raw: torch.Tensor,
        candidate_semantic: torch.Tensor,
    ) -> torch.Tensor:
        if context_state.ndim != 4:
            raise ValueError("V29 paired context state must be [B,Q,T,D]")
        if candidate_raw.ndim != 3 or candidate_semantic.ndim != 3:
            raise ValueError("V29 paired candidate features must be [B,K,D]")
        batch, queries, length, dimension = context_state.shape
        if candidate_raw.shape[0] != batch or candidate_semantic.shape[0] != batch:
            raise ValueError("V29 paired batch dimensions do not align")
        candidate_query = self._candidate_queries(candidate_raw, candidate_semantic)
        candidate_query = candidate_query[:, None].expand(-1, queries, -1, -1)
        flat_context = context_state.reshape(batch * queries, length, dimension)
        flat_query = candidate_query.reshape(
            batch * queries, candidate_query.shape[2], dimension
        )
        score = self._score_query_chunk(flat_context, flat_query)
        return score.reshape(batch, queries, -1)

    def score_encoded_batched(
        self,
        context_state: torch.Tensor,
        candidate_raw: torch.Tensor,
        candidate_semantic: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Score a different candidate-image bank for each context row."""

        if context_state.ndim != 3:
            raise ValueError("V29 batched context state must be [B,T,D]")
        if candidate_raw.ndim != 3 or candidate_semantic.ndim != 3:
            raise ValueError("V29 batched candidate features must be [B,N,D]")
        if candidate_raw.shape[:2] != candidate_semantic.shape[:2]:
            raise ValueError("V29 batched candidate shapes do not align")
        width = self.config.score_chunk_size if chunk_size is None else chunk_size
        if width < 1:
            raise ValueError("V29 score chunk must be positive")
        scores: list[torch.Tensor] = []
        for start in range(0, candidate_raw.shape[1], width):
            query = self._candidate_queries(
                candidate_raw[:, start : start + width],
                candidate_semantic[:, start : start + width],
            )
            scores.append(self._score_query_chunk(context_state, query))
        return torch.cat(scores, dim=1)

    def score_shared_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError("V29 shared candidates must be [N,1,32,32]")
        state = self.encode_context(context)
        raw, semantic = self.encode_image_parts(candidates, target=True)
        return self.score_encoded_shared(
            state, raw, semantic, chunk_size=chunk_size
        )

    def score_paired_candidates(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or candidates.ndim != 5:
            raise ValueError(
                "V29 pairs must be [B,Q,T,1,32,32] and [B,K,1,32,32]"
            )
        if contexts.shape[0] != candidates.shape[0]:
            raise ValueError("V29 pair batches do not align")
        batch, queries = contexts.shape[:2]
        state = self.encode_context(
            contexts.reshape(batch * queries, *contexts.shape[2:])
        ).reshape(batch, queries, contexts.shape[2], self.config.model_dim)
        raw, semantic = self.encode_image_parts(candidates, target=True)
        return self.score_encoded_paired(state, raw, semantic)

    def score_exact_suffix_paired(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score one verified shared suffix and broadcast its exact score row."""

        if contexts.ndim != 6 or contexts.shape[1] != 2:
            raise ValueError("V29 exact suffix contexts must be [B,2,T,1,32,32]")
        suffix = contexts[:, :, -self.config.suffix_cells :]
        if not torch.equal(suffix[:, 0], suffix[:, 1]):
            raise ValueError("V29 exact suffix pixels differ between pair rows")
        score = self.score_paired_candidates(
            suffix[:, :1], candidates
        )
        return score.expand(-1, 2, -1)

    def load_v28_backbone_state(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        loaded: dict[str, int] = {}
        for name in self._BACKBONE_MODULES:
            prefix = f"{name}."
            module_state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not module_state:
                raise ValueError(f"V29 V28 source contains no {name} state")
            module = getattr(self, name)
            module.load_state_dict(module_state, strict=True)
            loaded[name] = len(module_state)
        self._freeze_perception()
        return {
            "source_architecture": "dense-visual-future-energy-v28",
            "loaded_modules": loaded,
            "discarded_future_heads": True,
        }


def row_center_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim < 2 or scores.shape[-1] < 2:
        raise ValueError("V29 score rows require at least two candidates")
    return scores.float() - scores.float().mean(dim=-1, keepdim=True)


def per_row_assignment_margin(
    logits: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 2):
        raise ValueError("V29 assignment logits must be [B,2,2]")
    if assignments.shape != logits.shape[:2] or assignments.dtype != torch.long:
        raise ValueError("V29 assignments must be int64 [B,2]")
    correct = logits.gather(2, assignments[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignments)[:, :, None])[:, :, 0]
    return correct.float() - other.float()


def conditional_visual_density_ratio_config_payload(
    config: ConditionalVisualDensityRatioConfig,
) -> dict[str, Any]:
    return asdict(config)


def conditional_visual_density_ratio_config_from_payload(
    payload: Mapping[str, Any],
) -> ConditionalVisualDensityRatioConfig:
    return ConditionalVisualDensityRatioConfig(**dict(payload))


def conditional_visual_density_ratio_boundary_receipt(
    config: ConditionalVisualDensityRatioConfig,
) -> dict[str, bool | str | list[int]]:
    return {
        "architecture": V29_ARCHITECTURE,
        "context_shape": [config.maximum_cells, 1, 32, 32],
        "candidate_shape": [1, 32, 32],
        "input_is_continuous_image_stream": True,
        "candidate_is_arbitrary_image": True,
        "output_is_candidate_conditioned_visual_energy": True,
        "retina_is_frozen": True,
        "semantic_adapters_are_frozen": True,
        "candidate_bank_deployed": False,
        "candidate_bank_in_model_state": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_embedding": False,
        "uses_vocabulary_output": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
    }


__all__ = [
    "CandidateEvidenceLayer",
    "ConditionalVisualDensityRatioConfig",
    "ConditionalVisualDensityRatioModel",
    "V29_ARCHITECTURE",
    "V29_SUFFIX_CELLS",
    "conditional_visual_density_ratio_boundary_receipt",
    "conditional_visual_density_ratio_config_from_payload",
    "conditional_visual_density_ratio_config_payload",
    "paired_assignment_loss",
    "per_row_assignment_margin",
    "row_center_scores",
]

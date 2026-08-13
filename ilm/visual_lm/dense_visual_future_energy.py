from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_visual_compatibility import paired_assignment_loss
from .saccade_lm import FovealRetina, VisualSaccadeConfig
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V28_HORIZONS = (1, 2, 4)


@dataclass(frozen=True)
class DenseVisualFutureConfig:
    cell_size: int = 32
    maximum_cells: int = 64
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
    hypotheses: int = 4
    horizons: tuple[int, ...] = V28_HORIZONS
    initial_raw_temperature: float = 0.08
    initial_semantic_temperature: float = 0.08

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V28 requires 32x32 visual cells")
        if self.maximum_cells != 64:
            raise ValueError("V28 fixes the maximum context to 64 cells")
        if self.visual_dim < 64 or self.semantic_dim < 64:
            raise ValueError("V28 visual dimensions are underspecified")
        if self.model_dim < 128 or self.model_dim % self.heads:
            raise ValueError("V28 causal width must divide into attention heads")
        if self.layers < 1 or self.heads < 1 or self.mlp_ratio < 2.0:
            raise ValueError("V28 causal field configuration is invalid")
        if self.retina_base_channels < 8 or self.semantic_hidden_dim < 8:
            raise ValueError("V28 visual encoders are underspecified")
        if not 0.0 < self.semantic_residual_scale <= 1.0:
            raise ValueError("V28 semantic residual scale must be in (0,1]")
        if self.hypotheses < 2:
            raise ValueError("V28 requires multiple continuous hypotheses")
        if tuple(self.horizons) != V28_HORIZONS:
            raise ValueError(f"V28 fixes future horizons to {V28_HORIZONS}")
        for value in (
            self.initial_raw_temperature,
            self.initial_semantic_temperature,
        ):
            if not 0.01 <= value <= 1.0:
                raise ValueError("V28 initial temperatures must be in [0.01,1]")


class ResidualSemanticAdapter(nn.Module):
    """Learn visual semantics while starting as an exact identity map."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(dimension)
        self.up = nn.Linear(dimension, hidden_dimension)
        self.down = nn.Linear(hidden_dimension, dimension, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.weight)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        update = self.down(F.silu(self.up(self.norm(raw))))
        return raw + self.residual_scale * update


class DenseVisualFutureModel(nn.Module):
    """Predict multi-horizon continuous writing from an image-only stream."""

    def __init__(self, config: DenseVisualFutureConfig) -> None:
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
        self.retina.requires_grad_(False).eval()
        if config.semantic_dim != config.visual_dim:
            raise ValueError("V28 identity initialization requires equal visual dimensions")
        self.semantic_adapter = ResidualSemanticAdapter(
            config.visual_dim,
            config.semantic_hidden_dim,
            config.semantic_residual_scale,
        )
        self.target_semantic_adapter = copy.deepcopy(self.semantic_adapter)
        self.target_semantic_adapter.requires_grad_(False).eval()

        self.context_input = nn.Linear(
            config.visual_dim + config.semantic_dim,
            config.model_dim,
            bias=False,
        )
        self.context_blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.context_norm = RMSNorm(config.model_dim)
        self.horizon_embedding = nn.Parameter(
            torch.empty(len(config.horizons), config.model_dim)
        )
        self.future_trunk = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
        )
        self.raw_queries = nn.Linear(
            config.model_dim,
            config.hypotheses * config.visual_dim,
            bias=False,
        )
        self.semantic_queries = nn.Linear(
            config.model_dim,
            config.hypotheses * config.semantic_dim,
            bias=False,
        )
        self.mixture_logits = nn.Linear(
            config.model_dim, config.hypotheses
        )
        self.log_raw_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_raw_temperature))
        )
        self.log_semantic_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_semantic_temperature))
        )
        self._initialize_language_path()

    def _initialize_language_path(self) -> None:
        nn.init.normal_(self.horizon_embedding, std=0.02)
        roots: list[nn.Module] = [
            self.context_input,
            self.context_blocks,
            self.future_trunk,
            self.raw_queries,
            self.semantic_queries,
            self.mixture_logits,
        ]
        for root in roots:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * max(1, self.config.layers))
        for block in self.context_blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def train(self, mode: bool = True) -> DenseVisualFutureModel:
        super().train(mode)
        self.retina.eval()
        self.target_semantic_adapter.eval()
        return self

    @property
    def raw_scale(self) -> torch.Tensor:
        return self.log_raw_scale.exp().clamp(max=100.0)

    @property
    def semantic_scale(self) -> torch.Tensor:
        return self.log_semantic_scale.exp().clamp(max=100.0)

    @staticmethod
    def _validate_images(images: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(images):
            raise TypeError(f"V28 {name} must be a floating image tensor")
        if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V28 {name} must end in [1,32,32]")

    def encode_image_parts(
        self,
        images: torch.Tensor,
        *,
        target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(images, name="images")
        leading = images.shape[:-3]
        flat = images.reshape(-1, 1, 32, 32).clamp(0, 1)
        with torch.no_grad():
            raw = F.normalize(self.retina(flat).float(), dim=-1)
        adapter = (
            self.target_semantic_adapter if target else self.semantic_adapter
        )
        if target:
            with torch.no_grad():
                semantic = F.normalize(adapter(raw).float(), dim=-1)
        else:
            semantic = F.normalize(
                adapter(raw.to(adapter.up.weight.dtype)).float(), dim=-1
            )
        return (
            raw.reshape(*leading, self.config.visual_dim),
            semantic.reshape(*leading, self.config.semantic_dim),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_images(context, name="context")
        if context.ndim != 5:
            raise ValueError("V28 context must have shape [B,T,1,32,32]")
        if not 1 <= context.shape[1] <= self.config.maximum_cells:
            raise ValueError("V28 context length must be in [1,64]")
        raw, semantic = self.encode_image_parts(context, target=False)
        combined = torch.cat((raw, semantic), dim=-1)
        state = self.context_input(combined.to(self.context_input.weight.dtype))
        for block in self.context_blocks:
            state = block(state)
        return self.context_norm(state)

    def future_distribution(
        self,
        state: torch.Tensor,
        *,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        if state.ndim < 2 or state.shape[-1] != self.config.model_dim:
            raise ValueError("V28 state must end in model_dim")
        try:
            horizon_index = self.config.horizons.index(int(horizon))
        except ValueError as exc:
            raise ValueError(f"unknown V28 horizon {horizon}") from exc
        hidden = self.future_trunk(
            state + self.horizon_embedding[horizon_index].to(state.dtype)
        )
        leading = hidden.shape[:-1]
        raw = self.raw_queries(hidden).reshape(
            *leading, self.config.hypotheses, self.config.visual_dim
        )
        semantic = self.semantic_queries(hidden).reshape(
            *leading, self.config.hypotheses, self.config.semantic_dim
        )
        return {
            "raw_queries": F.normalize(raw.float(), dim=-1),
            "semantic_queries": F.normalize(semantic.float(), dim=-1),
            "mixture_logits": self.mixture_logits(hidden).float(),
        }

    def score_distribution_shared(
        self,
        distribution: Mapping[str, torch.Tensor],
        candidate_raw: torch.Tensor,
        candidate_semantic: torch.Tensor,
    ) -> torch.Tensor:
        raw_queries = distribution["raw_queries"]
        semantic_queries = distribution["semantic_queries"]
        mixture = distribution["mixture_logits"]
        if candidate_raw.ndim != 2 or candidate_raw.shape[1] != self.config.visual_dim:
            raise ValueError("V28 shared raw candidates must be [N,visual_dim]")
        if candidate_semantic.ndim != 2 or candidate_semantic.shape != (
            candidate_raw.shape[0],
            self.config.semantic_dim,
        ):
            raise ValueError("V28 shared semantic candidates do not align")
        raw_score = torch.einsum(
            "...kd,nd->...kn", raw_queries.float(), candidate_raw.float()
        )
        semantic_score = torch.einsum(
            "...kd,nd->...kn",
            semantic_queries.float(),
            candidate_semantic.float(),
        )
        component = self.raw_scale.float() * raw_score
        component = component + self.semantic_scale.float() * semantic_score
        log_mixture = mixture.log_softmax(dim=-1)[..., :, None]
        return torch.logsumexp(log_mixture + component, dim=-2)

    def score_shared_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        *,
        horizon: int = 1,
    ) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError("V28 shared candidates must be [N,1,32,32]")
        state = self.encode_context(context)[:, -1]
        distribution = self.future_distribution(state, horizon=horizon)
        candidate_raw, candidate_semantic = self.encode_image_parts(
            candidates, target=True
        )
        return self.score_distribution_shared(
            distribution, candidate_raw, candidate_semantic
        )

    def score_paired_candidates(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
        *,
        horizon: int = 1,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or candidates.ndim != 5:
            raise ValueError(
                "V28 pairs must be [B,Q,T,1,32,32] and [B,K,1,32,32]"
            )
        if contexts.shape[0] != candidates.shape[0]:
            raise ValueError("V28 pair batches must align")
        batch, queries = contexts.shape[:2]
        state = self.encode_context(
            contexts.reshape(batch * queries, *contexts.shape[2:])
        )[:, -1].reshape(batch, queries, -1)
        distribution = self.future_distribution(state, horizon=horizon)
        candidate_raw, candidate_semantic = self.encode_image_parts(
            candidates, target=True
        )
        raw_score = torch.einsum(
            "bqhd,bkd->bqhk",
            distribution["raw_queries"].float(),
            candidate_raw.float(),
        )
        semantic_score = torch.einsum(
            "bqhd,bkd->bqhk",
            distribution["semantic_queries"].float(),
            candidate_semantic.float(),
        )
        component = self.raw_scale.float() * raw_score
        component = component + self.semantic_scale.float() * semantic_score
        log_mixture = distribution["mixture_logits"].log_softmax(dim=-1)
        return torch.logsumexp(log_mixture[..., :, None] + component, dim=-2)

    @torch.no_grad()
    def update_target_adapter(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("V28 EMA momentum must be in [0,1]")
        online = dict(self.semantic_adapter.named_parameters())
        for name, target in self.target_semantic_adapter.named_parameters():
            target.lerp_(online[name], 1.0 - momentum)
        online_buffers = dict(self.semantic_adapter.named_buffers())
        for name, target in self.target_semantic_adapter.named_buffers():
            target.copy_(online_buffers[name])


def weighted_multi_positive_nce(
    logits: torch.Tensor,
    query_groups: torch.Tensor,
    candidate_groups: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 2:
        raise ValueError("V28 contrastive logits must be [queries,candidates]")
    if query_groups.shape != (logits.shape[0],):
        raise ValueError("V28 query groups do not align with logits")
    if candidate_groups.shape != (logits.shape[1],):
        raise ValueError("V28 candidate groups do not align with logits")
    if query_groups.dtype != torch.long or candidate_groups.dtype != torch.long:
        raise TypeError("V28 temporary pixel groups must be int64")
    positives = query_groups[:, None] == candidate_groups[None]
    if not positives.any(dim=1).all():
        raise ValueError("every V28 contrastive query requires a positive image")
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    row_loss = -(
        torch.logsumexp(positive_logits.float(), dim=1)
        - torch.logsumexp(logits.float(), dim=1)
    )
    if weights is None:
        weights = torch.ones_like(row_loss)
    if weights.shape != row_loss.shape or bool((weights <= 0).any()):
        raise ValueError("V28 contrastive weights must be positive per query")
    loss = (row_loss * weights.float()).sum() / weights.float().sum()
    prediction = logits.argmax(dim=1)
    accuracy = positives[
        torch.arange(logits.shape[0], device=logits.device), prediction
    ].float()
    weighted_accuracy = (
        accuracy * weights.float()
    ).sum() / weights.float().sum()
    positive_log_probability = -row_loss
    return loss, {
        "loss": loss.detach(),
        "top1": weighted_accuracy.detach(),
        "positive_log_probability": (
            positive_log_probability * weights.float()
        ).sum().detach()
        / weights.float().sum(),
    }


def mixture_energy_score(
    raw_queries: torch.Tensor,
    mixture_logits: torch.Tensor,
    target_raw: torch.Tensor,
) -> torch.Tensor:
    if raw_queries.ndim < 3:
        raise ValueError("V28 raw hypotheses must end in [K,D]")
    if mixture_logits.shape != raw_queries.shape[:-1]:
        raise ValueError("V28 mixture logits do not align with hypotheses")
    if target_raw.shape != raw_queries.shape[:-2] + (raw_queries.shape[-1],):
        raise ValueError("V28 energy targets do not align with hypotheses")
    probabilities = mixture_logits.float().softmax(dim=-1)
    target_similarity = torch.einsum(
        "...kd,...d->...k", raw_queries.float(), target_raw.float()
    )
    target_distance = (
        2.0 - 2.0 * target_similarity.clamp(-1.0, 1.0)
    ).clamp_min(1e-12).sqrt()
    pair_similarity = torch.einsum(
        "...kd,...ld->...kl", raw_queries.float(), raw_queries.float()
    )
    pair_distance = (
        2.0 - 2.0 * pair_similarity.clamp(-1.0, 1.0)
    ).clamp_min(1e-12).sqrt()
    attraction = (probabilities * target_distance).sum(dim=-1)
    repulsion = 0.5 * (
        probabilities[..., :, None]
        * probabilities[..., None, :]
        * pair_distance
    ).sum(dim=(-2, -1))
    return attraction - repulsion


def assignment_margin(
    logits: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 2):
        raise ValueError("V28 assignment logits must be [B,2,2]")
    if assignments.shape != logits.shape[:2] or assignments.dtype != torch.long:
        raise ValueError("V28 assignments must be int64 [B,2]")
    correct = logits.gather(2, assignments[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignments)[:, :, None])[:, :, 0]
    return (correct - other).mean(dim=1)


def dense_visual_future_config_payload(
    config: DenseVisualFutureConfig,
) -> dict[str, Any]:
    return asdict(config)


def dense_visual_future_config_from_payload(
    payload: Mapping[str, Any],
) -> DenseVisualFutureConfig:
    values = dict(payload)
    if "horizons" in values:
        values["horizons"] = tuple(values["horizons"])
    return DenseVisualFutureConfig(**values)


def dense_visual_future_boundary_receipt(
    config: DenseVisualFutureConfig,
) -> dict[str, bool | str | list[int]]:
    return {
        "architecture": "dense-visual-future-energy-v28",
        "context_shape": [config.maximum_cells, 1, 32, 32],
        "candidate_shape": [1, 32, 32],
        "future_horizons": list(config.horizons),
        "input_is_continuous_image_stream": True,
        "candidate_is_arbitrary_image": True,
        "output_is_continuous_future_distribution": True,
        "retina_is_frozen": True,
        "target_semantic_route_is_ema": True,
        "candidate_bank_deployed": False,
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
    "DenseVisualFutureConfig",
    "DenseVisualFutureModel",
    "ResidualSemanticAdapter",
    "V28_HORIZONS",
    "assignment_margin",
    "dense_visual_future_boundary_receipt",
    "dense_visual_future_config_from_payload",
    "dense_visual_future_config_payload",
    "mixture_energy_score",
    "paired_assignment_loss",
    "weighted_multi_positive_nce",
]

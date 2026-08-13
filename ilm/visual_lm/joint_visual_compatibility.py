from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .saccade_lm import FovealRetina, VisualSaccadeConfig
from .visual_cell_stream import CausalVisualBlock, RMSNorm


@dataclass(frozen=True)
class JointVisualCompatibilityConfig:
    cell_size: int = 32
    maximum_cells: int = 64
    visual_dim: int = 192
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    retina_base_channels: int = 64
    candidate_hidden_dim: int = 384
    candidate_residual_scale: float = 0.10
    initial_temperature: float = 0.08

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V27 requires 32x32 visual cells")
        if self.maximum_cells != 64:
            raise ValueError("V27 fixes the maximum context to 64 cells")
        if self.visual_dim < 64 or self.model_dim < 128:
            raise ValueError("V27 visual dimensions are underspecified")
        if self.layers < 1 or self.heads < 1 or self.model_dim % self.heads:
            raise ValueError("V27 causal layer/head configuration is invalid")
        if self.mlp_ratio < 2.0:
            raise ValueError("V27 MLP ratio must be at least two")
        if self.retina_base_channels < 8 or self.candidate_hidden_dim < 8:
            raise ValueError("V27 visual projection is underspecified")
        if not 0.0 < self.candidate_residual_scale <= 1.0:
            raise ValueError("V27 candidate residual scale must be in (0,1]")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V27 initial temperature is invalid")


class ResidualVisualProjector(nn.Module):
    """Adapt retinal geometry while starting as an exact identity map."""

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

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        update = self.down(F.silu(self.up(self.norm(visual))))
        return visual + self.residual_scale * update


class JointVisualCompatibilityModel(nn.Module):
    """Score arbitrary candidate glyph images from a causal image history."""

    def __init__(self, config: JointVisualCompatibilityConfig) -> None:
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
        self.target_retina = copy.deepcopy(self.retina)
        self.target_retina.requires_grad_(False).eval()

        self.context_input = nn.Linear(
            config.visual_dim, config.model_dim, bias=False
        )
        self.context_blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.context_norm = RMSNorm(config.model_dim)
        self.query_projector = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.visual_dim, bias=False),
        )
        self.candidate_projector = ResidualVisualProjector(
            config.visual_dim,
            config.candidate_hidden_dim,
            config.candidate_residual_scale,
        )
        self.target_candidate_projector = copy.deepcopy(
            self.candidate_projector
        )
        self.target_candidate_projector.requires_grad_(False).eval()
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )
        self._initialize_context_path()

    def _initialize_context_path(self) -> None:
        roots: list[nn.Module] = [
            self.context_input,
            self.context_blocks,
            self.query_projector,
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

    def train(self, mode: bool = True) -> "JointVisualCompatibilityModel":
        super().train(mode)
        self.target_retina.eval()
        self.target_candidate_projector.eval()
        return self

    @property
    def compatibility_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    @staticmethod
    def _validate_images(images: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(images):
            raise TypeError(f"V27 {name} must be a floating image tensor")
        if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V27 {name} must end in [1,32,32]")

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_images(context, name="context")
        if context.ndim != 5:
            raise ValueError("V27 context must have shape [B,T,1,32,32]")
        if not 1 <= context.shape[1] <= self.config.maximum_cells:
            raise ValueError("V27 context length must be in [1,64]")
        batch, length = context.shape[:2]
        visual = self.retina(
            context.reshape(batch * length, 1, 32, 32).clamp(0, 1)
        )
        visual = F.normalize(visual.float(), dim=-1).reshape(batch, length, -1)
        state = self.context_input(visual.to(self.context_input.weight.dtype))
        for block in self.context_blocks:
            state = block(state)
        query = self.query_projector(self.context_norm(state[:, -1]))
        return F.normalize(query.float(), dim=-1)

    def encode_candidates(
        self,
        images: torch.Tensor,
        *,
        target: bool,
        normalize: bool = True,
    ) -> torch.Tensor:
        self._validate_images(images, name="candidates")
        leading = images.shape[:-3]
        flat = images.reshape(-1, 1, 32, 32).clamp(0, 1)
        retina = self.target_retina if target else self.retina
        projector = (
            self.target_candidate_projector
            if target
            else self.candidate_projector
        )
        visual = retina(flat)
        latent = projector(visual)
        if normalize:
            latent = F.normalize(latent.float(), dim=-1)
        return latent.reshape(*leading, self.config.visual_dim)

    def score_shared_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score one shared image bank with shape [N,1,32,32]."""

        if candidates.ndim != 4:
            raise ValueError("shared V27 candidates must have shape [N,1,32,32]")
        query = self.encode_context(context)
        with torch.no_grad():
            keys = self.encode_candidates(candidates, target=True)
        return self.compatibility_scale.float() * torch.einsum(
            "bd,nd->bn", query.float(), keys.float()
        )

    def score_paired_candidates(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score per-example context/candidate sets as [B,Q,K]."""

        if contexts.ndim != 6 or candidates.ndim != 5:
            raise ValueError(
                "paired V27 inputs must be [B,Q,T,1,32,32] and "
                "[B,K,1,32,32]"
            )
        if contexts.shape[0] != candidates.shape[0]:
            raise ValueError("V27 pair batches must align")
        batch, queries = contexts.shape[:2]
        query = self.encode_context(
            contexts.reshape(batch * queries, *contexts.shape[2:])
        ).reshape(batch, queries, -1)
        with torch.no_grad():
            keys = self.encode_candidates(candidates, target=True)
        return self.compatibility_scale.float() * torch.einsum(
            "bqd,bkd->bqk", query.float(), keys.float()
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("V27 EMA momentum must be in [0,1]")
        for online_module, target_module in (
            (self.retina, self.target_retina),
            (self.candidate_projector, self.target_candidate_projector),
        ):
            online = dict(online_module.named_parameters())
            for name, target in target_module.named_parameters():
                target.lerp_(online[name], 1.0 - momentum)
            online_buffers = dict(online_module.named_buffers())
            for name, target in target_module.named_buffers():
                target.copy_(online_buffers[name])


def exact_image_identity_mask(
    row_images: torch.Tensor,
    column_images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derive positive relations from pixels, never from character identity."""

    column_images = row_images if column_images is None else column_images
    for name, images in (("rows", row_images), ("columns", column_images)):
        if not torch.is_floating_point(images):
            raise TypeError(f"V27 canonical {name} must be floating images")
        if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V27 canonical {name} must end in [1,32,32]")
    rows = row_images.reshape(row_images.shape[0], -1)
    columns = column_images.reshape(column_images.shape[0], -1)
    return (rows[:, None] == columns[None]).all(dim=-1)


def multi_positive_nce(
    logits: torch.Tensor,
    positives: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 2 or positives.shape != logits.shape:
        raise ValueError("V27 logits and positive mask must be matching matrices")
    if positives.dtype != torch.bool:
        raise TypeError("V27 positive mask must be boolean")
    if not positives.any(dim=1).all():
        raise ValueError("every V27 contrastive row requires a positive")
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    loss = -(
        torch.logsumexp(positive_logits.float(), dim=1)
        - torch.logsumexp(logits.float(), dim=1)
    ).mean()
    prediction = logits.argmax(dim=1)
    accuracy = positives[
        torch.arange(logits.shape[0], device=logits.device), prediction
    ].float().mean()
    return loss, {
        "multi_positive_nce": loss.detach(),
        "multi_positive_top1": accuracy.detach(),
    }


def paired_assignment_loss(
    logits: torch.Tensor,
    row_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Recover a candidate permutation in both context and image directions."""

    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 2):
        raise ValueError("V27 pair logits must have shape [B,2,2]")
    if row_targets.shape != logits.shape[:2] or row_targets.dtype != torch.long:
        raise ValueError("V27 pair targets must be int64 with shape [B,2]")
    expected = torch.arange(2, device=logits.device).expand_as(row_targets)
    if not torch.equal(row_targets.sort(dim=1).values, expected):
        raise ValueError("each V27 pair target row must be a permutation")
    column_targets = torch.empty_like(row_targets)
    column_targets.scatter_(1, row_targets, expected)
    row_loss = F.cross_entropy(logits.reshape(-1, 2), row_targets.reshape(-1))
    column_loss = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, 2), column_targets.reshape(-1)
    )
    loss = 0.5 * (row_loss + column_loss)
    correct = logits.gather(2, row_targets[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - row_targets)[:, :, None])[:, :, 0]
    margins = correct - other
    ties = margins == 0
    credit = (margins > 0).float() + 0.5 * ties.float()
    return loss, {
        "pair_loss": loss.detach(),
        "pair_arm_accuracy": credit.mean().detach(),
        "pair_strict_arm_accuracy": (margins > 0).float().mean().detach(),
        "pair_tie_rate": ties.float().mean().detach(),
        "pair_both_correct_rate": (margins > 0).all(dim=1).float().mean().detach(),
        "pair_mean_margin": margins.mean().detach(),
    }


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("V27 VICReg views must be matching [B,D] matrices")
    invariance = F.mse_loss(first.float(), second.float())
    variance = torch.zeros((), device=first.device, dtype=torch.float32)
    covariance = torch.zeros_like(variance)
    for value in (first.float(), second.float()):
        centered = value - value.mean(dim=0, keepdim=True)
        std = (centered.square().mean(dim=0) + 1e-4).sqrt()
        variance = variance + F.relu(1.0 - std).mean()
        denominator = max(1, value.shape[0] - 1)
        matrix = centered.transpose(0, 1) @ centered / denominator
        off_diagonal = matrix - torch.diag_embed(matrix.diagonal())
        covariance = covariance + off_diagonal.square().sum() / value.shape[1]
    variance = 0.5 * variance
    covariance = 0.5 * covariance
    total = 25.0 * invariance + 25.0 * variance + covariance
    return total, {
        "vicreg_loss": total.detach(),
        "vicreg_invariance": invariance.detach(),
        "vicreg_variance": variance.detach(),
        "vicreg_covariance": covariance.detach(),
    }


def joint_visual_compatibility_config_payload(
    config: JointVisualCompatibilityConfig,
) -> dict[str, Any]:
    return asdict(config)


def joint_visual_compatibility_config_from_payload(
    payload: dict[str, Any],
) -> JointVisualCompatibilityConfig:
    return JointVisualCompatibilityConfig(**payload)


def joint_visual_compatibility_boundary_receipt(
    config: JointVisualCompatibilityConfig,
) -> dict[str, bool | str | list[int]]:
    return {
        "architecture": "joint-visual-compatibility-v27",
        "context_shape": [config.maximum_cells, 1, 32, 32],
        "candidate_shape": [1, 32, 32],
        "input_is_continuous_image_stream": True,
        "candidate_is_arbitrary_image": True,
        "output_is_continuous_compatibility": True,
        "target_route_is_ema": True,
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
        "candidate_bank_deployed": False,
    }

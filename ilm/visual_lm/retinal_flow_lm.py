from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ink_writer import (
    FovealInkFlow,
    FovealWriterConfig,
    flow_training_state,
    foveal_flow_loss,
    sample_foveal_ink,
)
from .saccade_lm import FovealRetina, VisualSaccadeConfig


@dataclass(frozen=True)
class RetinalFlowConfig:
    """Configuration for image-valued recurrent language dynamics."""

    fovea_size: int = 32
    visual_dim: int = 192
    state_dim: int = 384
    state_layers: int = 3
    retina_base_channels: int = 64
    dropout: float = 0.05
    flow_base_channels: int = 64
    flow_context_dim: int = 256
    energy_dim: int = 256
    condition_dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("fovea_size must be a multiple of eight and at least 16")
        if self.visual_dim < 64 or self.state_dim < 128:
            raise ValueError("retinal flow state is underspecified")
        if self.state_layers < 1:
            raise ValueError("state_layers must be positive")
        if self.flow_base_channels < 16 or self.flow_context_dim < 64:
            raise ValueError("retinal flow writer is underspecified")
        if self.energy_dim < 64:
            raise ValueError("visual energy field is underspecified")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")


class VisualCompatibilityEnergy(nn.Module):
    """Score arbitrary candidate images against a recurrent visual state.

    Candidates are continuous retina embeddings. There is no classifier table,
    glyph index, or fixed vocabulary in this module.
    """

    def __init__(self, config: RetinalFlowConfig):
        super().__init__()
        condition_dim = config.state_dim + config.visual_dim
        self.energy_dim = config.energy_dim
        self.condition = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, config.energy_dim * 2),
            nn.SiLU(),
            nn.Linear(config.energy_dim * 2, config.energy_dim),
            nn.LayerNorm(config.energy_dim),
        )
        self.candidate = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.energy_dim * 2),
            nn.SiLU(),
            nn.Linear(config.energy_dim * 2, config.energy_dim),
            nn.LayerNorm(config.energy_dim),
        )
        self.interaction = nn.Sequential(
            nn.LayerNorm(config.energy_dim * 2),
            nn.Linear(config.energy_dim * 2, config.energy_dim),
            nn.SiLU(),
            nn.Linear(config.energy_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))

    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(self, condition: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("visual energy condition must have shape [batch, dimension]")
        if candidates.ndim not in (2, 3):
            raise ValueError("visual candidates must have shape [count, dim] or [batch, count, dim]")
        query = F.normalize(self.condition(condition.float()), dim=-1)
        keys = F.normalize(self.candidate(candidates.float()), dim=-1)
        if keys.ndim == 2:
            query_field = query[:, None, :]
            key_field = keys[None, :, :]
        else:
            if keys.shape[0] != query.shape[0]:
                raise ValueError("per-example candidates must share the condition batch dimension")
            query_field = query[:, None, :]
            key_field = keys
        query_field, key_field = torch.broadcast_tensors(query_field, key_field)
        interaction = torch.cat(
            (query_field * key_field, (query_field - key_field).abs()),
            dim=-1,
        )
        residual = self.interaction(interaction).squeeze(-1)
        cosine = (query_field * key_field).sum(dim=-1)
        return self.scale * (cosine + 0.25 * residual)


class RetinalFlowLanguageModel(nn.Module):
    """Read ordered ink fixations and write the next fixation as an image.

    The only external representation consumed or produced by this model is a
    continuous image tensor. Text rendering and OCR belong outside the model.
    """

    def __init__(self, config: RetinalFlowConfig):
        super().__init__()
        self.config = config
        retina_config = VisualSaccadeConfig(
            fovea_size=config.fovea_size,
            visual_dim=config.visual_dim,
            state_dim=config.state_dim,
            state_layers=config.state_layers,
            retina_base_channels=config.retina_base_channels,
            dropout=config.dropout,
            visual_hypotheses=1,
        )
        self.online_retina = FovealRetina(retina_config)
        self.target_retina = copy.deepcopy(self.online_retina)
        self.target_retina.requires_grad_(False).eval()
        self.dynamics = nn.GRU(
            config.visual_dim,
            config.state_dim,
            num_layers=config.state_layers,
            batch_first=True,
            dropout=config.dropout if config.state_layers > 1 else 0.0,
        )
        condition_dim = config.state_dim + config.visual_dim
        self.energy = VisualCompatibilityEnergy(config)
        self.writer = FovealInkFlow(
            FovealWriterConfig(
                fovea_size=config.fovea_size,
                condition_dim=condition_dim,
                base_channels=config.flow_base_channels,
                context_dim=config.flow_context_dim,
                condition_dropout=config.condition_dropout,
            )
        )
        self.retina_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))

    def train(self, mode: bool = True) -> "RetinalFlowLanguageModel":
        super().train(mode)
        self.target_retina.eval()
        return self

    @property
    def retina_scale(self) -> torch.Tensor:
        return self.retina_logit_scale.exp().clamp(max=100.0)

    def encode_sequence(self, foveas: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        if foveas.ndim != 5:
            raise ValueError("foveal sequence must have shape [batch, length, 1, size, size]")
        batch, length = foveas.shape[:2]
        retina = self.target_retina if target else self.online_retina
        encoded = retina(foveas.reshape(batch * length, *foveas.shape[2:]))
        return encoded.reshape(batch, length, -1)

    def predict(
        self,
        context_foveas: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual = self.encode_sequence(context_foveas)
        state, final_state = self.dynamics(visual, initial_state)
        return {
            "current_visual": visual,
            "state": state,
            "condition": torch.cat((state, visual), dim=-1),
            "final_state": final_state,
        }

    def score_visual_candidates(
        self,
        prediction: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        *,
        position: int = -1,
    ) -> torch.Tensor:
        return self.energy(prediction["condition"][:, position], candidates)

    def forward(
        self,
        context_foveas: torch.Tensor,
        target_foveas: torch.Tensor | None = None,
        current_reference_foveas: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.predict(context_foveas)
        with torch.no_grad():
            if target_foveas is not None:
                output["target_visual"] = self.encode_sequence(target_foveas, target=True)
            if current_reference_foveas is not None:
                output["current_reference_visual"] = self.encode_sequence(
                    current_reference_foveas,
                    target=True,
                )
        return output

    @torch.no_grad()
    def sample_next(
        self,
        context_foveas: torch.Tensor,
        *,
        samples_per_context: int = 1,
        steps: int = 8,
        guidance_scale: float = 1.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if samples_per_context < 1:
            raise ValueError("samples_per_context must be positive")
        prediction = self.predict(context_foveas)
        condition = prediction["condition"][:, -1].repeat_interleave(samples_per_context, dim=0)
        current = context_foveas[:, -1].repeat_interleave(samples_per_context, dim=0)
        sampled = sample_foveal_ink(
            self.writer,
            condition,
            current * 2.0 - 1.0,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        sampled = ((sampled + 1.0) * 0.5).clamp(0, 1)
        return sampled.reshape(
            context_foveas.shape[0],
            samples_per_context,
            1,
            self.config.fovea_size,
            self.config.fovea_size,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        online = dict(self.online_retina.named_parameters())
        for name, target in self.target_retina.named_parameters():
            target.lerp_(online[name], 1.0 - momentum)
        online_buffers = dict(self.online_retina.named_buffers())
        for name, target in self.target_retina.named_buffers():
            target.copy_(online_buffers[name])


def _multi_positive_nce(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("multi-positive logits and mask must be matching matrices")
    if not positive_mask.any(dim=1).all():
        raise ValueError("every query must have at least one visual positive")
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
    retrieval = positive_mask.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    return loss, retrieval


def _visual_positive_mask(
    candidates: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = F.normalize(candidates.float(), dim=-1)
    similarity = candidates @ candidates.transpose(0, 1)
    mask = similarity >= threshold
    mask.fill_diagonal_(True)
    return mask, similarity


def retinal_flow_loss(
    model: RetinalFlowLanguageModel,
    outputs: dict[str, torch.Tensor],
    context_foveas: torch.Tensor,
    target_ink: torch.Tensor,
    *,
    energy_positions_per_sequence: int = 8,
    duplicate_similarity: float = 0.90,
    flow_weight: float = 1.0,
    energy_weight: float = 0.60,
    invariance_weight: float = 0.20,
    retina_contrastive_weight: float = 0.25,
    retina_variance_weight: float = 0.10,
    endpoint_weight: float = 0.10,
    stroke_weight: float = 2.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch, length = context_foveas.shape[:2]
    if not 1 <= energy_positions_per_sequence <= length:
        raise ValueError("energy_positions_per_sequence must be within the causal sequence")
    batch_indices = torch.arange(batch, device=context_foveas.device)
    random_order = torch.rand(
        batch,
        length,
        device=context_foveas.device,
        generator=generator,
    ).argsort(dim=1)[:, :energy_positions_per_sequence]
    energy_batches = batch_indices[:, None].expand_as(random_order)
    energy_condition = outputs["condition"][energy_batches, random_order].flatten(0, 1)
    energy_target_visual = (
        outputs["target_visual"][energy_batches, random_order].float().detach().flatten(0, 1)
    )

    visual_positive, target_similarity = _visual_positive_mask(
        energy_target_visual,
        duplicate_similarity,
    )
    energy_logits = model.energy(energy_condition, energy_target_visual)
    energy, energy_retrieval = _multi_positive_nce(energy_logits, visual_positive)

    positions = random_order[:, 0]
    condition = outputs["condition"][batch_indices, positions]
    target_visual = outputs["target_visual"][batch_indices, positions].float().detach()
    current_visual = outputs["current_visual"][batch_indices, positions].float()
    current_reference = outputs["current_reference_visual"][batch_indices, positions].float()
    current_fovea = context_foveas[batch_indices, positions].float()
    target_fovea = target_ink[batch_indices, positions].float().clamp(0, 1)

    flow_target = target_fovea * 2.0 - 1.0
    flow_state, velocity, time_field, _ = flow_training_state(
        flow_target,
        generator=generator,
    )
    keep = (
        torch.rand(batch, device=context_foveas.device, generator=generator)
        >= model.config.condition_dropout
    ).to(flow_target.dtype)
    predicted_velocity = model.writer(
        flow_state,
        time_field,
        condition,
        current_fovea * 2.0 - 1.0,
        condition_present=keep,
    )
    flow, flow_metrics = foveal_flow_loss(
        predicted_velocity,
        velocity,
        flow_state,
        flow_target,
        time_field,
        endpoint_weight=endpoint_weight,
        stroke_weight=stroke_weight,
    )

    normalized_current = F.normalize(current_visual, dim=-1)
    normalized_reference = F.normalize(current_reference, dim=-1)
    invariance = (1.0 - (normalized_current * normalized_reference).sum(dim=-1)).mean()
    retina_positive, _ = _visual_positive_mask(current_reference, duplicate_similarity)
    retina_logits = model.retina_scale * normalized_current @ normalized_reference.transpose(0, 1)
    retina_contrastive, retina_retrieval = _multi_positive_nce(retina_logits, retina_positive)
    retina_centered = F.layer_norm(current_visual, (current_visual.shape[-1],))
    retina_std = torch.sqrt(retina_centered.var(dim=0, unbiased=False) + 1e-4)
    retina_variance = F.relu(0.75 - retina_std).mean()

    total = (
        flow_weight * flow
        + energy_weight * energy
        + invariance_weight * invariance
        + retina_contrastive_weight * retina_contrastive
        + retina_variance_weight * retina_variance
    )
    metrics = {
        **flow_metrics,
        "visual_energy_nce": energy.detach(),
        "visual_energy_top1": energy_retrieval.detach(),
        "visual_positive_count": visual_positive.float().sum(dim=1).mean().detach(),
        "visual_energy_queries": energy.new_tensor(float(energy_condition.shape[0])),
        "visual_off_diagonal_similarity": (
            (target_similarity.sum() - target_similarity.diagonal().sum())
            / max(1, energy_condition.shape[0] * (energy_condition.shape[0] - 1))
        ).detach(),
        "cross_render_invariance": invariance.detach(),
        "retina_contrastive_loss": retina_contrastive.detach(),
        "retina_contrastive_top1": retina_retrieval.detach(),
        "retina_variance_penalty": retina_variance.detach(),
        "retina_feature_std": retina_std.mean().detach(),
        "condition_keep_fraction": keep.mean().detach(),
    }
    selected = {
        "batch_indices": batch_indices,
        "positions": positions,
        "condition": condition,
        "current_fovea": current_fovea,
        "target_fovea": target_fovea,
        "target_visual": target_visual,
    }
    return total, metrics, selected


def retinal_flow_config_payload(config: RetinalFlowConfig) -> dict[str, Any]:
    return asdict(config)


def retinal_flow_config_from_payload(payload: dict[str, Any]) -> RetinalFlowConfig:
    return RetinalFlowConfig(**payload)

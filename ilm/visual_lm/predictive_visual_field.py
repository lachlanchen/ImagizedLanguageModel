from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .saccade_lm import FovealRetina, VisualSaccadeConfig


@dataclass(frozen=True)
class PredictiveVisualFieldConfig:
    """Configuration for continuous next-retina language dynamics."""

    fovea_size: int = 32
    visual_dim: int = 192
    state_dim: int = 384
    state_layers: int = 3
    retina_base_channels: int = 64
    dropout: float = 0.05
    flow_hidden_dim: int = 512
    flow_blocks: int = 4
    time_dim: int = 128
    proposal_hidden_dim: int = 512
    proposal_blocks: int = 2
    condition_dropout: float = 0.10
    sample_temperature: float = 0.08
    flow_geometry: str = "euclidean"

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("fovea_size must be a multiple of eight and at least 16")
        if self.visual_dim < 64 or self.state_dim < 128:
            raise ValueError("predictive visual state is underspecified")
        if self.state_layers < 1:
            raise ValueError("state_layers must be positive")
        if self.flow_hidden_dim < 128 or self.flow_blocks < 1:
            raise ValueError("retinal state flow is underspecified")
        if self.proposal_hidden_dim < 128 or self.proposal_blocks < 1:
            raise ValueError("continuous visual proposal is underspecified")
        if self.time_dim < 32 or self.time_dim % 2:
            raise ValueError("time_dim must be an even integer of at least 32")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        if self.sample_temperature <= 0.0:
            raise ValueError("sample_temperature must be positive")
        if self.flow_geometry not in {"euclidean", "hypersphere"}:
            raise ValueError("flow_geometry must be euclidean or hypersphere")


def sphere_exponential_map(
    point: torch.Tensor,
    tangent: torch.Tensor,
    *,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Move along a unit-sphere geodesic by a tangent displacement."""

    point = F.normalize(point.float(), dim=-1)
    tangent = tangent.float() - (tangent.float() * point).sum(dim=-1, keepdim=True) * point
    distance = tangent.norm(dim=-1, keepdim=True)
    direction = tangent / distance.clamp_min(epsilon)
    moved = distance.cos() * point + distance.sin() * direction
    near_zero = distance <= epsilon
    return F.normalize(torch.where(near_zero, point, moved), dim=-1)


def hyperspherical_flow_path(
    target: torch.Tensor,
    source: torch.Tensor,
    time: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical interpolation and its forward velocity from target to source."""

    target = F.normalize(target.float(), dim=-1)
    source = F.normalize(source.float(), dim=-1)
    if time.ndim == 1:
        time = time[:, None]
    if time.ndim != 2 or time.shape != target.shape[:1] + (1,):
        raise ValueError("hyperspherical flow time must have shape [batch] or [batch, 1]")
    cosine = (target * source).sum(dim=-1, keepdim=True).clamp(
        min=-1.0 + epsilon,
        max=1.0 - epsilon,
    )
    angle = cosine.acos()
    sine = angle.sin().clamp_min(epsilon)
    target_phase = (1.0 - time.float()) * angle
    source_phase = time.float() * angle
    point = (
        target_phase.sin() / sine * target
        + source_phase.sin() / sine * source
    )
    velocity = angle / sine * (
        -target_phase.cos() * target
        + source_phase.cos() * source
    )
    point = F.normalize(point, dim=-1)
    velocity = velocity - (velocity * point).sum(dim=-1, keepdim=True) * point
    return point, velocity


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(torch.linspace(0.0, math.log(1_000.0), half))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 2 and time.shape[1] == 1:
            time = time[:, 0]
        if time.ndim != 1:
            raise ValueError("flow time must have shape [batch] or [batch, 1]")
        phase = time.float()[:, None] * self.frequencies[None] * (2.0 * math.pi)
        return self.output(torch.cat((phase.sin(), phase.cos()), dim=-1))


class StateFlowResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.SiLU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return (hidden + self.block(hidden)) * (2.0**-0.5)


class ContinuousVisualProposal(nn.Module):
    """Predict one image-derived state before stochastic form refinement."""

    def __init__(self, config: PredictiveVisualFieldConfig):
        super().__init__()
        condition_dim = config.state_dim + config.visual_dim
        self.condition_dim = condition_dim
        self.input = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, config.proposal_hidden_dim),
        )
        self.blocks = nn.ModuleList(
            StateFlowResidualBlock(config.proposal_hidden_dim)
            for _ in range(config.proposal_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.proposal_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.proposal_hidden_dim, config.visual_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError("proposal condition must have shape [batch, condition_dim]")
        hidden = self.input(condition.float())
        for block in self.blocks:
            hidden = block(hidden)
        return F.normalize(self.output(hidden), dim=-1)


class RetinalStateFlow(nn.Module):
    """Conditional velocity field in an image-derived continuous state space."""

    def __init__(self, config: PredictiveVisualFieldConfig):
        super().__init__()
        self.config = config
        condition_dim = config.state_dim + config.visual_dim
        self.visual_dim = config.visual_dim
        self.state_norm = nn.LayerNorm(config.visual_dim)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.state_projection = nn.Linear(config.visual_dim, config.flow_hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, config.flow_hidden_dim)
        self.time_embedding = FourierTimeEmbedding(config.time_dim)
        self.time_projection = nn.Linear(config.time_dim, config.flow_hidden_dim)
        self.condition_present = nn.Linear(1, config.flow_hidden_dim, bias=False)
        self.null_condition = nn.Parameter(torch.zeros(condition_dim))
        self.blocks = nn.ModuleList(
            StateFlowResidualBlock(config.flow_hidden_dim)
            for _ in range(config.flow_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.flow_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.flow_hidden_dim, config.visual_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.visual_dim:
            raise ValueError("retinal flow state must have shape [batch, visual_dim]")
        if condition.ndim != 2 or condition.shape[0] != state.shape[0]:
            raise ValueError("retinal flow condition must share the state batch")
        if condition_present is None:
            condition_present = state.new_ones(state.shape[0])
        if condition_present.ndim == 2 and condition_present.shape[1] == 1:
            condition_present = condition_present[:, 0]
        if tuple(condition_present.shape) != (state.shape[0],):
            raise ValueError("condition_present must have shape [batch]")
        present = condition_present.float().clamp(0, 1)[:, None]
        null = self.null_condition[None].expand_as(condition)
        selected_condition = present * condition.float() + (1.0 - present) * null
        hidden = (
            self.state_projection(self.state_norm(state.float()))
            + self.condition_projection(self.condition_norm(selected_condition))
            + self.time_projection(self.time_embedding(time))
            + self.condition_present(present)
        )
        for block in self.blocks:
            hidden = block(hidden)
        velocity = self.output(hidden)
        if self.config.flow_geometry == "hypersphere":
            point = F.normalize(state.float(), dim=-1)
            velocity = velocity - (velocity * point).sum(dim=-1, keepdim=True) * point
        return velocity


class PredictiveVisualField(nn.Module):
    """Predict the next image-derived retinal state without a symbolic alphabet."""

    def __init__(self, config: PredictiveVisualFieldConfig):
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
        self.retina = FovealRetina(retina_config)
        self.retina.requires_grad_(False).eval()
        self.dynamics = nn.GRU(
            config.visual_dim,
            config.state_dim,
            num_layers=config.state_layers,
            batch_first=True,
            dropout=config.dropout if config.state_layers > 1 else 0.0,
        )
        self.visual_proposal = ContinuousVisualProposal(config)
        self.state_flow = RetinalStateFlow(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.sample_temperature))
        )

    def train(self, mode: bool = True) -> "PredictiveVisualField":
        super().train(mode)
        self.retina.eval()
        return self

    @property
    def sample_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("retinal images must have shape [batch, 1, size, size]")
        with torch.no_grad():
            return self.retina(images).float()

    def encode_sequence(self, foveas: torch.Tensor) -> torch.Tensor:
        if foveas.ndim != 5:
            raise ValueError("foveal sequence must have shape [batch, length, 1, size, size]")
        batch, length = foveas.shape[:2]
        encoded = self.encode_images(foveas.reshape(batch * length, *foveas.shape[2:]))
        return encoded.reshape(batch, length, self.config.visual_dim)

    def predict(
        self,
        context_foveas: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual = self.encode_sequence(context_foveas)
        state, final_state = self.dynamics(visual, initial_state)
        condition = torch.cat((state, visual), dim=-1)
        proposal = self.visual_proposal(
            condition.reshape(-1, condition.shape[-1])
        ).reshape(*condition.shape[:-1], self.config.visual_dim)
        return {
            "current_visual": visual,
            "state": state,
            "condition": condition,
            "proposal_visual": proposal,
            "final_state": final_state,
        }

    def forward(
        self,
        context_foveas: torch.Tensor,
        target_foveas: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.predict(context_foveas)
        if target_foveas is not None:
            output["target_visual"] = self.encode_sequence(target_foveas)
        return output

    def sample_states(
        self,
        condition: torch.Tensor,
        *,
        samples_per_context: int = 4,
        steps: int = 8,
        guidance_scale: float = 1.0,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("state-flow condition must have shape [batch, dimension]")
        if samples_per_context < 1 or steps < 1:
            raise ValueError("sample count and integration steps must be positive")
        batch = condition.shape[0]
        expected = (batch, samples_per_context, self.config.visual_dim)
        if noise is None:
            noise = torch.randn(
                expected,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        elif tuple(noise.shape) != expected:
            raise ValueError(f"state-flow noise must have shape {expected}")
        if self.config.flow_geometry == "hypersphere":
            noise = F.normalize(noise.float(), dim=-1)
        state = noise.float().reshape(batch * samples_per_context, self.config.visual_dim)
        repeated_condition = condition.float().repeat_interleave(samples_per_context, dim=0)
        delta = 1.0 / steps
        for index in range(steps, 0, -1):
            time = state.new_full((state.shape[0],), index / steps)
            conditional = self.state_flow(state, time, repeated_condition)
            if guidance_scale != 1.0:
                unconditional = self.state_flow(
                    state,
                    time,
                    repeated_condition,
                    condition_present=state.new_zeros(state.shape[0]),
                )
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            else:
                velocity = conditional
            if self.config.flow_geometry == "hypersphere":
                state = sphere_exponential_map(state, -delta * velocity)
            else:
                state = state - delta * velocity
        return F.normalize(
            state.reshape(batch, samples_per_context, self.config.visual_dim),
            dim=-1,
        )

    def score_candidates(
        self,
        sampled_states: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Kernel-density score from continuous samples to image-derived candidates."""

        if sampled_states.ndim != 3:
            raise ValueError("sampled states must have shape [batch, samples, visual_dim]")
        sampled = F.normalize(sampled_states.float(), dim=-1)
        candidate = F.normalize(candidates.float(), dim=-1)
        if candidate.ndim == 2:
            similarities = torch.einsum("bsd,nd->bsn", sampled, candidate)
            count = sampled.shape[1]
            return torch.logsumexp(self.sample_scale * similarities, dim=1) - math.log(count)
        if candidate.ndim == 3:
            similarities = torch.einsum("bsd,nvd->bsnv", sampled, candidate)
            count = sampled.shape[1] * candidate.shape[1]
            return torch.logsumexp(
                self.sample_scale * similarities,
                dim=(1, 3),
            ) - math.log(count)
        raise ValueError("candidates must have shape [count, dim] or [count, views, dim]")


def _multi_positive_nce_rows(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("multi-positive logits and mask must be matching matrices")
    if not positive_mask.any(dim=1).all():
        raise ValueError("every query must have at least one visual positive")
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    return torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)


def _sample_identity_inputs(
    model: PredictiveVisualField,
    outputs: dict[str, torch.Tensor],
    target_ink: torch.Tensor,
    positions: torch.Tensor,
    *,
    duplicate_similarity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = target_ink.shape[0]
    if positions.ndim != 2 or positions.shape[0] != batch:
        raise ValueError("sampled identity positions must have shape [batch, positions]")
    batch_indices = torch.arange(batch, device=target_ink.device)[:, None].expand_as(positions)
    condition = outputs["condition"][batch_indices, positions].flatten(0, 1)
    current_visual = (
        outputs["current_visual"][batch_indices, positions].detach().flatten(0, 1)
    )
    target_visual = F.normalize(
        outputs["target_visual"][batch_indices, positions].float().detach().flatten(0, 1),
        dim=-1,
    )
    target_source = F.normalize(
        model.encode_images(target_ink[batch_indices, positions].flatten(0, 1)),
        dim=-1,
    )
    candidates = torch.cat((target_visual, target_source), dim=0)
    similarity = target_visual @ candidates.transpose(0, 1)
    positive = similarity >= duplicate_similarity
    query = torch.arange(condition.shape[0], device=target_ink.device)
    positive[query, query] = True
    positive[query, condition.shape[0] + query] = True
    with torch.no_grad():
        last_state, _ = model.dynamics(current_visual[:, None])
        last_condition = torch.cat((last_state[:, 0], current_visual), dim=-1)
    return condition, last_condition, candidates, positive


def sampled_visual_anchor_losses(
    model: PredictiveVisualField,
    full_samples: torch.Tensor,
    last_samples: torch.Tensor,
    target_visual: torch.Tensor,
    anchor_candidates: torch.Tensor | None,
    *,
    positive_similarity: float,
    context_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Calibrate sampled states against independent image-only visual anchors."""

    zero = full_samples.sum() * 0.0
    disabled = {
        "visual_anchor_active": zero.detach(),
        "visual_anchor_coverage": zero.detach(),
        "visual_anchor_positive_count": zero.detach(),
        "visual_anchor_identity_nce": zero.detach(),
        "visual_anchor_full_top1": zero.detach(),
        "visual_anchor_last_top1": zero.detach(),
        "visual_anchor_context_log_probability_gain": zero.detach(),
        "visual_anchor_context_advantage_loss": zero.detach(),
        "visual_anchor_context_satisfied_fraction": zero.detach(),
    }
    if anchor_candidates is None:
        return zero, zero, disabled
    if anchor_candidates.ndim != 3:
        raise ValueError("visual anchors must have shape [objects, views, visual_dim]")
    if anchor_candidates.shape[-1] != model.config.visual_dim:
        raise ValueError("visual anchor dimension differs from the retinal state")
    if not -1.0 <= positive_similarity <= 1.0:
        raise ValueError("visual anchor positive similarity must be in [-1, 1]")
    if context_margin < 0.0:
        raise ValueError("visual anchor context margin must be non-negative")

    target = F.normalize(target_visual.float().detach(), dim=-1)
    anchors = F.normalize(anchor_candidates.float().detach(), dim=-1)
    similarity = torch.einsum("bd,nvd->bnv", target, anchors).amax(dim=-1)
    positive = similarity >= positive_similarity
    eligible = positive.any(dim=1)
    coverage = eligible.float().mean()
    if not eligible.any():
        disabled["visual_anchor_coverage"] = coverage.detach()
        return zero, zero, disabled

    full_logits = model.score_candidates(full_samples, anchors)[eligible]
    last_logits = model.score_candidates(last_samples, anchors).detach()[eligible]
    selected_positive = positive[eligible]
    full_rows = _multi_positive_nce_rows(full_logits, selected_positive)
    last_rows = _multi_positive_nce_rows(last_logits, selected_positive)
    context_gain = last_rows - full_rows
    context_advantage = F.relu(context_margin - context_gain).mean()
    full_top1 = selected_positive.gather(
        1,
        full_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    last_top1 = selected_positive.gather(
        1,
        last_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    identity = full_rows.mean()
    metrics = {
        "visual_anchor_active": identity.new_tensor(1.0).detach(),
        "visual_anchor_coverage": coverage.detach(),
        "visual_anchor_positive_count": selected_positive.float().sum(dim=1).mean().detach(),
        "visual_anchor_identity_nce": identity.detach(),
        "visual_anchor_full_top1": full_top1.detach(),
        "visual_anchor_last_top1": last_top1.detach(),
        "visual_anchor_context_log_probability_gain": context_gain.mean().detach(),
        "visual_anchor_context_advantage_loss": context_advantage.detach(),
        "visual_anchor_context_satisfied_fraction": (
            context_gain >= context_margin
        ).float().mean().detach(),
    }
    return identity, context_advantage, metrics


def predictive_visual_field_loss(
    model: PredictiveVisualField,
    outputs: dict[str, torch.Tensor],
    target_ink: torch.Tensor,
    *,
    flow_positions_per_sequence: int = 8,
    duplicate_similarity: float = 0.90,
    flow_weight: float = 1.0,
    endpoint_weight: float = 0.25,
    sampled_identity_weight: float = 0.50,
    sampled_endpoint_weight: float = 0.0,
    proposal_geodesic_weight: float = 0.0,
    proposal_identity_weight: float = 0.0,
    proposal_context_weight: float = 0.0,
    proposal_anchor_identity_weight: float = 0.0,
    proposal_anchor_context_weight: float = 0.0,
    context_advantage_weight: float = 0.50,
    context_advantage_margin: float = 0.10,
    sampled_positions_per_sequence: int = 1,
    visual_anchor_candidates: torch.Tensor | None = None,
    visual_anchor_positive_similarity: float = 0.85,
    visual_anchor_identity_weight: float = 0.0,
    visual_anchor_context_weight: float = 0.0,
    visual_anchor_context_margin: float = 0.10,
    samples_per_context: int = 2,
    sample_steps: int = 2,
    guidance_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "target_visual" not in outputs:
        raise ValueError("predictive visual field loss requires target image states")
    batch, length = target_ink.shape[:2]
    if not 1 <= flow_positions_per_sequence <= length:
        raise ValueError("flow_positions_per_sequence must be within the sequence")
    if not 1 <= sampled_positions_per_sequence <= flow_positions_per_sequence:
        raise ValueError("sampled positions must be within the selected flow positions")
    order = torch.rand(
        batch,
        length,
        device=target_ink.device,
        generator=generator,
    ).argsort(dim=1)[:, :flow_positions_per_sequence]
    batch_indices = torch.arange(batch, device=target_ink.device)
    selected_batch = batch_indices[:, None].expand_as(order)
    condition = outputs["condition"][selected_batch, order].flatten(0, 1)
    target = F.normalize(
        outputs["target_visual"][selected_batch, order].float().detach().flatten(0, 1),
        dim=-1,
    )
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    time = torch.rand(
        target.shape[0],
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    ).clamp_(0.001, 0.999)
    if model.config.flow_geometry == "hypersphere":
        noise = F.normalize(noise.float(), dim=-1)
        mixed, target_velocity = hyperspherical_flow_path(target, noise, time)
    else:
        mixed = (1.0 - time[:, None]) * target + time[:, None] * noise
        target_velocity = noise - target
    keep = (
        torch.rand(target.shape[0], device=target.device, generator=generator)
        >= model.config.condition_dropout
    ).to(target.dtype)
    predicted_velocity = model.state_flow(
        mixed,
        time,
        condition,
        condition_present=keep,
    )
    flow_error = (predicted_velocity - target_velocity).float()
    flow_mse = flow_error.pow(2).mean()
    if model.config.flow_geometry == "hypersphere":
        # The intrinsic objective is tangent-vector energy, not per-coordinate MSE.
        flow_objective = flow_error.pow(2).sum(dim=-1).mean()
    else:
        flow_objective = flow_mse
    if model.config.flow_geometry == "hypersphere":
        endpoint = sphere_exponential_map(
            mixed.float(),
            -time.float()[:, None] * predicted_velocity.float(),
        )
    else:
        endpoint = F.normalize(
            mixed.float() - time.float()[:, None] * predicted_velocity.float(),
            dim=-1,
        )
    endpoint_cosine = (endpoint * target).sum(dim=-1)
    endpoint_loss = (1.0 - endpoint_cosine).mean()

    identity_positions = order[:, :sampled_positions_per_sequence]
    full_condition, last_condition, candidates, positive = _sample_identity_inputs(
        model,
        outputs,
        target_ink,
        identity_positions,
        duplicate_similarity=duplicate_similarity,
    )
    identity_queries = full_condition.shape[0]
    sample_noise = torch.randn(
        identity_queries,
        samples_per_context,
        model.config.visual_dim,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    full_samples = model.sample_states(
        full_condition,
        samples_per_context=samples_per_context,
        steps=sample_steps,
        guidance_scale=guidance_scale,
        noise=sample_noise,
    )
    with torch.no_grad():
        last_samples = model.sample_states(
            last_condition,
            samples_per_context=samples_per_context,
            steps=sample_steps,
            guidance_scale=guidance_scale,
            noise=sample_noise,
        )
    full_logits = model.score_candidates(full_samples, candidates)
    last_logits = model.score_candidates(last_samples, candidates).detach()
    full_nce_rows = _multi_positive_nce_rows(full_logits, positive)
    last_nce_rows = _multi_positive_nce_rows(last_logits, positive)
    sampled_identity = full_nce_rows.mean()
    context_log_probability_gain = last_nce_rows - full_nce_rows
    context_advantage = F.relu(
        context_advantage_margin - context_log_probability_gain
    ).mean()
    full_retrieval = positive.gather(
        1,
        full_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    last_retrieval = positive.gather(
        1,
        last_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    target_for_identity = F.normalize(
        outputs["target_visual"][
            batch_indices[:, None].expand_as(identity_positions),
            identity_positions,
        ].float().detach().flatten(0, 1),
        dim=-1,
    )
    full_proposal = model.visual_proposal(full_condition)
    with torch.no_grad():
        last_proposal = model.visual_proposal(last_condition)
    proposal_target_cosine = (full_proposal * target_for_identity).sum(dim=-1)
    proposal_geodesic = torch.acos(
        proposal_target_cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ).pow(2).mean()
    proposal_full_logits = model.score_candidates(full_proposal[:, None], candidates)
    proposal_last_logits = model.score_candidates(last_proposal[:, None], candidates).detach()
    proposal_full_rows = _multi_positive_nce_rows(proposal_full_logits, positive)
    proposal_last_rows = _multi_positive_nce_rows(proposal_last_logits, positive)
    proposal_identity = proposal_full_rows.mean()
    proposal_context_gain = proposal_last_rows - proposal_full_rows
    proposal_context = F.relu(
        context_advantage_margin - proposal_context_gain
    ).mean()
    proposal_full_top1 = positive.gather(
        1,
        proposal_full_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    proposal_last_top1 = positive.gather(
        1,
        proposal_last_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    proposal_anchor_identity, proposal_anchor_context, proposal_anchor_raw_metrics = (
        sampled_visual_anchor_losses(
            model,
            full_proposal[:, None],
            last_proposal[:, None],
            target_for_identity,
            visual_anchor_candidates,
            positive_similarity=visual_anchor_positive_similarity,
            context_margin=visual_anchor_context_margin,
        )
    )
    proposal_anchor_metrics = {
        key.replace("visual_anchor_", "proposal_anchor_"): value
        for key, value in proposal_anchor_raw_metrics.items()
    }
    sample_target_cosines = torch.einsum(
        "bsd,bd->bs",
        full_samples,
        target_for_identity,
    )
    best_sample_target_cosine = sample_target_cosines.amax(dim=1)
    sampled_endpoint_geodesic = torch.acos(
        best_sample_target_cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ).pow(2).mean()
    sample_pair_cosine = (
        (full_samples[:, 0] * full_samples[:, -1]).sum(dim=-1).mean()
        if samples_per_context > 1
        else full_samples.new_tensor(1.0)
    )
    visual_anchor_identity, visual_anchor_context, visual_anchor_metrics = (
        sampled_visual_anchor_losses(
            model,
            full_samples,
            last_samples,
            target_for_identity,
            visual_anchor_candidates,
            positive_similarity=visual_anchor_positive_similarity,
            context_margin=visual_anchor_context_margin,
        )
    )
    total = (
        flow_weight * flow_objective
        + endpoint_weight * endpoint_loss
        + sampled_identity_weight * sampled_identity
        + sampled_endpoint_weight * sampled_endpoint_geodesic
        + proposal_geodesic_weight * proposal_geodesic
        + proposal_identity_weight * proposal_identity
        + proposal_context_weight * proposal_context
        + proposal_anchor_identity_weight * proposal_anchor_identity
        + proposal_anchor_context_weight * proposal_anchor_context
        + context_advantage_weight * context_advantage
        + visual_anchor_identity_weight * visual_anchor_identity
        + visual_anchor_context_weight * visual_anchor_context
    )
    metrics = {
        "state_flow_objective": flow_objective.detach(),
        "state_flow_mse": flow_mse.detach(),
        "state_endpoint_cosine": endpoint_cosine.mean().detach(),
        "sampled_state_identity_nce": sampled_identity.detach(),
        "sampled_state_endpoint_geodesic": sampled_endpoint_geodesic.detach(),
        "sampled_state_full_top1": full_retrieval.detach(),
        "sampled_state_last_top1": last_retrieval.detach(),
        "sampled_state_context_log_probability_gain": (
            context_log_probability_gain.mean().detach()
        ),
        "sampled_state_context_advantage_loss": context_advantage.detach(),
        "sampled_state_context_satisfied_fraction": (
            context_log_probability_gain >= context_advantage_margin
        ).float().mean().detach(),
        "sampled_state_target_cosine": best_sample_target_cosine.mean().detach(),
        "sampled_state_pair_cosine": sample_pair_cosine.detach(),
        "proposal_geodesic": proposal_geodesic.detach(),
        "proposal_target_cosine": proposal_target_cosine.mean().detach(),
        "proposal_identity_nce": proposal_identity.detach(),
        "proposal_full_top1": proposal_full_top1.detach(),
        "proposal_last_top1": proposal_last_top1.detach(),
        "proposal_context_log_probability_gain": proposal_context_gain.mean().detach(),
        "proposal_context_advantage_loss": proposal_context.detach(),
        "proposal_context_satisfied_fraction": (
            proposal_context_gain >= context_advantage_margin
        ).float().mean().detach(),
        "condition_keep_fraction": keep.mean().detach(),
        "sample_scale": model.sample_scale.detach(),
        **visual_anchor_metrics,
        **proposal_anchor_metrics,
    }
    return total, metrics


def initialize_from_retinal_flow_checkpoint(
    model: PredictiveVisualField,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint.get("architecture") != "retinal-flow-language-model-v1":
        raise ValueError("initialization checkpoint is not a retinal flow model")
    source_config = checkpoint.get("model_config", {})
    for key in (
        "fovea_size",
        "visual_dim",
        "state_dim",
        "state_layers",
        "retina_base_channels",
    ):
        if int(source_config[key]) != int(getattr(model.config, key)):
            raise ValueError(f"retinal flow initialization differs at {key}")
    source_state = checkpoint["model"]
    retina_state = {
        key.removeprefix("target_retina."): value
        for key, value in source_state.items()
        if key.startswith("target_retina.")
    }
    dynamics_state = {
        key.removeprefix("dynamics."): value
        for key, value in source_state.items()
        if key.startswith("dynamics.")
    }
    model.retina.load_state_dict(retina_state)
    model.dynamics.load_state_dict(dynamics_state)
    return {
        "source_architecture": checkpoint["architecture"],
        "source_step": int(checkpoint.get("global_step", 0)),
        "retina_parameters_loaded": len(retina_state),
        "dynamics_parameters_loaded": len(dynamics_state),
    }


def predictive_visual_field_config_payload(
    config: PredictiveVisualFieldConfig,
) -> dict[str, Any]:
    return asdict(config)


def predictive_visual_field_config_from_payload(
    payload: dict[str, Any],
) -> PredictiveVisualFieldConfig:
    return PredictiveVisualFieldConfig(**payload)

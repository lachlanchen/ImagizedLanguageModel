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
    integrate_foveal_ink,
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
    def sample_visual_candidates(
        self,
        condition: torch.Tensor,
        current_foveas: torch.Tensor,
        *,
        samples_per_context: int = 1,
        steps: int = 8,
        guidance_scale: float = 1.5,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample, reread, and energy-rank image candidates without symbolic IDs."""

        if condition.ndim != 2:
            raise ValueError("candidate condition must have shape [batch, dimension]")
        expected = (
            condition.shape[0],
            1,
            self.config.fovea_size,
            self.config.fovea_size,
        )
        if tuple(current_foveas.shape) != expected:
            raise ValueError(f"current foveas must have shape {expected}")
        if samples_per_context < 1:
            raise ValueError("samples_per_context must be positive")

        batch = condition.shape[0]
        repeated_condition = condition.repeat_interleave(samples_per_context, dim=0)
        repeated_current = current_foveas.repeat_interleave(samples_per_context, dim=0)
        sampled = sample_foveal_ink(
            self.writer,
            repeated_condition,
            repeated_current * 2.0 - 1.0,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        candidates = ((sampled + 1.0) * 0.5).clamp(0, 1).reshape(
            batch,
            samples_per_context,
            1,
            self.config.fovea_size,
            self.config.fovea_size,
        )
        candidate_visual = self.target_retina(candidates.flatten(0, 1)).float().reshape(
            batch,
            samples_per_context,
            self.config.visual_dim,
        )
        energy = self.energy(condition, candidate_visual)
        choice = energy.argmax(dim=1)
        batch_indices = torch.arange(batch, device=condition.device)
        return {
            "candidates": candidates,
            "candidate_visual": candidate_visual,
            "energy": energy,
            "choice": choice,
            "selected": candidates[batch_indices, choice],
            "selected_visual": candidate_visual[batch_indices, choice],
        }

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
        sampled = self.sample_visual_candidates(
            prediction["condition"][:, -1],
            context_foveas[:, -1],
            samples_per_context=samples_per_context,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return sampled["candidates"]

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
    rows = _multi_positive_nce_rows(logits, positive_mask)
    retrieval = positive_mask.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    return rows.mean(), retrieval


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


def _visual_positive_mask(
    candidates: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = F.normalize(candidates.float(), dim=-1)
    similarity = candidates @ candidates.transpose(0, 1)
    mask = similarity >= threshold
    mask.fill_diagonal_(True)
    return mask, similarity


def visual_context_advantage_loss(
    full_nce_rows: torch.Tensor,
    last_nce_rows: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require full visual history to improve normalized target likelihood."""

    if full_nce_rows.ndim != 1 or full_nce_rows.shape != last_nce_rows.shape:
        raise ValueError("full and last visual NCE rows must be matching vectors")
    if margin < 0.0:
        raise ValueError("visual context margin must be non-negative")
    log_probability_gain = last_nce_rows.detach() - full_nce_rows
    loss = F.relu(margin - log_probability_gain).mean()
    return loss, log_probability_gain


def visual_rollout_losses(
    model: RetinalFlowLanguageModel,
    outputs: dict[str, torch.Tensor],
    context_foveas: torch.Tensor,
    target_ink: torch.Tensor,
    candidate_bank: torch.Tensor,
    *,
    rollout_batch_size: int,
    rollout_steps: int,
    rollout_candidates: int,
    rollout_sample_steps: int,
    rollout_guidance_scale: float,
    rollout_min_prefix: int,
    duplicate_similarity: float,
    endpoint_weight: float,
    stroke_weight: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Train on the visual states induced by the model's own sampled writing.

    Candidate generation and selection are intentionally stop-gradient. The
    selected bitmap is then reread by the online retina, so the recurrent state,
    visual energy, and recovery writer learn on the same pixel feedback used at
    inference time.
    """

    batch, length = context_foveas.shape[:2]
    if rollout_batch_size < 1:
        zero = outputs["condition"].sum() * 0.0
        return (
            {"state": zero, "energy": zero, "recovery_flow": zero},
            {
                "rollout_active": zero.detach(),
                "rollout_state_cosine": zero.detach(),
                "rollout_selected_target_cosine": zero.detach(),
                "rollout_next_energy_nce": zero.detach(),
                "rollout_next_energy_top1": zero.detach(),
                "rollout_recovery_flow": zero.detach(),
                "rollout_recovery_endpoint_f1": zero.detach(),
                "rollout_ink_fraction": zero.detach(),
            },
            {},
        )
    if rollout_steps < 1 or rollout_candidates < 1 or rollout_sample_steps < 1:
        raise ValueError("rollout steps, candidates, and sample steps must be positive")
    if rollout_min_prefix < 1:
        raise ValueError("rollout_min_prefix must be positive")
    last_start = length - rollout_steps - 1
    first_start = rollout_min_prefix - 1
    if last_start < first_start:
        raise ValueError("causal sequence is too short for the requested visual rollout")

    count = min(batch, rollout_batch_size)
    subset = torch.randperm(batch, device=context_foveas.device, generator=generator)[:count]
    start = int(
        torch.randint(
            first_start,
            last_start + 1,
            (1,),
            device=context_foveas.device,
            generator=generator,
        ).item()
    )
    prefix_visual = outputs["current_visual"][subset, : start + 1]
    prefix_state, recurrent = model.dynamics(prefix_visual)
    current_state = prefix_state[:, -1]
    current_visual = prefix_visual[:, -1]
    current_fovea = context_foveas[subset, start].float()

    state_losses: list[torch.Tensor] = []
    state_cosines: list[torch.Tensor] = []
    energy_losses: list[torch.Tensor] = []
    energy_top1: list[torch.Tensor] = []
    selected_target_cosines: list[torch.Tensor] = []
    selected_ink: list[torch.Tensor] = []
    generated: list[torch.Tensor] = []

    for offset in range(rollout_steps):
        condition = torch.cat((current_state, current_visual), dim=-1)
        sampled = model.sample_visual_candidates(
            condition.detach(),
            current_fovea.detach(),
            samples_per_context=rollout_candidates,
            steps=rollout_sample_steps,
            guidance_scale=rollout_guidance_scale,
            generator=generator,
        )
        selected_fovea = sampled["selected"].detach()
        selected_visual = sampled["selected_visual"].float().detach()
        expected_visual = outputs["target_visual"][subset, start + offset].float().detach()
        selected_target_cosines.append(
            (F.normalize(selected_visual, dim=-1) * F.normalize(expected_visual, dim=-1))
            .sum(dim=-1)
            .mean()
        )
        selected_ink.append(selected_fovea.mean())
        generated.append(selected_fovea)

        current_visual = model.online_retina(selected_fovea)
        state_step, recurrent = model.dynamics(current_visual[:, None], recurrent)
        current_state = state_step[:, 0]
        current_fovea = selected_fovea

        clean_state = outputs["state"][subset, start + offset + 1].float().detach()
        normalized_rollout_state = F.normalize(
            F.layer_norm(current_state.float(), (current_state.shape[-1],)),
            dim=-1,
        )
        normalized_clean_state = F.normalize(
            F.layer_norm(clean_state, (clean_state.shape[-1],)),
            dim=-1,
        )
        state_cosine = (normalized_rollout_state * normalized_clean_state).sum(dim=-1)
        state_cosines.append(state_cosine.mean())
        state_losses.append((1.0 - state_cosine).mean())

        next_position = start + offset + 1
        next_target = outputs["target_visual"][subset, next_position].float().detach()
        with torch.no_grad():
            next_source = model.target_retina(target_ink[subset, next_position].float()).float()
        candidates = torch.cat((candidate_bank.detach(), next_target, next_source), dim=0)
        normalized_target = F.normalize(next_target, dim=-1)
        normalized_candidates = F.normalize(candidates, dim=-1)
        positive = normalized_target @ normalized_candidates.transpose(0, 1) >= duplicate_similarity
        known = torch.arange(count, device=context_foveas.device)
        positive[known, candidate_bank.shape[0] + known] = True
        positive[known, candidate_bank.shape[0] + count + known] = True
        rollout_condition = torch.cat((current_state, current_visual), dim=-1)
        logits = model.energy(rollout_condition, candidates)
        step_energy, step_top1 = _multi_positive_nce(logits, positive)
        energy_losses.append(step_energy)
        energy_top1.append(step_top1)

    recovery_position = start + rollout_steps
    recovery_target = target_ink[subset, recovery_position].float().clamp(0, 1)
    recovery_flow_target = recovery_target * 2.0 - 1.0
    recovery_time = torch.rand(
        count,
        device=context_foveas.device,
        dtype=recovery_flow_target.dtype,
        generator=generator,
    ).sqrt()
    recovery_state, recovery_velocity, recovery_time, _ = flow_training_state(
        recovery_flow_target,
        time=recovery_time,
        generator=generator,
    )
    recovery_keep = (
        torch.rand(count, device=context_foveas.device, generator=generator)
        >= model.config.condition_dropout
    ).to(recovery_flow_target.dtype)
    recovery_condition = torch.cat((current_state, current_visual), dim=-1)
    recovery_prediction = model.writer(
        recovery_state,
        recovery_time,
        recovery_condition,
        current_fovea * 2.0 - 1.0,
        condition_present=recovery_keep,
    )
    recovery_flow, recovery_metrics = foveal_flow_loss(
        recovery_prediction,
        recovery_velocity,
        recovery_state,
        recovery_flow_target,
        recovery_time,
        endpoint_weight=endpoint_weight,
        stroke_weight=stroke_weight,
    )

    rollout_state = torch.stack(state_losses).mean()
    rollout_energy = torch.stack(energy_losses).mean()
    metrics = {
        "rollout_active": rollout_state.new_tensor(1.0).detach(),
        "rollout_examples": rollout_state.new_tensor(float(count)).detach(),
        "rollout_steps": rollout_state.new_tensor(float(rollout_steps)).detach(),
        "rollout_start_position": rollout_state.new_tensor(float(start)).detach(),
        "rollout_state_cosine": torch.stack(state_cosines).mean().detach(),
        "rollout_selected_target_cosine": torch.stack(selected_target_cosines).mean().detach(),
        "rollout_next_energy_nce": rollout_energy.detach(),
        "rollout_next_energy_top1": torch.stack(energy_top1).mean().detach(),
        "rollout_recovery_flow": recovery_flow.detach(),
        "rollout_recovery_endpoint_f1": recovery_metrics["endpoint_ink_f1"],
        "rollout_ink_fraction": torch.stack(selected_ink).mean().detach(),
    }
    trace = {
        "subset": subset,
        "generated": torch.stack(generated, dim=1),
        "start": rollout_state.new_tensor(start, dtype=torch.long),
    }
    return (
        {"state": rollout_state, "energy": rollout_energy, "recovery_flow": recovery_flow},
        metrics,
        trace,
    )


def sampled_endpoint_identity_loss(
    model: RetinalFlowLanguageModel,
    condition: torch.Tensor,
    current_fovea: torch.Tensor,
    target_fovea: torch.Tensor,
    target_visual: torch.Tensor,
    target_source_visual: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    duplicate_similarity: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Align images from the deployed flow integrator with real target views."""

    if condition.ndim != 2:
        raise ValueError("sampled identity condition must have shape [batch, dimension]")
    batch = condition.shape[0]
    expected_fovea = (batch, 1, model.config.fovea_size, model.config.fovea_size)
    if tuple(current_fovea.shape) != expected_fovea:
        raise ValueError(f"sampled identity foveas must have shape {expected_fovea}")
    if tuple(target_fovea.shape) != expected_fovea:
        raise ValueError(f"sampled identity targets must have shape {expected_fovea}")
    if target_visual.shape != target_source_visual.shape:
        raise ValueError("sampled identity target views must have matching shapes")
    if target_visual.shape != (batch, model.config.visual_dim):
        raise ValueError("sampled identity target views have the wrong visual shape")
    if batch_size < 1:
        zero = condition.sum() * 0.0
        return (
            zero,
            {
                "sampled_identity_active": zero.detach(),
                "sampled_identity_examples": zero.detach(),
                "sampled_identity_steps": zero.detach(),
                "sampled_identity_nce": zero.detach(),
                "sampled_identity_top1": zero.detach(),
                "sampled_identity_target_cosine": zero.detach(),
                "sampled_identity_ink_fraction": zero.detach(),
                "sampled_identity_pixel_f1": zero.detach(),
            },
            {},
        )
    if steps < 1:
        raise ValueError("sampled identity flow steps must be positive")

    count = min(batch, batch_size)
    subset = torch.randperm(batch, device=condition.device, generator=generator)[:count]
    sampled_state = integrate_foveal_ink(
        model.writer,
        condition[subset],
        current_fovea[subset] * 2.0 - 1.0,
        steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    sampled_ink = ((sampled_state + 1.0) * 0.5).clamp(0, 1)
    sampled_visual = model.target_retina(sampled_ink).float()

    candidates = torch.cat((target_visual.detach(), target_source_visual.detach()), dim=0)
    normalized_target = F.normalize(target_visual[subset].float().detach(), dim=-1)
    normalized_candidates = F.normalize(candidates.float(), dim=-1)
    positive = normalized_target @ normalized_candidates.transpose(0, 1) >= duplicate_similarity
    rows = torch.arange(count, device=condition.device)
    positive[rows, subset] = True
    positive[rows, batch + subset] = True
    normalized_sampled = F.normalize(sampled_visual, dim=-1)
    logits = model.retina_scale.detach() * normalized_sampled @ normalized_candidates.transpose(0, 1)
    identity, retrieval = _multi_positive_nce(logits, positive)

    target_cosine = (normalized_sampled * normalized_target).sum(dim=-1).mean()
    sampled_binary = sampled_ink >= 0.5
    target_binary = target_fovea[subset] >= 0.5
    true_positive = (sampled_binary & target_binary).sum(dim=(1, 2, 3)).float()
    pixel_f1 = 2.0 * true_positive / (
        sampled_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))
    ).clamp_min(1)
    metrics = {
        "sampled_identity_active": identity.new_tensor(1.0).detach(),
        "sampled_identity_examples": identity.new_tensor(float(count)).detach(),
        "sampled_identity_steps": identity.new_tensor(float(steps)).detach(),
        "sampled_identity_nce": identity.detach(),
        "sampled_identity_top1": retrieval.detach(),
        "sampled_identity_target_cosine": target_cosine.detach(),
        "sampled_identity_ink_fraction": sampled_ink.mean().detach(),
        "sampled_identity_pixel_f1": pixel_f1.mean().detach(),
    }
    trace = {
        "subset": subset,
        "generated": sampled_ink,
        "sampled_binary": sampled_binary,
    }
    return identity, metrics, trace


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
    candidate_invariance_weight: float = 0.10,
    writer_cycle_weight: float = 0.35,
    context_advantage_weight: float = 0.25,
    context_advantage_margin: float = 0.25,
    sampled_identity_weight: float = 0.30,
    sampled_identity_batch_size: int = 8,
    sampled_identity_steps: int = 2,
    sampled_identity_guidance_scale: float = 1.5,
    context_identity_weight_scale: float = 1.0,
    rollout_batch_size: int = 0,
    rollout_steps: int = 2,
    rollout_candidates: int = 2,
    rollout_sample_steps: int = 2,
    rollout_guidance_scale: float = 1.5,
    rollout_min_prefix: int = 8,
    rollout_state_weight: float = 0.15,
    rollout_energy_weight: float = 0.35,
    rollout_recovery_flow_weight: float = 0.30,
    rollout_weight_scale: float = 1.0,
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
    energy_current_visual = outputs["current_visual"][energy_batches, random_order].flatten(0, 1)
    energy_target_visual = (
        outputs["target_visual"][energy_batches, random_order].float().detach().flatten(0, 1)
    )
    energy_target_ink = target_ink[energy_batches, random_order].flatten(0, 1)
    with torch.no_grad():
        source_target_visual = model.target_retina(energy_target_ink).float()
    energy_candidates = torch.cat((energy_target_visual, source_target_visual), dim=0)
    normalized_energy_target = F.normalize(energy_target_visual, dim=-1)
    normalized_energy_candidates = F.normalize(energy_candidates, dim=-1)
    energy_similarity = normalized_energy_target @ normalized_energy_candidates.transpose(0, 1)
    visual_positive = energy_similarity >= duplicate_similarity
    query_indices = torch.arange(energy_condition.shape[0], device=context_foveas.device)
    visual_positive[query_indices, query_indices] = True
    visual_positive[query_indices, energy_condition.shape[0] + query_indices] = True
    energy_logits = model.energy(energy_condition, energy_candidates)
    energy_rows = _multi_positive_nce_rows(energy_logits, visual_positive)
    energy = energy_rows.mean()
    energy_retrieval = visual_positive.gather(
        1,
        energy_logits.argmax(dim=1, keepdim=True),
    ).float().mean()
    with torch.no_grad():
        last_state, _ = model.dynamics(energy_current_visual.detach()[:, None])
        last_condition = torch.cat(
            (last_state[:, 0], energy_current_visual.detach()),
            dim=-1,
        )
        last_energy_logits = model.energy(last_condition, energy_candidates.detach())
        last_energy_rows = _multi_positive_nce_rows(last_energy_logits, visual_positive)
        last_energy_retrieval = visual_positive.gather(
            1,
            last_energy_logits.argmax(dim=1, keepdim=True),
        ).float().mean()
    context_advantage, context_log_probability_gain = visual_context_advantage_loss(
        energy_rows,
        last_energy_rows,
        margin=context_advantage_margin,
    )
    projected_reference = F.normalize(model.energy.candidate(energy_target_visual), dim=-1)
    projected_source = F.normalize(model.energy.candidate(source_target_visual), dim=-1)
    candidate_invariance = (
        1.0 - (projected_reference * projected_source).sum(dim=-1)
    ).mean()
    target_similarity = normalized_energy_target @ normalized_energy_target.transpose(0, 1)

    positions = random_order[:, 0]
    condition = outputs["condition"][batch_indices, positions]
    target_visual = outputs["target_visual"][batch_indices, positions].float().detach()
    current_visual = outputs["current_visual"][batch_indices, positions].float()
    current_reference = outputs["current_reference_visual"][batch_indices, positions].float()
    current_fovea = context_foveas[batch_indices, positions].float()
    target_fovea = target_ink[batch_indices, positions].float().clamp(0, 1)
    with torch.no_grad():
        target_source_visual = model.target_retina(target_fovea).float()

    flow_target = target_fovea * 2.0 - 1.0
    high_noise_time = torch.rand(
        batch,
        device=flow_target.device,
        dtype=flow_target.dtype,
        generator=generator,
    ).sqrt()
    flow_state, velocity, time_field, _ = flow_training_state(
        flow_target,
        time=high_noise_time,
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

    endpoint = flow_state.float() - time_field.float()[:, None, None, None] * predicted_velocity.float()
    endpoint_ink = ((endpoint + 1.0) * 0.5).clamp(0, 1)
    generated_visual = model.target_retina(endpoint_ink)
    normalized_generated = F.normalize(generated_visual.float(), dim=-1)
    normalized_target = F.normalize(target_visual, dim=-1)
    writer_positive, _ = _visual_positive_mask(target_visual, duplicate_similarity)
    writer_logits = model.retina_scale.detach() * (
        normalized_generated @ normalized_target.transpose(0, 1)
    )
    writer_cycle_rows = _multi_positive_nce_rows(writer_logits, writer_positive)
    cycle_weights = keep.float() * (0.05 + 0.95 * time_field.float().pow(4))
    cycle_weight_sum = cycle_weights.sum().clamp_min(1e-6)
    writer_cycle = (writer_cycle_rows * cycle_weights).sum() / cycle_weight_sum
    with torch.no_grad():
        writer_cycle_retrieval = writer_positive.gather(
            1,
            writer_logits.argmax(dim=1, keepdim=True),
        ).float()
        writer_cycle_top1 = (
            writer_cycle_retrieval[:, 0] * cycle_weights
        ).sum() / cycle_weight_sum
        writer_target_cosine = (
            (normalized_generated * normalized_target).sum(dim=-1) * cycle_weights
        ).sum() / cycle_weight_sum

    normalized_current = F.normalize(current_visual, dim=-1)
    normalized_reference = F.normalize(current_reference, dim=-1)
    invariance = (1.0 - (normalized_current * normalized_reference).sum(dim=-1)).mean()
    retina_positive, _ = _visual_positive_mask(current_reference, duplicate_similarity)
    retina_logits = model.retina_scale * normalized_current @ normalized_reference.transpose(0, 1)
    retina_contrastive, retina_retrieval = _multi_positive_nce(retina_logits, retina_positive)
    retina_centered = F.layer_norm(current_visual, (current_visual.shape[-1],))
    retina_std = torch.sqrt(retina_centered.var(dim=0, unbiased=False) + 1e-4)
    retina_variance = F.relu(0.75 - retina_std).mean()

    sampled_identity, sampled_identity_metrics, sampled_identity_trace = (
        sampled_endpoint_identity_loss(
            model,
            condition,
            current_fovea,
            target_fovea,
            target_visual,
            target_source_visual,
            batch_size=(
                sampled_identity_batch_size if context_identity_weight_scale > 0.0 else 0
            ),
            steps=sampled_identity_steps,
            guidance_scale=sampled_identity_guidance_scale,
            duplicate_similarity=duplicate_similarity,
            generator=generator,
        )
    )

    rollout_losses, rollout_metrics, rollout_trace = visual_rollout_losses(
        model,
        outputs,
        context_foveas,
        target_ink,
        energy_candidates,
        rollout_batch_size=rollout_batch_size if rollout_weight_scale > 0.0 else 0,
        rollout_steps=rollout_steps,
        rollout_candidates=rollout_candidates,
        rollout_sample_steps=rollout_sample_steps,
        rollout_guidance_scale=rollout_guidance_scale,
        rollout_min_prefix=rollout_min_prefix,
        duplicate_similarity=duplicate_similarity,
        endpoint_weight=endpoint_weight,
        stroke_weight=stroke_weight,
        generator=generator,
    )
    rollout_total = (
        rollout_state_weight * rollout_losses["state"]
        + rollout_energy_weight * rollout_losses["energy"]
        + rollout_recovery_flow_weight * rollout_losses["recovery_flow"]
    )
    context_identity_total = (
        context_advantage_weight * context_advantage
        + sampled_identity_weight * sampled_identity
    )

    total = (
        flow_weight * flow
        + energy_weight * energy
        + invariance_weight * invariance
        + retina_contrastive_weight * retina_contrastive
        + retina_variance_weight * retina_variance
        + candidate_invariance_weight * candidate_invariance
        + writer_cycle_weight * writer_cycle
        + context_identity_weight_scale * context_identity_total
        + rollout_weight_scale * rollout_total
    )
    metrics = {
        **flow_metrics,
        "visual_energy_nce": energy.detach(),
        "visual_energy_top1": energy_retrieval.detach(),
        "last_visual_energy_nce": last_energy_rows.mean().detach(),
        "last_visual_energy_top1": last_energy_retrieval.detach(),
        "context_log_probability_gain": context_log_probability_gain.mean().detach(),
        "context_advantage_margin": energy.new_tensor(float(context_advantage_margin)).detach(),
        "context_advantage_loss": context_advantage.detach(),
        "context_advantage_satisfied_fraction": (
            context_log_probability_gain >= context_advantage_margin
        ).float().mean().detach(),
        "context_identity_weight_scale": total.new_tensor(
            float(context_identity_weight_scale)
        ).detach(),
        "context_identity_total": context_identity_total.detach(),
        "visual_positive_count": visual_positive.float().sum(dim=1).mean().detach(),
        "visual_energy_queries": energy.new_tensor(float(energy_condition.shape[0])),
        "visual_energy_candidates": energy.new_tensor(float(energy_candidates.shape[0])),
        "candidate_cross_render_invariance": candidate_invariance.detach(),
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
        "writer_cycle_nce": writer_cycle.detach(),
        "writer_cycle_top1": writer_cycle_top1.detach(),
        "writer_target_cosine": writer_target_cosine.detach(),
        **sampled_identity_metrics,
        "rollout_weight_scale": total.new_tensor(float(rollout_weight_scale)).detach(),
        "rollout_total": rollout_total.detach(),
        **rollout_metrics,
    }
    selected = {
        "batch_indices": batch_indices,
        "positions": positions,
        "condition": condition,
        "current_fovea": current_fovea,
        "target_fovea": target_fovea,
        "target_visual": target_visual,
        **{f"sampled_identity_{key}": value for key, value in sampled_identity_trace.items()},
        **{f"rollout_{key}": value for key, value in rollout_trace.items()},
    }
    return total, metrics, selected


def retinal_flow_config_payload(config: RetinalFlowConfig) -> dict[str, Any]:
    return asdict(config)


def retinal_flow_config_from_payload(payload: dict[str, Any]) -> RetinalFlowConfig:
    return RetinalFlowConfig(**payload)

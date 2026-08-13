from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .conditional_visual_field_flow import (
    V31_FIELD_CELLS,
    V31_TRAIN_PROBE_TIMES,
    ConditionalVisualFieldFlowModel,
)
from .joint_visual_compatibility import paired_assignment_loss
from .spatial_visual_next_field import per_row_assignment_margin
from .spatial_visual_next_field_training import (
    shuffle_visual_prefix,
    target_log_probability,
)


V31_ORDER_MARGIN = 0.10


def coherent_base_from_vectors(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError("V31 base vectors must be [B,C]")
    normalized = F.normalize(vectors.float(), dim=-1)
    return normalized[:, None].expand(-1, V31_FIELD_CELLS, -1).clone()


def flow_matching_loss(
    model: ConditionalVisualFieldFlowModel,
    condition: torch.Tensor,
    target_fields: torch.Tensor,
    *,
    times: torch.Tensor | None = None,
    base_fields: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if condition.ndim != 2 or target_fields.ndim != 3:
        raise ValueError("V31 flow rows require [B,D] condition and [B,16,C] target")
    if condition.shape[0] != target_fields.shape[0]:
        raise ValueError("V31 flow condition and target batches do not align")
    rows = condition.shape[0]
    if times is None:
        times = torch.rand(rows, device=condition.device) * 0.998 + 0.001
    if times.shape != (rows,):
        raise ValueError("V31 training times must be [B]")
    if base_fields is None:
        base_fields = model.make_coherent_base(rows, device=condition.device)
    if base_fields.shape != target_fields.shape:
        raise ValueError("V31 base and target fields must have identical shape")
    noisy = (1.0 - times[:, None, None]) * base_fields + (
        times[:, None, None] * target_fields
    )
    target_velocity = target_fields - base_fields
    predicted = model.velocity(condition, noisy, times)
    residual = predicted.float() - target_velocity.float()
    per_row = residual.square().sum(dim=-1).mean(dim=-1)
    loss = per_row.mean()
    endpoint = F.normalize(
        noisy.float() + (1.0 - times[:, None, None]) * predicted.float(),
        dim=-1,
    )
    cosine = (endpoint * target_fields.float()).sum(dim=-1).mean()
    return loss, {
        "flow_loss": loss.detach(),
        "flow_endpoint_cosine": cosine.detach(),
        "flow_velocity_norm": predicted.float().norm(dim=-1).mean().detach(),
        "flow_target_velocity_norm": (
            target_velocity.float().norm(dim=-1).mean().detach()
        ),
    }


def make_training_probes(
    model: ConditionalVisualFieldFlowModel,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.tensor(V31_TRAIN_PROBE_TIMES, device=device, dtype=torch.float32)
    bases = model.make_coherent_base(len(V31_TRAIN_PROBE_TIMES), device=device)
    return bases, times


def natural_direction_objective(
    model: ConditionalVisualFieldFlowModel,
    context: torch.Tensor,
    candidate_fields: torch.Tensor,
    probe_bases: torch.Tensor,
    probe_times: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if context.ndim != 5 or candidate_fields.shape[0] != context.shape[0]:
        raise ValueError("V31 natural context and in-batch candidates do not align")
    targets = torch.arange(context.shape[0], device=context.device)
    full_condition = model.context_condition(context)
    shuffled_condition = model.context_condition(shuffle_visual_prefix(context))
    flow_loss, flow_metrics = flow_matching_loss(
        model, full_condition, candidate_fields
    )
    full = model.path_score_encoded_shared(
        full_condition, candidate_fields, probe_bases, probe_times
    )
    shuffled = model.path_score_encoded_shared(
        shuffled_condition, candidate_fields, probe_bases, probe_times
    )
    classification = F.cross_entropy(full.float(), targets)
    full_logp = target_log_probability(full, targets)
    shuffled_logp = target_log_probability(shuffled, targets)
    order = F.softplus(V31_ORDER_MARGIN - (full_logp - shuffled_logp)).mean()
    return {
        "flow": flow_loss,
        "classification": classification,
        "order": order,
    }, {
        "natural_flow_loss": flow_metrics["flow_loss"],
        "natural_flow_endpoint_cosine": flow_metrics["flow_endpoint_cosine"],
        "natural_velocity_norm": flow_metrics["flow_velocity_norm"],
        "natural_target_velocity_norm": flow_metrics["flow_target_velocity_norm"],
        "natural_classification_loss": classification.detach(),
        "natural_order_loss": order.detach(),
        "natural_full_top1": (full.argmax(dim=-1) == targets).float().mean(),
        "natural_shuffled_top1": (shuffled.argmax(dim=-1) == targets).float().mean(),
        "natural_full_target_log_probability": full_logp.mean().detach(),
        "natural_shuffled_target_log_probability": shuffled_logp.mean().detach(),
        "natural_order_gain": (full_logp - shuffled_logp).mean().detach(),
    }


def natural_objective(
    model: ConditionalVisualFieldFlowModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    required = {
        "first_context",
        "second_context",
        "first_target",
        "second_target",
    }
    if set(batch) != required:
        raise ValueError(f"V31 natural batch must contain exactly {sorted(required)}")
    probe_bases, probe_times = make_training_probes(
        model, device=batch["first_context"].device
    )
    directions = []
    for context_key, target_key in (
        ("first_context", "second_target"),
        ("second_context", "first_target"),
    ):
        candidate_fields = model.encode_route_candidates(batch[target_key])
        directions.append(
            natural_direction_objective(
                model,
                batch[context_key],
                candidate_fields,
                probe_bases,
                probe_times,
            )
        )
    losses = {
        name: torch.stack([direction[0][name] for direction in directions]).mean()
        for name in directions[0][0]
    }
    metrics = {
        name: torch.stack([direction[1][name] for direction in directions]).mean()
        for name in directions[0][1]
    }
    return losses, metrics


def gather_assigned_fields(
    candidate_fields: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    if candidate_fields.ndim != 4 or assignments.ndim != 2:
        raise ValueError("V31 assigned fields require [B,K,16,C] and [B,Q]")
    if candidate_fields.shape[0] != assignments.shape[0]:
        raise ValueError("V31 assigned field batches do not align")
    if assignments.dtype != torch.long:
        raise TypeError("V31 pair assignments must be int64")
    batch = torch.arange(candidate_fields.shape[0], device=candidate_fields.device)
    return candidate_fields[batch[:, None], assignments]


def pair_direction_objective(
    model: ConditionalVisualFieldFlowModel,
    contexts: torch.Tensor,
    candidates: torch.Tensor,
    assignments: torch.Tensor,
    probe_bases: torch.Tensor,
    probe_times: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if contexts.ndim != 6 or tuple(contexts.shape[:2]) != tuple(assignments.shape):
        raise ValueError("V31 pair contexts and assignments do not align")
    candidate_fields = model.encode_route_candidates(candidates)
    batch, queries = contexts.shape[:2]
    condition = model.context_condition(
        contexts.reshape(batch * queries, *contexts.shape[2:])
    ).reshape(batch, queries, model.config.model_dim)
    shuffled = shuffle_visual_prefix(contexts)
    shuffled_condition = model.context_condition(
        shuffled.reshape(batch * queries, *shuffled.shape[2:])
    ).reshape(batch, queries, model.config.model_dim)
    assigned = gather_assigned_fields(candidate_fields, assignments)
    flow_loss, flow_metrics = flow_matching_loss(
        model,
        condition.reshape(batch * queries, model.config.model_dim),
        assigned.reshape(
            batch * queries,
            V31_FIELD_CELLS,
            model.config.field_channels,
        ),
    )
    full = model.path_score_encoded_batched(
        condition, candidate_fields, probe_bases, probe_times
    )
    shuffled_score = model.path_score_encoded_batched(
        shuffled_condition, candidate_fields, probe_bases, probe_times
    )
    suffix = model.path_score_exact_suffix_paired(
        contexts, candidates, probe_bases, probe_times
    )
    assignment_loss, assignment_metrics = paired_assignment_loss(full, assignments)
    margin = per_row_assignment_margin(full, assignments)
    shuffled_margin = per_row_assignment_margin(shuffled_score, assignments)
    positive = F.softplus(V31_ORDER_MARGIN - margin).mean()
    order = F.softplus(V31_ORDER_MARGIN - (margin - shuffled_margin)).mean()
    return {
        "flow": flow_loss,
        "assignment": assignment_loss,
        "positive": positive,
        "order": order,
    }, {
        "pair_flow_loss": flow_metrics["flow_loss"],
        "pair_flow_endpoint_cosine": flow_metrics["flow_endpoint_cosine"],
        "pair_velocity_norm": flow_metrics["flow_velocity_norm"],
        "pair_target_velocity_norm": flow_metrics["flow_target_velocity_norm"],
        "pair_assignment_loss": assignment_loss.detach(),
        "pair_positive_loss": positive.detach(),
        "pair_order_loss": order.detach(),
        "pair_full_arm_accuracy": assignment_metrics["pair_arm_accuracy"],
        "pair_full_both_correct_rate": assignment_metrics["pair_both_correct_rate"],
        "pair_full_mean_margin": assignment_metrics["pair_mean_margin"],
        "pair_shuffled_mean_margin": shuffled_margin.mean().detach(),
        "pair_order_gain": (margin - shuffled_margin).mean().detach(),
        "pair_suffix_row_max_error": (suffix[:, 0] - suffix[:, 1])
        .abs()
        .amax()
        .detach(),
    }


def pair_objective(
    model: ConditionalVisualFieldFlowModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    probe_bases, probe_times = make_training_probes(
        model, device=batch["contexts"].device
    )
    directions = []
    for context_key, candidate_key, assignment_key in (
        ("contexts", "candidates", "assignment"),
        (
            "reference_contexts",
            "reference_candidates",
            "reference_assignment",
        ),
    ):
        directions.append(
            pair_direction_objective(
                model,
                batch[context_key],
                batch[candidate_key],
                batch[assignment_key],
                probe_bases,
                probe_times,
            )
        )
    losses = {
        name: torch.stack([direction[0][name] for direction in directions]).mean()
        for name in directions[0][0]
    }
    metrics = {
        name: torch.stack([direction[1][name] for direction in directions]).mean()
        for name in directions[0][1]
    }
    return losses, metrics


def conditional_visual_field_flow_training_microstep(
    model: ConditionalVisualFieldFlowModel,
    natural_batch: Mapping[str, torch.Tensor],
    pair_batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    natural_losses, natural_metrics = natural_objective(model, natural_batch)
    pair_losses, pair_metrics = pair_objective(model, pair_batch)
    total = (
        natural_losses["flow"]
        + natural_losses["classification"]
        + 0.50 * natural_losses["order"]
        + pair_losses["flow"]
        + 4.0 * pair_losses["assignment"]
        + pair_losses["positive"]
        + pair_losses["order"]
    )
    return total, {
        "loss": total.detach(),
        **natural_metrics,
        **pair_metrics,
    }


__all__ = [
    "V31_ORDER_MARGIN",
    "coherent_base_from_vectors",
    "conditional_visual_field_flow_training_microstep",
    "flow_matching_loss",
    "gather_assigned_fields",
    "make_training_probes",
    "natural_direction_objective",
    "natural_objective",
    "pair_direction_objective",
    "pair_objective",
]

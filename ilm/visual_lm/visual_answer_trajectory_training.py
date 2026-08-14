from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_answer_trajectory import (
    V39_MAX_SEGMENTS,
    VisualAnswerEncoding,
    VisualAnswerTrajectoryModel,
    VisualAnswerTrajectoryOutput,
)
from .visual_answer_trajectory_data import V39_TARGET_ARCHITECTURE
from .visual_semantic_distillation_data import V37_PATCHES


@dataclass
class VisualAnswerTrajectoryTargets:
    prompt: torch.Tensor
    answer: torch.Tensor
    segments: torch.Tensor
    segment_mask: torch.Tensor
    segment_lengths: torch.Tensor
    stop_targets: torch.Tensor
    stop_mask: torch.Tensor
    sampled_segments: torch.Tensor
    bank_indices: torch.Tensor
    segment_bank_indices: torch.Tensor
    sampled_segment_bank_indices: torch.Tensor

    def validate(self) -> None:
        batch, dimension = self.prompt.shape
        expected_segments = (batch, V39_MAX_SEGMENTS, dimension)
        expected_fields = (batch, V39_MAX_SEGMENTS)
        if self.answer.shape != self.prompt.shape:
            raise ValueError("V39 global targets do not align")
        if self.segments.shape != expected_segments:
            raise ValueError("V39 segment targets have an invalid shape")
        if any(
            value.shape != expected_fields
            for value in (
                self.segment_mask,
                self.segment_lengths,
                self.stop_targets,
                self.stop_mask,
                self.segment_bank_indices,
            )
        ):
            raise ValueError("V39 segment target fields do not align")
        if self.sampled_segments.shape != (batch, dimension):
            raise ValueError("V39 sampled segment targets do not align")
        if self.bank_indices.shape != (batch,):
            raise ValueError("V39 target record indices do not align")
        if self.sampled_segment_bank_indices.shape != (batch,):
            raise ValueError("V39 sampled segment bank indices do not align")
        if any(
            value.dtype != torch.long
            for value in (
                self.bank_indices,
                self.segment_bank_indices,
                self.sampled_segment_bank_indices,
            )
        ):
            raise TypeError("V39 bank indices must be long")
        if not all(
            torch.is_floating_point(value)
            for value in (
                self.prompt,
                self.answer,
                self.segments,
                self.segment_mask,
                self.segment_lengths,
                self.stop_targets,
                self.stop_mask,
                self.sampled_segments,
            )
        ):
            raise TypeError("V39 targets must be floating except bank indices")
        active = self.segment_mask > 0
        if not bool(active.any(dim=1).all()):
            raise ValueError("V39 every record requires an active segment")
        if bool((self.segment_bank_indices[active] < 0).any()):
            raise ValueError("V39 active segment bank indices must be nonnegative")
        if bool((self.segment_bank_indices[~active] != -1).any()):
            raise ValueError("V39 inactive segment bank indices must be -1")
        if not bool(torch.isfinite(self.prompt).all()):
            raise ValueError("V39 prompt targets must be finite")
        for value in (
            self.answer,
            self.segments,
            self.segment_lengths,
            self.stop_targets,
            self.stop_mask,
            self.sampled_segments,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("V39 target fields must be finite")


@dataclass
class VisualAnswerTrajectoryGlobalCandidates:
    prompt: torch.Tensor
    answer: torch.Tensor
    positive_labels: torch.Tensor
    bank_indices: torch.Tensor

    def validate(self, *, batch: int, dimension: int) -> None:
        count = self.prompt.shape[0]
        if self.prompt.shape != (count, dimension) or self.answer.shape != (
            count,
            dimension,
        ):
            raise ValueError("V39 global candidate matrices have invalid shapes")
        if self.positive_labels.shape != (batch,):
            raise ValueError("V39 global positive labels do not align")
        if self.bank_indices.shape != (count,):
            raise ValueError("V39 global candidate indices do not align")
        if self.positive_labels.dtype != torch.long or self.bank_indices.dtype != torch.long:
            raise TypeError("V39 global candidate indices must be long")
        if not bool(
            ((self.positive_labels >= 0) & (self.positive_labels < count)).all()
        ):
            raise ValueError("V39 global positive label is outside the candidates")


@dataclass
class VisualAnswerTrajectorySegmentCandidates:
    states: torch.Tensor
    positive_labels: torch.Tensor
    sampled_labels: torch.Tensor
    bank_indices: torch.Tensor

    def validate(
        self,
        *,
        active_segments: int,
        batch: int,
        dimension: int,
    ) -> None:
        count = self.states.shape[0]
        if self.states.shape != (count, dimension):
            raise ValueError("V39 segment candidates have an invalid shape")
        if self.positive_labels.shape != (active_segments,):
            raise ValueError("V39 segment positive labels do not align")
        if self.sampled_labels.shape != (batch,):
            raise ValueError("V39 sampled segment labels do not align")
        if self.bank_indices.shape != (count,):
            raise ValueError("V39 segment candidate bank indices do not align")
        if any(
            value.dtype != torch.long
            for value in (
                self.positive_labels,
                self.sampled_labels,
                self.bank_indices,
            )
        ):
            raise TypeError("V39 segment candidate indices must be long")
        for labels in (self.positive_labels, self.sampled_labels):
            if not bool(((labels >= 0) & (labels < count)).all()):
                raise ValueError("V39 segment label is outside the candidates")


def _deterministic_candidate_indices(
    positives: Sequence[int],
    *,
    population: int,
    count: int,
    seed: int,
) -> tuple[list[int], torch.Tensor]:
    if population < 2 or not positives:
        raise ValueError("V39 candidate population is invalid")
    if min(positives) < 0 or max(positives) >= population:
        raise ValueError("V39 candidate positive lies outside the population")
    unique = list(dict.fromkeys(int(value) for value in positives))
    if len(unique) > count or count > population:
        raise ValueError("V39 candidate count cannot contain every positive")
    selected = list(unique)
    selected_set = set(selected)
    rng = random.Random(int(seed))
    while len(selected) < count:
        candidate = rng.randrange(population)
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
    positions = {value: index for index, value in enumerate(selected)}
    labels = torch.tensor([positions[int(value)] for value in positives], dtype=torch.long)
    return selected, labels


@dataclass
class VisualAnswerTrajectoryTargetBank:
    identifiers: tuple[str, ...]
    prompt_targets: torch.Tensor
    answer_targets: torch.Tensor
    segment_targets: torch.Tensor
    segment_offsets: torch.Tensor
    segment_lengths: torch.Tensor
    teacher_mean: torch.Tensor
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.identifiers)
        if count < 2 or len(set(self.identifiers)) != count:
            raise ValueError("V39 target identifiers must be unique")
        if self.prompt_targets.ndim != 2 or self.prompt_targets.shape[0] != count:
            raise ValueError("V39 prompt target bank has an invalid shape")
        if self.answer_targets.shape != self.prompt_targets.shape:
            raise ValueError("V39 answer target bank does not align")
        dimension = self.prompt_targets.shape[1]
        if dimension < 8 or self.teacher_mean.shape != (dimension,):
            raise ValueError("V39 target dimension is invalid")
        if self.segment_targets.ndim != 2 or self.segment_targets.shape[1] != dimension:
            raise ValueError("V39 flat segment target bank has an invalid shape")
        segments = self.segment_targets.shape[0]
        if self.segment_offsets.shape != (count + 1,):
            raise ValueError("V39 segment offsets do not align")
        if self.segment_offsets.dtype != torch.long:
            raise TypeError("V39 segment offsets must be long")
        if int(self.segment_offsets[0]) != 0 or int(self.segment_offsets[-1]) != segments:
            raise ValueError("V39 segment offsets do not cover the flat bank")
        counts = self.segment_offsets[1:] - self.segment_offsets[:-1]
        if not bool(((counts >= 1) & (counts <= V39_MAX_SEGMENTS)).all()):
            raise ValueError("V39 record segment counts must be in [1,16]")
        if not bool((self.segment_offsets[1:] >= self.segment_offsets[:-1]).all()):
            raise ValueError("V39 segment offsets must be monotone")
        if self.segment_lengths.shape != (segments,):
            raise ValueError("V39 segment lengths do not align")
        for name, value in (
            ("prompt", self.prompt_targets),
            ("answer", self.answer_targets),
            ("segment", self.segment_targets),
            ("length", self.segment_lengths),
            ("teacher_mean", self.teacher_mean),
        ):
            if not torch.is_floating_point(value) or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"V39 {name} bank must be finite floating data")
        for name, value in (
            ("prompt", self.prompt_targets),
            ("answer", self.answer_targets),
            ("segment", self.segment_targets),
        ):
            norms = value.float().norm(dim=-1)
            if not bool(((norms - 1).abs() < 0.02).all()):
                raise ValueError(f"V39 {name} targets are not normalized")
        if not bool(
            ((self.segment_lengths.float() >= 1) & (self.segment_lengths.float() <= V37_PATCHES)).all()
        ):
            raise ValueError("V39 segment lengths lie outside the visual strip")
        self._index = {
            identifier: index for index, identifier in enumerate(self.identifiers)
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> VisualAnswerTrajectoryTargetBank:
        if state.get("architecture") != V39_TARGET_ARCHITECTURE:
            raise ValueError("V39 requires a V39 detached target bank")
        return cls(
            identifiers=tuple(str(value) for value in state["identifiers"]),
            prompt_targets=state["prompt_targets"],
            answer_targets=state["answer_targets"],
            segment_targets=state["segment_targets"],
            segment_offsets=state["segment_offsets"],
            segment_lengths=state["segment_lengths"],
            teacher_mean=state["teacher_mean"],
            receipt=dict(state["receipt"]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": V39_TARGET_ARCHITECTURE,
            "identifiers": self.identifiers,
            "prompt_targets": self.prompt_targets,
            "answer_targets": self.answer_targets,
            "segment_targets": self.segment_targets,
            "segment_offsets": self.segment_offsets,
            "segment_lengths": self.segment_lengths,
            "teacher_mean": self.teacher_mean,
            "receipt": dict(self.receipt),
        }

    def lookup(
        self,
        identifiers: Sequence[str],
        segment_indices: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualAnswerTrajectoryTargets:
        if segment_indices.shape != (len(identifiers),) or segment_indices.dtype != torch.long:
            raise ValueError("V39 sampled segment indices must be long [B]")
        try:
            record_indices = torch.tensor(
                [self._index[identifier] for identifier in identifiers],
                dtype=torch.long,
            )
        except KeyError as error:
            raise KeyError(f"V39 target bank lacks {error.args[0]!r}") from error
        batch = len(identifiers)
        dimension = self.prompt_targets.shape[1]
        flat_indices = torch.full((batch, V39_MAX_SEGMENTS), -1, dtype=torch.long)
        for row, record_index in enumerate(record_indices.tolist()):
            start = int(self.segment_offsets[record_index])
            stop = int(self.segment_offsets[record_index + 1])
            count = stop - start
            sampled = int(segment_indices[row])
            if not 0 <= sampled < count:
                raise ValueError("V39 sampled segment index is outside its record")
            flat_indices[row, :count] = torch.arange(start, stop)
        active = flat_indices >= 0
        gathered = self.segment_targets[flat_indices.clamp_min(0)]
        lengths = self.segment_lengths[flat_indices.clamp_min(0)]
        gathered = gathered.masked_fill(~active.unsqueeze(-1), 0)
        lengths = lengths.masked_fill(~active, 0)
        stop_targets = torch.zeros(batch, V39_MAX_SEGMENTS)
        stop_mask = active.float()
        counts = active.sum(dim=1)
        stop_targets[torch.arange(batch), counts - 1] = 1.0
        sampled_flat = flat_indices[
            torch.arange(batch),
            segment_indices.cpu(),
        ]
        result = VisualAnswerTrajectoryTargets(
            prompt=self.prompt_targets[record_indices].to(device=device, dtype=dtype),
            answer=self.answer_targets[record_indices].to(device=device, dtype=dtype),
            segments=gathered.to(device=device, dtype=dtype),
            segment_mask=active.to(device=device, dtype=dtype),
            segment_lengths=lengths.to(device=device, dtype=torch.float32),
            stop_targets=stop_targets.to(device=device, dtype=torch.float32),
            stop_mask=stop_mask.to(device=device, dtype=torch.float32),
            sampled_segments=self.segment_targets[sampled_flat].to(
                device=device,
                dtype=dtype,
            ),
            bank_indices=record_indices.to(device=device),
            segment_bank_indices=flat_indices.to(device=device),
            sampled_segment_bank_indices=sampled_flat.to(device=device),
        )
        result.validate()
        if result.segments.shape != (batch, V39_MAX_SEGMENTS, dimension):
            raise RuntimeError("V39 target lookup changed the segment geometry")
        return result

    def global_candidate_set(
        self,
        positive_indices: torch.Tensor,
        *,
        count: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualAnswerTrajectoryGlobalCandidates:
        positives = [int(value) for value in positive_indices.detach().cpu()]
        selected, labels = _deterministic_candidate_indices(
            positives,
            population=len(self.identifiers),
            count=count,
            seed=seed,
        )
        indices = torch.tensor(selected, dtype=torch.long)
        result = VisualAnswerTrajectoryGlobalCandidates(
            prompt=self.prompt_targets[indices].to(device=device, dtype=dtype),
            answer=self.answer_targets[indices].to(device=device, dtype=dtype),
            positive_labels=labels.to(device=device),
            bank_indices=indices.to(device=device),
        )
        result.validate(batch=len(positives), dimension=self.prompt_targets.shape[1])
        return result

    def segment_candidate_set(
        self,
        targets: VisualAnswerTrajectoryTargets,
        *,
        count: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualAnswerTrajectorySegmentCandidates:
        targets.validate()
        active = targets.segment_mask.bool()
        positives = [
            int(value)
            for value in targets.segment_bank_indices[active].detach().cpu()
        ]
        sampled = [
            int(value)
            for value in targets.sampled_segment_bank_indices.detach().cpu()
        ]
        required = positives + sampled
        selected, all_labels = _deterministic_candidate_indices(
            required,
            population=self.segment_targets.shape[0],
            count=count,
            seed=seed,
        )
        indices = torch.tensor(selected, dtype=torch.long)
        positive_count = len(positives)
        result = VisualAnswerTrajectorySegmentCandidates(
            states=self.segment_targets[indices].to(device=device, dtype=dtype),
            positive_labels=all_labels[:positive_count].to(device=device),
            sampled_labels=all_labels[positive_count:].to(device=device),
            bank_indices=indices.to(device=device),
        )
        result.validate(
            active_segments=positive_count,
            batch=targets.prompt.shape[0],
            dimension=self.segment_targets.shape[1],
        )
        return result


@dataclass(frozen=True)
class VisualAnswerTrajectoryLossWeights:
    temperature: float = 0.07
    prompt_nce: float = 1.0
    prompt_alignment: float = 0.5
    segment_read_nce: float = 1.0
    segment_read_alignment: float = 0.5
    answer_nce: float = 2.0
    answer_alignment: float = 1.0
    stage1_answer: float = 0.25
    span_nce: float = 2.0
    span_alignment: float = 1.0
    stage1_span: float = 0.25
    path: float = 0.5
    order: float = 0.5
    transition: float = 0.5
    relation: float = 0.1
    stop: float = 0.2
    length: float = 0.02
    variance: float = 0.1
    covariance: float = 0.01
    order_margin: float = 0.10

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if self.temperature <= 0 or self.order_margin <= 0:
            raise ValueError("V39 loss temperature and margin must be positive")
        if any(value < 0 for value in values):
            raise ValueError("V39 loss weights cannot be negative")


V39_LOSS_WEIGHTS = VisualAnswerTrajectoryLossWeights()


@dataclass
class VisualAnswerTrajectoryLoss:
    loss: torch.Tensor
    prompt_nce: torch.Tensor
    prompt_alignment: torch.Tensor
    segment_read_nce: torch.Tensor
    segment_read_alignment: torch.Tensor
    answer_nce: torch.Tensor
    answer_alignment: torch.Tensor
    stage1_answer: torch.Tensor
    span_nce: torch.Tensor
    span_alignment: torch.Tensor
    stage1_span: torch.Tensor
    path: torch.Tensor
    order: torch.Tensor
    transition: torch.Tensor
    relation: torch.Tensor
    stop: torch.Tensor
    length: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    prompt_batch_top1: torch.Tensor
    segment_read_batch_top1: torch.Tensor
    answer_batch_top1: torch.Tensor
    span_batch_top1: torch.Tensor
    prompt_cosine: torch.Tensor
    segment_read_cosine: torch.Tensor
    answer_cosine: torch.Tensor
    span_cosine: torch.Tensor
    predicted_segment_count: torch.Tensor


def _route_objective(
    states: Sequence[torch.Tensor],
    target: torch.Tensor,
    candidates: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    alignments: list[torch.Tensor] = []
    top1: list[torch.Tensor] = []
    cosines: list[torch.Tensor] = []
    for state in states:
        if state.shape != target.shape:
            raise ValueError("V39 route states and targets do not align")
        logits = state.float() @ candidates.float().T / temperature
        losses.append(F.cross_entropy(logits, labels))
        cosine = F.cosine_similarity(state.float(), target.float(), dim=-1)
        alignments.append((1 - cosine).mean())
        top1.append((logits.argmax(dim=1) == labels).float().mean())
        cosines.append(cosine.mean())
    return (
        torch.stack(losses).mean(),
        torch.stack(alignments).mean(),
        torch.stack(top1).mean(),
        torch.stack(cosines).mean(),
    )


def _path_loss(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return torch.stack(
        [
            (1 - F.cosine_similarity(left.float(), right.float(), dim=-1)).mean()
            for left, right in pairs
        ]
    ).mean()


def _ordered_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order_losses: list[torch.Tensor] = []
    transition_losses: list[torch.Tensor] = []
    relation_losses: list[torch.Tensor] = []
    for row in range(predicted.shape[0]):
        count = int(mask[row].sum())
        if count < 2:
            continue
        student = predicted[row, :count].float()
        teacher = target[row, :count].float()
        positive = F.cosine_similarity(student, teacher, dim=-1)
        adjacent = F.cosine_similarity(student, teacher.roll(shifts=-1, dims=0), dim=-1)
        order_losses.append(F.relu(margin - positive + adjacent).mean())
        student_delta = F.normalize(student[1:] - student[:-1], dim=-1)
        teacher_delta = F.normalize(teacher[1:] - teacher[:-1], dim=-1)
        transition_losses.append(
            (1 - F.cosine_similarity(student_delta, teacher_delta, dim=-1)).mean()
        )
        relation_losses.append(
            F.mse_loss(student @ student.T, teacher @ teacher.T)
        )
    zero = predicted.float().sum() * 0
    def mean_or_zero(values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tuple(values)).mean() if values else zero

    return (
        mean_or_zero(order_losses),
        mean_or_zero(transition_losses),
        mean_or_zero(relation_losses),
    )


def variance_covariance_loss(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("V39 variance control requires [N,D] with N >= 2")
    scaled = states.float() * math.sqrt(states.shape[1])
    centered = scaled - scaled.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1 - standard_deviation).mean()
    covariance_matrix = centered.T @ centered / max(1, states.shape[0] - 1)
    covariance = (
        covariance_matrix.square().sum()
        - covariance_matrix.diagonal().square().sum()
    ) / states.shape[1]
    return variance, covariance


def visual_answer_trajectory_loss(
    prompt_anchor: VisualAnswerTrajectoryOutput,
    prompt_view: VisualAnswerTrajectoryOutput,
    segment_anchor: VisualAnswerEncoding,
    segment_view: VisualAnswerEncoding,
    targets: VisualAnswerTrajectoryTargets,
    global_candidates: VisualAnswerTrajectoryGlobalCandidates,
    segment_candidates: VisualAnswerTrajectorySegmentCandidates,
    *,
    weights: VisualAnswerTrajectoryLossWeights = V39_LOSS_WEIGHTS,
) -> VisualAnswerTrajectoryLoss:
    targets.validate()
    batch, dimension = targets.prompt.shape
    active = targets.segment_mask.bool()
    active_count = int(active.sum())
    global_candidates.validate(batch=batch, dimension=dimension)
    segment_candidates.validate(
        active_segments=active_count,
        batch=batch,
        dimension=dimension,
    )
    for output in (prompt_anchor, prompt_view):
        if output.answer_state.shape != (batch, dimension):
            raise ValueError("V39 global output and target dimensions do not align")
        if output.segment_states.shape != (batch, V39_MAX_SEGMENTS, dimension):
            raise ValueError("V39 trajectory output and targets do not align")
    if segment_anchor.read_state.shape != (batch, dimension):
        raise ValueError("V39 answer read state has an invalid shape")

    prompt_nce, prompt_alignment, prompt_top1, prompt_cosine = _route_objective(
        (prompt_anchor.read_state, prompt_view.read_state),
        targets.prompt,
        global_candidates.prompt,
        global_candidates.positive_labels,
        temperature=weights.temperature,
    )
    (
        segment_read_nce,
        segment_read_alignment,
        segment_read_top1,
        segment_read_cosine,
    ) = _route_objective(
        (segment_anchor.read_state, segment_view.read_state),
        targets.sampled_segments,
        segment_candidates.states,
        segment_candidates.sampled_labels,
        temperature=weights.temperature,
    )
    answer_nce, answer_alignment, answer_top1, answer_cosine = _route_objective(
        (prompt_anchor.answer_state, prompt_view.answer_state),
        targets.answer,
        global_candidates.answer,
        global_candidates.positive_labels,
        temperature=weights.temperature,
    )
    stage1_answer, _, _, _ = _route_objective(
        (prompt_anchor.stage1_answer_state, prompt_view.stage1_answer_state),
        targets.answer,
        global_candidates.answer,
        global_candidates.positive_labels,
        temperature=weights.temperature,
    )

    target_segments = targets.segments[active]
    final_segments = (
        prompt_anchor.segment_states[active],
        prompt_view.segment_states[active],
    )
    span_nce, span_alignment, span_top1, span_cosine = _route_objective(
        final_segments,
        target_segments,
        segment_candidates.states,
        segment_candidates.positive_labels,
        temperature=weights.temperature,
    )
    stage1_span, _, _, _ = _route_objective(
        (
            prompt_anchor.stage1_segment_states[active],
            prompt_view.stage1_segment_states[active],
        ),
        target_segments,
        segment_candidates.states,
        segment_candidates.positive_labels,
        temperature=weights.temperature,
    )
    path = _path_loss(
        (
            (prompt_anchor.read_state, prompt_view.read_state),
            (prompt_anchor.answer_state, prompt_view.answer_state),
            (prompt_anchor.segment_states[active], prompt_view.segment_states[active]),
            (segment_anchor.read_state, segment_view.read_state),
        )
    )
    order_anchor, transition_anchor, relation_anchor = _ordered_losses(
        prompt_anchor.segment_states,
        targets.segments,
        targets.segment_mask,
        margin=weights.order_margin,
    )
    order_view, transition_view, relation_view = _ordered_losses(
        prompt_view.segment_states,
        targets.segments,
        targets.segment_mask,
        margin=weights.order_margin,
    )
    order = (order_anchor + order_view) / 2
    transition = (transition_anchor + transition_view) / 2
    relation = (relation_anchor + relation_view) / 2

    stop_losses: list[torch.Tensor] = []
    length_losses: list[torch.Tensor] = []
    for output in (prompt_anchor, prompt_view):
        stop_bce = F.binary_cross_entropy_with_logits(
            output.stop_logits,
            targets.stop_targets,
            reduction="none",
        )
        stop_bce = (stop_bce * targets.stop_mask).sum() / targets.stop_mask.sum()
        active_bce = F.binary_cross_entropy(
            output.active_probabilities.clamp(1e-5, 1 - 1e-5),
            targets.segment_mask,
        )
        stop_losses.append(stop_bce + 0.25 * active_bce)
        length_value = F.smooth_l1_loss(
            output.lengths,
            targets.segment_lengths,
            reduction="none",
        )
        length_losses.append(
            (length_value * targets.segment_mask).sum() / targets.segment_mask.sum()
        )
    stop = torch.stack(stop_losses).mean()
    length = torch.stack(length_losses).mean()
    variance, covariance = variance_covariance_loss(
        torch.cat(
            (
                prompt_anchor.read_state,
                prompt_view.read_state,
                prompt_anchor.answer_state,
                prompt_view.answer_state,
                segment_anchor.read_state,
                segment_view.read_state,
                *final_segments,
            ),
            dim=0,
        )
    )
    loss = (
        weights.prompt_nce * prompt_nce
        + weights.prompt_alignment * prompt_alignment
        + weights.segment_read_nce * segment_read_nce
        + weights.segment_read_alignment * segment_read_alignment
        + weights.answer_nce * answer_nce
        + weights.answer_alignment * answer_alignment
        + weights.stage1_answer * stage1_answer
        + weights.span_nce * span_nce
        + weights.span_alignment * span_alignment
        + weights.stage1_span * stage1_span
        + weights.path * path
        + weights.order * order
        + weights.transition * transition
        + weights.relation * relation
        + weights.stop * stop
        + weights.length * length
        + weights.variance * variance
        + weights.covariance * covariance
    )
    predicted_segment_count = torch.stack(
        (
            prompt_anchor.active_probabilities.sum(dim=1).mean(),
            prompt_view.active_probabilities.sum(dim=1).mean(),
        )
    ).mean()
    return VisualAnswerTrajectoryLoss(
        loss=loss,
        prompt_nce=prompt_nce,
        prompt_alignment=prompt_alignment,
        segment_read_nce=segment_read_nce,
        segment_read_alignment=segment_read_alignment,
        answer_nce=answer_nce,
        answer_alignment=answer_alignment,
        stage1_answer=stage1_answer,
        span_nce=span_nce,
        span_alignment=span_alignment,
        stage1_span=stage1_span,
        path=path,
        order=order,
        transition=transition,
        relation=relation,
        stop=stop,
        length=length,
        variance=variance,
        covariance=covariance,
        prompt_batch_top1=prompt_top1,
        segment_read_batch_top1=segment_read_top1,
        answer_batch_top1=answer_top1,
        span_batch_top1=span_top1,
        prompt_cosine=prompt_cosine,
        segment_read_cosine=segment_read_cosine,
        answer_cosine=answer_cosine,
        span_cosine=span_cosine,
        predicted_segment_count=predicted_segment_count,
    )


def set_v39_stage_trainability(
    model: VisualAnswerTrajectoryModel,
    stage: str,
) -> None:
    if stage not in {"trajectory-head", "full-path-adaptation"}:
        raise ValueError("V39 has no such trainability stage")
    model.requires_grad_(True)
    if stage == "trajectory-head":
        model.freeze_reader()
    else:
        model.unfreeze_reader()


def _normalization_and_bias_ids(model: nn.Module) -> set[int]:
    identifiers: set[int] = set()
    for module in model.modules():
        if isinstance(
            module,
            (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d),
        ):
            identifiers.update(id(parameter) for parameter in module.parameters(False))
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter):
            identifiers.add(id(bias))
    return identifiers


def visual_answer_trajectory_optimizer_groups(
    model: VisualAnswerTrajectoryModel,
    *,
    head_learning_rate: float,
    reader_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    if min(head_learning_rate, weight_decay) < 0 or reader_learning_rate < 0:
        raise ValueError("V39 optimizer settings cannot be negative")
    no_decay = _normalization_and_bias_ids(model)
    groups: list[dict[str, Any]] = []
    for role, prefix, learning_rate in (
        ("reader", "reader.", reader_learning_rate),
        ("reader-head", "prompt_head.", reader_learning_rate),
        ("trajectory", "", head_learning_rate),
    ):
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                name.startswith(prefix)
                if prefix
                else not name.startswith(("reader.", "prompt_head."))
            )
        ]
        for decay in (True, False):
            parameters = [
                parameter
                for _name, parameter in named
                if (id(parameter) not in no_decay) == decay
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": weight_decay if decay else 0.0,
                        "role": role,
                        "decay": decay,
                    }
                )
    covered = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(covered) != len(set(covered)) or set(covered) != set(expected):
        raise RuntimeError("V39 optimizer groups do not partition trainable parameters")
    return groups


class VisualAnswerTrajectoryEMA:
    def __init__(
        self,
        model: nn.Module,
        names: Sequence[str],
        *,
        decay: float,
    ) -> None:
        if not 0 < decay < 1:
            raise ValueError("V39 EMA decay must be in (0,1)")
        parameters = dict(model.named_parameters())
        if set(names) != set(parameters):
            raise ValueError("V39 EMA must cover every deployable parameter")
        self.decay = float(decay)
        self.names = tuple(names)
        self.shadow = {
            name: parameters[name].detach().float().clone() for name in self.names
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters) != set(self.names):
            raise ValueError("V39 EMA model parameter set changed")
        for name in self.names:
            value = parameters[name].detach().float()
            self.shadow[name].lerp_(value, 1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters) != set(self.names):
            raise ValueError("V39 EMA destination parameter set changed")
        for name in self.names:
            parameters[name].copy_(self.shadow[name].to(parameters[name]))

    def state_dict(self, *, cpu: bool = True) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "names": self.names,
            "shadow": {
                name: value.detach().cpu().clone() if cpu else value.detach().clone()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state.get("decay", -1)) != self.decay:
            raise ValueError("V39 EMA decay changed")
        if tuple(state.get("names", ())) != self.names:
            raise ValueError("V39 EMA parameter names changed")
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.names):
            raise ValueError("V39 EMA shadow is incomplete")
        for name in self.names:
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape:
                raise ValueError(f"V39 EMA tensor changed for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name]))


__all__ = [
    "V39_LOSS_WEIGHTS",
    "VisualAnswerTrajectoryEMA",
    "VisualAnswerTrajectoryGlobalCandidates",
    "VisualAnswerTrajectoryLoss",
    "VisualAnswerTrajectoryLossWeights",
    "VisualAnswerTrajectorySegmentCandidates",
    "VisualAnswerTrajectoryTargetBank",
    "VisualAnswerTrajectoryTargets",
    "set_v39_stage_trainability",
    "variance_covariance_loss",
    "visual_answer_trajectory_loss",
    "visual_answer_trajectory_optimizer_groups",
]

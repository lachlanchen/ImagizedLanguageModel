from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_path_alignment import (
    VisualPathAlignmentModel,
    VisualPathAlignmentOutput,
)
from .visual_semantic_distillation_data import V37_TARGET_ARCHITECTURE
from .visual_semantic_distillation_training import (
    VisualSemanticDistillationEMA,
    centered_effective_rank,
)


@dataclass(frozen=True)
class VisualPathAlignmentLossWeights:
    prompt_nce: float = 0.40
    prompt_alignment: float = 3.00
    answer_read_nce: float = 0.20
    answer_read_alignment: float = 1.50
    plan_nce: float = 0.60
    plan_alignment: float = 4.00
    margin: float = 0.50
    exact_path: float = 0.75
    semantic_path: float = 0.50
    binding: float = 0.50
    relation: float = 0.10
    variance: float = 0.02
    covariance: float = 0.001
    length: float = 0.10
    temperature: float = 0.05
    negative_margin: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "temperature":
                if value <= 0:
                    raise ValueError("V38 contrastive temperature must be positive")
            elif value < 0:
                raise ValueError("V38 loss weights must be non-negative")


V38_LOSS_WEIGHTS = VisualPathAlignmentLossWeights()


@dataclass
class VisualPathAlignmentTargets:
    prompt: torch.Tensor
    answer: torch.Tensor
    length: torch.Tensor
    bank_indices: torch.Tensor

    def validate(self) -> None:
        batch = self.prompt.shape[0]
        if self.prompt.ndim != 2 or self.answer.shape != self.prompt.shape:
            raise ValueError("V38 prompt and answer targets must match [B,D]")
        if self.length.shape != (batch,):
            raise ValueError("V38 target lengths must be [B]")
        if self.bank_indices.shape != (batch,) or self.bank_indices.dtype != torch.long:
            raise ValueError("V38 bank indices must be long [B]")
        if not all(
            torch.is_floating_point(value)
            for value in (self.prompt, self.answer, self.length)
        ):
            raise TypeError("V38 targets and lengths must be floating")


@dataclass
class VisualPathAlignmentCandidates:
    prompt: torch.Tensor
    answer: torch.Tensor
    positive_labels: torch.Tensor
    nearest_labels: torch.Tensor
    bank_indices: torch.Tensor

    def validate(self, *, batch: int, dimension: int) -> None:
        count = self.prompt.shape[0]
        if self.prompt.shape != (count, dimension) or self.answer.shape != self.prompt.shape:
            raise ValueError("V38 candidate matrices have invalid shapes")
        if self.positive_labels.shape != (batch,):
            raise ValueError("V38 positive labels do not align")
        if self.nearest_labels.ndim != 2 or self.nearest_labels.shape[0] != batch:
            raise ValueError("V38 nearest labels must be [B,H]")
        if self.nearest_labels.shape[1] < 1:
            raise ValueError("V38 requires at least one nearest negative")
        if self.bank_indices.shape != (count,):
            raise ValueError("V38 candidate bank indices do not align")
        if any(
            value.dtype != torch.long
            for value in (self.positive_labels, self.nearest_labels, self.bank_indices)
        ):
            raise TypeError("V38 candidate indices must be long tensors")
        for labels in (self.positive_labels, self.nearest_labels.reshape(-1)):
            if not bool(((labels >= 0) & (labels < count)).all()):
                raise ValueError("V38 candidate label lies outside the matrix")


@dataclass
class VisualPathAlignmentTargetBank:
    identifiers: tuple[str, ...]
    prompt_targets: torch.Tensor
    answer_targets: torch.Tensor
    lengths: torch.Tensor
    teacher_mean: torch.Tensor
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.identifiers)
        if count < 2 or len(set(self.identifiers)) != count:
            raise ValueError("V38 target identifiers must be unique")
        if self.prompt_targets.ndim != 2 or self.prompt_targets.shape[0] != count:
            raise ValueError("V38 prompt bank has an invalid shape")
        if self.answer_targets.shape != self.prompt_targets.shape:
            raise ValueError("V38 answer bank does not align")
        dimension = self.prompt_targets.shape[1]
        if dimension < 8 or self.lengths.shape != (count,):
            raise ValueError("V38 target geometry is invalid")
        if self.teacher_mean.shape != (dimension,):
            raise ValueError("V38 teacher mean has an invalid shape")
        for name, value in (
            ("prompt", self.prompt_targets),
            ("answer", self.answer_targets),
            ("length", self.lengths),
            ("teacher_mean", self.teacher_mean),
        ):
            if not torch.is_floating_point(value) or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"V38 {name} tensor must be finite floating data")
        if not bool(((self.prompt_targets.float().norm(dim=-1) - 1).abs() < 0.01).all()):
            raise ValueError("V38 prompt targets are not normalized")
        if not bool(((self.answer_targets.float().norm(dim=-1) - 1).abs() < 0.01).all()):
            raise ValueError("V38 answer targets are not normalized")
        self._index = {
            identifier: index for index, identifier in enumerate(self.identifiers)
        }
        self._nearest_cache: dict[tuple[int, float], torch.Tensor] = {}

    @classmethod
    def from_v37_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> VisualPathAlignmentTargetBank:
        if state.get("architecture") != V37_TARGET_ARCHITECTURE:
            raise ValueError("V38 requires a V37 detached target bank")
        return cls(
            identifiers=tuple(str(value) for value in state["identifiers"]),
            prompt_targets=state["prompt_targets"],
            answer_targets=state["answer_targets"],
            lengths=state["lengths"],
            teacher_mean=state["teacher_mean"],
            receipt=dict(state["receipt"]),
        )

    def lookup(
        self,
        identifiers: Sequence[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualPathAlignmentTargets:
        try:
            indices = torch.tensor(
                [self._index[identifier] for identifier in identifiers],
                dtype=torch.long,
            )
        except KeyError as error:
            raise KeyError(f"V38 target bank lacks {error.args[0]!r}") from error
        result = VisualPathAlignmentTargets(
            prompt=self.prompt_targets[indices].to(device=device, dtype=dtype),
            answer=self.answer_targets[indices].to(device=device, dtype=dtype),
            length=self.lengths[indices].to(device=device, dtype=torch.float32),
            bank_indices=indices.to(device=device),
        )
        result.validate()
        return result

    def nearest_answer_indices(
        self,
        *,
        neighbors: int,
        teacher_ceiling: float,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        if neighbors < 1 or neighbors >= len(self.identifiers):
            raise ValueError("V38 nearest-neighbor count is invalid")
        if not -1 < teacher_ceiling < 1:
            raise ValueError("V38 nearest-neighbor ceiling must be in (-1,1)")
        key = (int(neighbors), float(teacher_ceiling))
        cached = self._nearest_cache.get(key)
        if cached is not None:
            return cached
        answer = F.normalize(self.answer_targets.float(), dim=-1)
        rows: list[torch.Tensor] = []
        all_indices = torch.arange(len(answer))
        for start in range(0, len(answer), chunk_size):
            stop = min(start + chunk_size, len(answer))
            similarities = answer[start:stop] @ answer.T
            local_indices = torch.arange(start, stop)
            similarities[
                torch.arange(stop - start),
                local_indices,
            ] = float("-inf")
            similarities.masked_fill_(similarities >= teacher_ceiling, float("-inf"))
            values, indices = similarities.topk(neighbors, dim=1)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("V38 cannot find enough valid nearest negatives")
            if bool((indices == local_indices[:, None]).any()):
                raise RuntimeError("V38 nearest-negative table contains a positive")
            if not bool(((indices >= all_indices.min()) & (indices <= all_indices.max())).all()):
                raise RuntimeError("V38 nearest-negative table lies outside the bank")
            rows.append(indices.cpu())
        result = torch.cat(rows)
        self._nearest_cache[key] = result
        return result

    def candidate_set(
        self,
        positive_indices: torch.Tensor,
        *,
        count: int,
        seed: int,
        neighbors: int,
        teacher_ceiling: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualPathAlignmentCandidates:
        positives = [int(value) for value in positive_indices.detach().cpu()]
        if len(set(positives)) != len(positives):
            raise ValueError("V38 physical batch contains duplicate positives")
        if not positives or min(positives) < 0 or max(positives) >= len(self.identifiers):
            raise ValueError("V38 positive index lies outside target bank")
        nearest = self.nearest_answer_indices(
            neighbors=neighbors,
            teacher_ceiling=teacher_ceiling,
        )[positives]
        required = set(positives)
        required.update(int(value) for value in nearest.reshape(-1))
        if len(required) > count or count > len(self.identifiers):
            raise ValueError("V38 candidate count cannot contain injected negatives")

        selected = list(positives)
        selected_set = set(selected)
        for row in nearest:
            for value in row:
                candidate = int(value)
                if candidate not in selected_set:
                    selected.append(candidate)
                    selected_set.add(candidate)
        rng = random.Random(int(seed))
        offset = rng.randrange(len(self.identifiers))
        stride = rng.randrange(1, len(self.identifiers))
        while math.gcd(stride, len(self.identifiers)) != 1:
            stride = stride % (len(self.identifiers) - 1) + 1
        position = 0
        while len(selected) < count:
            candidate = (offset + position * stride) % len(self.identifiers)
            position += 1
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        label_by_bank_index = {value: index for index, value in enumerate(selected)}
        labels = torch.tensor(
            [label_by_bank_index[value] for value in positives],
            dtype=torch.long,
            device=device,
        )
        nearest_labels = torch.tensor(
            [
                [label_by_bank_index[int(value)] for value in row]
                for row in nearest
            ],
            dtype=torch.long,
            device=device,
        )
        bank_indices = torch.tensor(selected, dtype=torch.long)
        result = VisualPathAlignmentCandidates(
            prompt=self.prompt_targets[bank_indices].to(device=device, dtype=dtype),
            answer=self.answer_targets[bank_indices].to(device=device, dtype=dtype),
            positive_labels=labels,
            nearest_labels=nearest_labels,
            bank_indices=bank_indices.to(device=device),
        )
        result.validate(batch=len(positives), dimension=self.prompt_targets.shape[1])
        return result


@torch.no_grad()
def orthogonal_prompt_answer_rotation(
    bank: VisualPathAlignmentTargetBank,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt = F.normalize(bank.prompt_targets.float(), dim=-1)
    answer = F.normalize(bank.answer_targets.float(), dim=-1)
    left, _singular, right = torch.linalg.svd(prompt.T @ answer, full_matrices=False)
    rotation = left @ right
    mapped = F.normalize(prompt @ rotation, dim=-1)
    similarities = mapped @ answer.T
    ranking = similarities.argsort(dim=1, descending=True)
    correct = torch.arange(len(prompt))
    positions = (ranking == correct[:, None]).nonzero(as_tuple=False)[:, 1]
    receipt = {
        "orthogonality_max_error": float(
            (rotation.T @ rotation - torch.eye(rotation.shape[0])).abs().max()
        ),
        "train_top1": float((positions == 0).float().mean()),
        "train_top5": float((positions < min(5, len(prompt))).float().mean()),
        "train_mrr": float((1 / (positions.float() + 1)).mean()),
        "train_cosine": float((mapped * answer).sum(dim=-1).mean()),
    }
    return rotation, receipt


@dataclass
class VisualPathAlignmentLoss:
    loss: torch.Tensor
    prompt_nce: torch.Tensor
    prompt_alignment: torch.Tensor
    answer_read_nce: torch.Tensor
    answer_read_alignment: torch.Tensor
    plan_nce: torch.Tensor
    plan_alignment: torch.Tensor
    margin: torch.Tensor
    exact_path: torch.Tensor
    semantic_path: torch.Tensor
    binding: torch.Tensor
    relation: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    length: torch.Tensor
    prompt_batch_top1: torch.Tensor
    answer_read_batch_top1: torch.Tensor
    plan_batch_top1: torch.Tensor
    prompt_cosine: torch.Tensor
    answer_read_cosine: torch.Tensor
    plan_cosine: torch.Tensor
    nearest_negative_cosine: torch.Tensor
    answer_correction_norm: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.__dict__.items()}


def _route_objective(
    states: Sequence[torch.Tensor],
    target: torch.Tensor,
    candidates: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nces: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    top1: list[torch.Tensor] = []
    cosines: list[torch.Tensor] = []
    for state in states:
        logits = state.float() @ candidates.float().T / temperature
        cosine = F.cosine_similarity(state.float(), target.float(), dim=-1)
        nces.append(F.cross_entropy(logits, labels))
        distances.append((1 - cosine).mean())
        top1.append((logits.argmax(dim=-1) == labels).float().mean())
        cosines.append(cosine.mean())
    return (
        torch.stack(nces).mean(),
        torch.stack(distances).mean(),
        torch.stack(top1).mean(),
        torch.stack(cosines).mean(),
    )


def _mean_cosine_distance(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return torch.stack(
        [
            (
                1
                - F.cosine_similarity(first.float(), second.float(), dim=-1)
            ).mean()
            for first, second in pairs
        ]
    ).mean()


def _nearest_margin(
    states: Sequence[torch.Tensor],
    target: torch.Tensor,
    candidates: VisualPathAlignmentCandidates,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    hard_cosines: list[torch.Tensor] = []
    for state in states:
        similarities = state.float() @ candidates.answer.float().T
        nearest = similarities.gather(1, candidates.nearest_labels)
        hard = nearest.max(dim=1).values
        correct = F.cosine_similarity(state.float(), target.float(), dim=-1)
        losses.append(F.relu(margin - correct + hard).mean())
        hard_cosines.append(hard.mean())
    return torch.stack(losses).mean(), torch.stack(hard_cosines).mean()


def _relation_loss(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if student.shape[0] < 2:
        return student.float().sum() * 0
    return F.mse_loss(
        student.float() @ student.float().T,
        target.float() @ target.float().T,
    )


def variance_covariance_loss(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("V38 variance control requires [N,D] with N >= 2")
    scaled = states.float() * math.sqrt(states.shape[1])
    centered = scaled - scaled.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1 - standard_deviation).mean()
    covariance_matrix = centered.T @ centered / max(1, states.shape[0] - 1)
    covariance = (
        covariance_matrix.square().sum() - covariance_matrix.diagonal().square().sum()
    ) / states.shape[1]
    return variance, covariance


def visual_path_alignment_loss(
    prompt_anchor: VisualPathAlignmentOutput,
    prompt_view: VisualPathAlignmentOutput,
    semantic_view: VisualPathAlignmentOutput,
    answer_anchor: VisualPathAlignmentOutput,
    answer_view: VisualPathAlignmentOutput,
    targets: VisualPathAlignmentTargets,
    candidates: VisualPathAlignmentCandidates,
    *,
    weights: VisualPathAlignmentLossWeights = V38_LOSS_WEIGHTS,
) -> VisualPathAlignmentLoss:
    targets.validate()
    batch, dimension = targets.prompt.shape
    candidates.validate(batch=batch, dimension=dimension)
    outputs = (prompt_anchor, prompt_view, semantic_view, answer_anchor, answer_view)
    if any(output.prompt_state.shape != (batch, dimension) for output in outputs):
        raise ValueError("V38 visual outputs and semantic targets do not align")

    prompt_states = (
        prompt_anchor.prompt_state,
        prompt_view.prompt_state,
        semantic_view.prompt_state,
    )
    answer_read_states = (answer_anchor.prompt_state, answer_view.prompt_state)
    plan_states = (
        prompt_anchor.answer_state,
        prompt_view.answer_state,
        semantic_view.answer_state,
    )
    prompt_nce, prompt_alignment, prompt_top1, prompt_cosine = _route_objective(
        prompt_states,
        targets.prompt,
        candidates.prompt,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    answer_read_nce, answer_read_alignment, answer_top1, answer_cosine = (
        _route_objective(
            answer_read_states,
            targets.answer,
            candidates.answer,
            candidates.positive_labels,
            temperature=weights.temperature,
        )
    )
    plan_nce, plan_alignment, plan_top1, plan_cosine = _route_objective(
        plan_states,
        targets.answer,
        candidates.answer,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    nearest_margin, nearest_cosine = _nearest_margin(
        plan_states,
        targets.answer,
        candidates,
        margin=weights.negative_margin,
    )
    exact_path = _mean_cosine_distance(
        (
            (prompt_anchor.prompt_state, prompt_view.prompt_state),
            (prompt_anchor.answer_state, prompt_view.answer_state),
            (answer_anchor.prompt_state, answer_view.prompt_state),
        )
    )
    semantic_path = _mean_cosine_distance(
        (
            (prompt_anchor.prompt_state, semantic_view.prompt_state),
            (prompt_anchor.answer_state, semantic_view.answer_state),
        )
    )
    binding = _mean_cosine_distance(
        (
            (prompt_anchor.answer_state, answer_anchor.prompt_state),
            (prompt_view.answer_state, answer_view.prompt_state),
        )
    )
    relation = _relation_loss(prompt_anchor.prompt_state, targets.prompt) + (
        _relation_loss(prompt_anchor.answer_state, targets.answer)
    )
    variance, covariance = variance_covariance_loss(
        torch.cat(
            (
                prompt_anchor.prompt_state,
                prompt_view.prompt_state,
                semantic_view.prompt_state,
                answer_anchor.prompt_state,
                answer_view.prompt_state,
                prompt_anchor.answer_state,
                prompt_view.answer_state,
                semantic_view.answer_state,
            ),
            dim=0,
        )
    )
    length = torch.stack(
        [
            F.smooth_l1_loss(output.length.float(), targets.length.float())
            for output in (prompt_anchor, prompt_view, semantic_view)
        ]
    ).mean()
    correction_norm = torch.stack(
        [output.answer_correction.float().norm(dim=-1).mean() for output in outputs[:3]]
    ).mean()
    loss = (
        weights.prompt_nce * prompt_nce
        + weights.prompt_alignment * prompt_alignment
        + weights.answer_read_nce * answer_read_nce
        + weights.answer_read_alignment * answer_read_alignment
        + weights.plan_nce * plan_nce
        + weights.plan_alignment * plan_alignment
        + weights.margin * nearest_margin
        + weights.exact_path * exact_path
        + weights.semantic_path * semantic_path
        + weights.binding * binding
        + weights.relation * relation
        + weights.variance * variance
        + weights.covariance * covariance
        + weights.length * length
    )
    return VisualPathAlignmentLoss(
        loss=loss,
        prompt_nce=prompt_nce,
        prompt_alignment=prompt_alignment,
        answer_read_nce=answer_read_nce,
        answer_read_alignment=answer_read_alignment,
        plan_nce=plan_nce,
        plan_alignment=plan_alignment,
        margin=nearest_margin,
        exact_path=exact_path,
        semantic_path=semantic_path,
        binding=binding,
        relation=relation,
        variance=variance,
        covariance=covariance,
        length=length,
        prompt_batch_top1=prompt_top1,
        answer_read_batch_top1=answer_top1,
        plan_batch_top1=plan_top1,
        prompt_cosine=prompt_cosine,
        answer_read_cosine=answer_cosine,
        plan_cosine=plan_cosine,
        nearest_negative_cosine=nearest_cosine,
        answer_correction_norm=correction_norm,
    )


def set_v38_stage_trainability(model: VisualPathAlignmentModel, stage: str) -> None:
    if stage not in {"head-realignment", "full-path-adaptation"}:
        raise ValueError("V38 has no such trainability stage")
    model.requires_grad_(True)
    if stage == "head-realignment":
        model.freeze_reader()
    else:
        model.unfreeze_reader()


def _normalization_and_bias_ids(model: nn.Module) -> set[int]:
    identifiers: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias" or "norm" in module.__class__.__name__.lower():
                identifiers.add(id(parameter))
    return identifiers


def visual_path_alignment_optimizer_groups(
    model: VisualPathAlignmentModel,
    *,
    head_learning_rate: float,
    reader_learning_rate: float,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    if min(head_learning_rate, reader_learning_rate, weight_decay) < 0:
        raise ValueError("V38 optimizer values must be non-negative")
    if max(head_learning_rate, reader_learning_rate) <= 0:
        raise ValueError("V38 optimizer requires a positive learning rate")
    no_decay = _normalization_and_bias_ids(model)
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        role = "reader" if name.startswith("reader.") else "head"
        grouped.setdefault((role, id(parameter) not in no_decay), []).append(parameter)
    rates = {"head": head_learning_rate, "reader": reader_learning_rate}
    result: list[dict[str, Any]] = []
    for role in ("head", "reader"):
        for decay in (True, False):
            parameters = grouped.get((role, decay), [])
            if parameters:
                result.append(
                    {
                        "params": parameters,
                        "lr": rates[role],
                        "weight_decay": weight_decay if decay else 0.0,
                        "role": role,
                        "decay": decay,
                    }
                )
    return result


def set_v38_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    head: float,
    reader: float,
) -> None:
    rates = {"head": float(head), "reader": float(reader)}
    if min(rates.values()) < 0:
        raise ValueError("V38 optimizer learning rates must be non-negative")
    for group in optimizer.param_groups:
        role = str(group.get("role", ""))
        if role not in rates:
            raise ValueError("V38 optimizer group has no recognized role")
        group["lr"] = rates[role]


VisualPathAlignmentEMA = VisualSemanticDistillationEMA


__all__ = [
    "V38_LOSS_WEIGHTS",
    "VisualPathAlignmentCandidates",
    "VisualPathAlignmentEMA",
    "VisualPathAlignmentLoss",
    "VisualPathAlignmentLossWeights",
    "VisualPathAlignmentTargetBank",
    "VisualPathAlignmentTargets",
    "centered_effective_rank",
    "orthogonal_prompt_answer_rotation",
    "set_v38_optimizer_learning_rates",
    "set_v38_stage_trainability",
    "variance_covariance_loss",
    "visual_path_alignment_loss",
    "visual_path_alignment_optimizer_groups",
]

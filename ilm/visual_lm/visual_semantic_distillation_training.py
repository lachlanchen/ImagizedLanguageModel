from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_semantic_distillation import (
    VisualSemanticDistillationModel,
    VisualSemanticDistillationOutput,
)
from .visual_semantic_distillation_data import (
    V37_SEMANTIC_DIM,
    V37_TARGET_ARCHITECTURE,
)


@dataclass(frozen=True)
class VisualSemanticDistillationLossWeights:
    prompt: float = 0.70
    answer: float = 0.70
    plan: float = 1.00
    margin: float = 0.50
    relation: float = 0.20
    view: float = 0.20
    variance: float = 0.05
    covariance: float = 0.005
    length: float = 0.02
    residual: float = 0.01
    temperature: float = 0.05
    negative_teacher_ceiling: float = 0.85
    negative_margin: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "temperature":
                if value <= 0:
                    raise ValueError("V37 contrastive temperature must be positive")
            elif value < 0:
                raise ValueError("V37 loss weights must be non-negative")
        if not 0 < self.negative_teacher_ceiling < 1:
            raise ValueError("V37 teacher ceiling must be in (0,1)")


V37_LOSS_WEIGHTS = VisualSemanticDistillationLossWeights()


@dataclass
class VisualSemanticDistillationTargets:
    prompt: torch.Tensor
    answer: torch.Tensor
    length: torch.Tensor
    bank_indices: torch.Tensor

    def validate(self) -> None:
        batch = self.prompt.shape[0]
        if self.prompt.ndim != 2 or self.answer.shape != self.prompt.shape:
            raise ValueError("V37 prompt and answer targets must match [B,D]")
        if self.length.shape != (batch,):
            raise ValueError("V37 target lengths must be [B]")
        if self.bank_indices.shape != (batch,) or self.bank_indices.dtype != torch.long:
            raise ValueError("V37 bank indices must be long [B]")
        if not all(
            torch.is_floating_point(value)
            for value in (self.prompt, self.answer, self.length)
        ):
            raise TypeError("V37 semantic targets and lengths must be floating")


@dataclass
class VisualSemanticDistillationCandidates:
    prompt: torch.Tensor
    answer: torch.Tensor
    positive_labels: torch.Tensor
    bank_indices: torch.Tensor

    def validate(self, *, batch: int, dimension: int) -> None:
        candidates = self.prompt.shape[0]
        if self.prompt.shape != (candidates, dimension):
            raise ValueError("V37 prompt candidates have an invalid shape")
        if self.answer.shape != self.prompt.shape:
            raise ValueError("V37 candidate modalities do not align")
        if self.positive_labels.shape != (batch,):
            raise ValueError("V37 positive labels do not align")
        if self.bank_indices.shape != (candidates,):
            raise ValueError("V37 candidate bank indices do not align")
        if (
            self.positive_labels.dtype != torch.long
            or self.bank_indices.dtype != torch.long
        ):
            raise TypeError("V37 candidate indices must be long tensors")
        if not bool(
            ((self.positive_labels >= 0) & (self.positive_labels < candidates)).all()
        ):
            raise ValueError("V37 positive label lies outside candidate matrix")


@dataclass
class VisualSemanticDistillationTargetBank:
    identifiers: tuple[str, ...]
    prompt_targets: torch.Tensor
    answer_targets: torch.Tensor
    lengths: torch.Tensor
    teacher_mean: torch.Tensor
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.identifiers)
        if count < 1 or len(set(self.identifiers)) != count:
            raise ValueError("V37 target identifiers must be nonempty and unique")
        if self.prompt_targets.ndim != 2 or self.prompt_targets.shape[0] != count:
            raise ValueError("V37 prompt target bank has an invalid shape")
        if self.answer_targets.shape != self.prompt_targets.shape:
            raise ValueError("V37 answer target bank does not align")
        if self.prompt_targets.shape[1] != V37_SEMANTIC_DIM:
            raise ValueError("V37 semantic target width changed")
        if self.lengths.shape != (count,):
            raise ValueError("V37 target lengths have an invalid shape")
        if self.teacher_mean.shape != (V37_SEMANTIC_DIM,):
            raise ValueError("V37 teacher mean has an invalid shape")
        for name, value in (
            ("prompt", self.prompt_targets),
            ("answer", self.answer_targets),
            ("length", self.lengths),
            ("teacher_mean", self.teacher_mean),
        ):
            if not torch.is_floating_point(value) or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"V37 {name} target tensor must be finite floating data"
                )
        prompt_norms = self.prompt_targets.float().norm(dim=-1)
        answer_norms = self.answer_targets.float().norm(dim=-1)
        if not bool(((prompt_norms - 1).abs() < 0.01).all()):
            raise ValueError("V37 prompt targets are not normalized")
        if not bool(((answer_norms - 1).abs() < 0.01).all()):
            raise ValueError("V37 answer targets are not normalized")
        self._index = {
            identifier: index for index, identifier in enumerate(self.identifiers)
        }

    def lookup(
        self,
        identifiers: Sequence[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualSemanticDistillationTargets:
        try:
            indices = torch.tensor(
                [self._index[identifier] for identifier in identifiers],
                dtype=torch.long,
            )
        except KeyError as error:
            raise KeyError(f"V37 target bank lacks {error.args[0]!r}") from error
        targets = VisualSemanticDistillationTargets(
            prompt=self.prompt_targets[indices].to(device=device, dtype=dtype),
            answer=self.answer_targets[indices].to(device=device, dtype=dtype),
            length=self.lengths[indices].to(device=device, dtype=torch.float32),
            bank_indices=indices.to(device=device),
        )
        targets.validate()
        return targets

    def candidate_set(
        self,
        positive_indices: torch.Tensor,
        *,
        count: int,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualSemanticDistillationCandidates:
        positives = [int(value) for value in positive_indices.detach().cpu()]
        if len(set(positives)) != len(positives):
            raise ValueError("V37 physical batch contains duplicate positives")
        if (
            not positives
            or min(positives) < 0
            or max(positives) >= len(self.identifiers)
        ):
            raise ValueError("V37 positive index lies outside target bank")
        if not len(positives) <= count <= len(self.identifiers):
            raise ValueError("V37 candidate count cannot contain every positive")

        selected = list(positives)
        selected_set = set(selected)
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
        bank_indices = torch.tensor(selected, dtype=torch.long)
        candidates = VisualSemanticDistillationCandidates(
            prompt=self.prompt_targets[bank_indices].to(device=device, dtype=dtype),
            answer=self.answer_targets[bank_indices].to(device=device, dtype=dtype),
            positive_labels=labels,
            bank_indices=bank_indices.to(device=device),
        )
        candidates.validate(
            batch=len(positives),
            dimension=self.prompt_targets.shape[1],
        )
        return candidates

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": V37_TARGET_ARCHITECTURE,
            "identifiers": list(self.identifiers),
            "prompt_targets": self.prompt_targets.detach().cpu(),
            "answer_targets": self.answer_targets.detach().cpu(),
            "lengths": self.lengths.detach().cpu(),
            "teacher_mean": self.teacher_mean.detach().cpu(),
            "receipt": self.receipt,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> VisualSemanticDistillationTargetBank:
        if state.get("architecture") != V37_TARGET_ARCHITECTURE:
            raise ValueError("target bank is not V37")
        return cls(
            identifiers=tuple(str(value) for value in state["identifiers"]),
            prompt_targets=state["prompt_targets"],
            answer_targets=state["answer_targets"],
            lengths=state["lengths"],
            teacher_mean=state["teacher_mean"],
            receipt=dict(state["receipt"]),
        )


@dataclass
class VisualSemanticDistillationLoss:
    loss: torch.Tensor
    prompt: torch.Tensor
    answer: torch.Tensor
    plan: torch.Tensor
    margin: torch.Tensor
    relation: torch.Tensor
    view: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    length: torch.Tensor
    residual: torch.Tensor
    prompt_batch_top1: torch.Tensor
    answer_batch_top1: torch.Tensor
    plan_batch_top1: torch.Tensor
    prompt_cosine: torch.Tensor
    answer_cosine: torch.Tensor
    plan_cosine: torch.Tensor
    hard_negative_cosine: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.__dict__.items()}


def _candidate_objective(
    state: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = state.float() @ candidates.float().T / temperature
    nce = F.cross_entropy(logits, labels)
    cosine = (1 - F.cosine_similarity(state.float(), target.float(), dim=-1)).mean()
    top1 = (logits.argmax(dim=-1) == labels).float().mean()
    return nce, cosine, top1


def _hard_negative_margin(
    plan: torch.Tensor,
    target: torch.Tensor,
    candidates: VisualSemanticDistillationCandidates,
    *,
    teacher_ceiling: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan_similarity = plan.float() @ candidates.answer.float().T
    teacher_similarity = target.float() @ candidates.answer.float().T
    candidate_positions = torch.arange(
        candidates.answer.shape[0],
        device=plan.device,
    )
    positive = candidate_positions.unsqueeze(0) == candidates.positive_labels.unsqueeze(
        1
    )
    valid = (~positive) & (teacher_similarity < teacher_ceiling)
    selected = plan_similarity.masked_fill(~valid, float("-inf")).argmax(dim=1)
    fallback = (candidates.positive_labels + 1) % candidates.answer.shape[0]
    selected = torch.where(valid.any(dim=1), selected, fallback)
    hard_cosine = plan_similarity.gather(1, selected.unsqueeze(1)).squeeze(1)
    correct_cosine = F.cosine_similarity(plan.float(), target.float(), dim=-1)
    return F.relu(margin - correct_cosine + hard_cosine).mean(), hard_cosine.mean()


def _relation_loss(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if student.shape[0] < 2:
        return student.float().sum() * 0
    return F.mse_loss(
        student.float() @ student.float().T, target.float() @ target.float().T
    )


def vicreg_variance_covariance(
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("V37 VICReg requires at least two [B,D] states")
    scaled = states.float() * math.sqrt(states.shape[1])
    centered = scaled - scaled.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1 - standard_deviation).mean()
    covariance_matrix = centered.T @ centered / max(1, states.shape[0] - 1)
    covariance = (
        covariance_matrix.square().sum() - covariance_matrix.diagonal().square().sum()
    ) / states.shape[1]
    return variance, covariance


def visual_semantic_distillation_loss(
    prompt: VisualSemanticDistillationOutput,
    prompt_view: VisualSemanticDistillationOutput,
    answer: VisualSemanticDistillationOutput,
    answer_view: VisualSemanticDistillationOutput,
    targets: VisualSemanticDistillationTargets,
    candidates: VisualSemanticDistillationCandidates,
    *,
    weights: VisualSemanticDistillationLossWeights = V37_LOSS_WEIGHTS,
) -> VisualSemanticDistillationLoss:
    targets.validate()
    batch, dimension = targets.prompt.shape
    candidates.validate(batch=batch, dimension=dimension)
    outputs = (prompt, prompt_view, answer, answer_view)
    if any(output.semantic_state.shape != (batch, dimension) for output in outputs):
        raise ValueError("V37 visual outputs and semantic targets do not align")

    prompt_nce, prompt_cosine_a, prompt_top1 = _candidate_objective(
        prompt.semantic_state,
        targets.prompt,
        candidates.prompt,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    prompt_view_nce, prompt_cosine_b, _ = _candidate_objective(
        prompt_view.semantic_state,
        targets.prompt,
        candidates.prompt,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    prompt_loss = 0.5 * (prompt_nce + prompt_view_nce) + 0.25 * (
        prompt_cosine_a + prompt_cosine_b
    )

    answer_nce, answer_cosine_a, answer_top1 = _candidate_objective(
        answer.semantic_state,
        targets.answer,
        candidates.answer,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    answer_view_nce, answer_cosine_b, _ = _candidate_objective(
        answer_view.semantic_state,
        targets.answer,
        candidates.answer,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    answer_loss = 0.5 * (answer_nce + answer_view_nce) + 0.25 * (
        answer_cosine_a + answer_cosine_b
    )

    plan_nce, plan_cosine_loss, plan_top1 = _candidate_objective(
        prompt.answer_plan,
        targets.answer,
        candidates.answer,
        candidates.positive_labels,
        temperature=weights.temperature,
    )
    plan_loss = plan_nce + 0.5 * plan_cosine_loss
    margin, hard_negative_cosine = _hard_negative_margin(
        prompt.answer_plan,
        targets.answer,
        candidates,
        teacher_ceiling=weights.negative_teacher_ceiling,
        margin=weights.negative_margin,
    )
    relation = _relation_loss(prompt.semantic_state, targets.prompt) + _relation_loss(
        answer.semantic_state,
        targets.answer,
    )
    view = (
        0.5
        * (
            1
            - F.cosine_similarity(
                prompt.semantic_state.float(),
                prompt_view.semantic_state.float(),
                dim=-1,
            )
        ).mean()
        + 0.5
        * (
            1
            - F.cosine_similarity(
                answer.semantic_state.float(),
                answer_view.semantic_state.float(),
                dim=-1,
            )
        ).mean()
    )
    variance, covariance = vicreg_variance_covariance(
        torch.cat(
            (
                prompt.semantic_state,
                answer.semantic_state,
                prompt.answer_plan,
            ),
            dim=0,
        )
    )
    length = F.smooth_l1_loss(prompt.length.float(), targets.length.float())
    residual = prompt.scaled_residual.float().square().sum(dim=-1).mean()
    loss = (
        weights.prompt * prompt_loss
        + weights.answer * answer_loss
        + weights.plan * plan_loss
        + weights.margin * margin
        + weights.relation * relation
        + weights.view * view
        + weights.variance * variance
        + weights.covariance * covariance
        + weights.length * length
        + weights.residual * residual
    )
    return VisualSemanticDistillationLoss(
        loss=loss,
        prompt=prompt_loss,
        answer=answer_loss,
        plan=plan_loss,
        margin=margin,
        relation=relation,
        view=view,
        variance=variance,
        covariance=covariance,
        length=length,
        residual=residual,
        prompt_batch_top1=prompt_top1,
        answer_batch_top1=answer_top1,
        plan_batch_top1=plan_top1,
        prompt_cosine=1 - prompt_cosine_a,
        answer_cosine=1 - answer_cosine_a,
        plan_cosine=1 - plan_cosine_loss,
        hard_negative_cosine=hard_negative_cosine,
    )


def set_v37_stage_trainability(
    model: VisualSemanticDistillationModel,
    stage: str,
) -> None:
    if stage not in {"projection-warmup", "full-visual-adaptation"}:
        raise ValueError("V37 has no such trainability stage")
    model.requires_grad_(True)
    if stage == "projection-warmup":
        model.freeze_reader()
    else:
        model.unfreeze_reader()


def _normalization_and_bias_ids(model: nn.Module) -> set[int]:
    identifiers: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias" or "norm" in module.__class__.__name__.lower():
                identifiers.add(id(parameter))
    identifiers.add(id(model.residual_logit))
    return identifiers


def visual_semantic_distillation_optimizer_groups(
    model: VisualSemanticDistillationModel,
    *,
    head_learning_rate: float,
    reader_learning_rate: float,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    if min(head_learning_rate, reader_learning_rate, weight_decay) < 0:
        raise ValueError("V37 optimizer values must be non-negative")
    if max(head_learning_rate, reader_learning_rate) <= 0:
        raise ValueError("V37 optimizer requires a positive learning rate")
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


def set_v37_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    head: float,
    reader: float,
) -> None:
    rates = {"head": float(head), "reader": float(reader)}
    if min(rates.values()) < 0:
        raise ValueError("V37 optimizer learning rates must be non-negative")
    for group in optimizer.param_groups:
        role = str(group.get("role", ""))
        if role not in rates:
            raise ValueError("V37 optimizer group has no recognized role")
        group["lr"] = rates[role]


class VisualSemanticDistillationEMA:
    def __init__(
        self,
        model: nn.Module,
        parameter_names: Sequence[str],
        *,
        decay: float = 0.999,
    ) -> None:
        if not 0 <= decay < 1:
            raise ValueError("V37 EMA decay must be in [0,1)")
        source = dict(model.named_parameters())
        self.names = tuple(dict.fromkeys(parameter_names))
        missing = set(self.names).difference(source)
        if missing:
            raise ValueError(f"V37 EMA parameters are missing: {sorted(missing)}")
        self.decay = float(decay)
        self.shadow = {
            name: source[name].detach().float().clone() for name in self.names
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].lerp_(source[name].detach().float(), 1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        destination = dict(model.named_parameters())
        for name in self.names:
            destination[name].copy_(self.shadow[name].to(destination[name]))

    def state_dict(self, *, cpu: bool = True) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "names": list(self.names),
            "shadow": {
                name: value.detach().cpu().clone() if cpu else value.detach().clone()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["names"]) != self.names or float(state["decay"]) != self.decay:
            raise ValueError("V37 EMA definition differs from checkpoint")
        for name in self.names:
            value = state["shadow"][name]
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"V37 EMA shape differs for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name]))


def centered_effective_rank(states: torch.Tensor) -> float:
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("V37 effective rank requires [N,D] with N >= 2")
    states = F.normalize(states.float(), dim=-1)
    centered = states - states.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square() / states.shape[0]
    denominator = eigenvalues.square().sum()
    return float(eigenvalues.sum().square() / denominator.clamp_min(1e-12))

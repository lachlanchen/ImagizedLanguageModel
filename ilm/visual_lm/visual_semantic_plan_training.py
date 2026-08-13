from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_semantic_plan import (
    VisualSemanticPlanModel,
    VisualSemanticPlanOutput,
    VisualSentenceImageTeacher,
)
from .visual_semantic_plan_data import V36_CHUNKS, V36_PLAN_SLOTS


@dataclass(frozen=True)
class VisualSemanticPlanLossWeights:
    global_cosine: float = 0.50
    margin: float = 0.50
    chunk_nce: float = 0.35
    chunk_cosine: float = 0.20
    view: float = 0.20
    length: float = 0.05
    temperature: float = 0.07
    negative_teacher_ceiling: float = 0.85
    negative_margin: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "temperature":
                if value <= 0.0:
                    raise ValueError("V36 contrastive temperature must be positive")
            elif value < 0.0:
                raise ValueError("V36 loss values must be non-negative")
        if not 0.0 < self.negative_teacher_ceiling < 1.0:
            raise ValueError("V36 negative teacher ceiling must be in (0,1)")


V36_LOSS_WEIGHTS = VisualSemanticPlanLossWeights()


@dataclass
class VisualSemanticTeacherTargets:
    global_plan: torch.Tensor
    chunk_plans: torch.Tensor
    chunk_active: torch.Tensor
    length: torch.Tensor

    def validate(self) -> None:
        batch = self.global_plan.shape[0]
        if self.global_plan.ndim != 2:
            raise ValueError("V36 global teacher plans must be [B,D]")
        if self.chunk_plans.shape != (
            batch,
            V36_CHUNKS,
            self.global_plan.shape[-1],
        ):
            raise ValueError("V36 chunk teacher plans have an invalid shape")
        if self.chunk_active.shape != (batch, V36_CHUNKS):
            raise ValueError("V36 chunk activity has an invalid shape")
        if self.length.shape != (batch,):
            raise ValueError("V36 target visual length has an invalid shape")
        if not all(
            torch.is_floating_point(value)
            for value in (
                self.global_plan,
                self.chunk_plans,
                self.chunk_active,
                self.length,
            )
        ):
            raise TypeError("V36 teacher targets must be floating tensors")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> VisualSemanticTeacherTargets:
        plan_dtype = self.global_plan.dtype if dtype is None else dtype
        return VisualSemanticTeacherTargets(
            global_plan=self.global_plan.to(device=device, dtype=plan_dtype),
            chunk_plans=self.chunk_plans.to(device=device, dtype=plan_dtype),
            chunk_active=self.chunk_active.to(device=device, dtype=torch.float32),
            length=self.length.to(device=device, dtype=torch.float32),
        )


@torch.no_grad()
def encode_visual_semantic_teacher_targets(
    teacher: VisualSentenceImageTeacher,
    batch: Mapping[str, torch.Tensor],
) -> VisualSemanticTeacherTargets:
    required = {
        "answer_pixels",
        "answer_mask",
        "answer_chunk_pixels",
        "answer_chunk_mask",
        "answer_length",
    }
    if not required.issubset(batch):
        raise ValueError("V36 teacher batch lacks answer raster tensors")
    answer_pixels = batch["answer_pixels"]
    answer_mask = batch["answer_mask"]
    chunk_pixels = batch["answer_chunk_pixels"]
    chunk_mask = batch["answer_chunk_mask"]
    batch_size = answer_pixels.shape[0]
    if chunk_pixels.shape[:2] != (batch_size, V36_CHUNKS):
        raise ValueError("V36 teacher answer chunks do not align")

    active = chunk_mask.sum(dim=-1) > 0
    active_pixels = chunk_pixels[active]
    active_masks = chunk_mask[active]
    combined_pixels = torch.cat((answer_pixels, active_pixels), dim=0)
    combined_masks = torch.cat((answer_mask, active_masks), dim=0)
    embeddings = teacher(combined_pixels, combined_masks).float()
    global_plan = embeddings[:batch_size]
    chunk_plans = embeddings.new_zeros(
        batch_size,
        V36_CHUNKS,
        embeddings.shape[-1],
    )
    chunk_plans[active] = embeddings[batch_size:]
    targets = VisualSemanticTeacherTargets(
        global_plan=global_plan,
        chunk_plans=chunk_plans,
        chunk_active=active.float(),
        length=batch["answer_length"].float(),
    )
    targets.validate()
    return targets


@dataclass
class VisualSemanticPlanLoss:
    loss: torch.Tensor
    global_nce: torch.Tensor
    global_cosine: torch.Tensor
    margin: torch.Tensor
    chunk_nce: torch.Tensor
    chunk_cosine: torch.Tensor
    view: torch.Tensor
    length: torch.Tensor
    batch_top1: torch.Tensor
    correct_cosine: torch.Tensor
    hard_negative_cosine: torch.Tensor
    plan_variance: torch.Tensor
    active_chunks: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "global_nce": float(self.global_nce.detach()),
            "global_cosine": float(self.global_cosine.detach()),
            "margin": float(self.margin.detach()),
            "chunk_nce": float(self.chunk_nce.detach()),
            "chunk_cosine": float(self.chunk_cosine.detach()),
            "view": float(self.view.detach()),
            "length": float(self.length.detach()),
            "batch_top1": float(self.batch_top1.detach()),
            "correct_cosine": float(self.correct_cosine.detach()),
            "hard_negative_cosine": float(self.hard_negative_cosine.detach()),
            "plan_variance": float(self.plan_variance.detach()),
            "active_chunks": float(self.active_chunks.detach()),
        }


def _hard_negative_indices(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    teacher_ceiling: float,
) -> torch.Tensor:
    if predicted.shape != targets.shape or predicted.ndim != 2:
        raise ValueError("V36 hard-negative plans must have matching [B,D] shapes")
    batch = predicted.shape[0]
    if batch < 2:
        raise ValueError("V36 contrastive training requires at least two examples")
    teacher_similarity = targets.float() @ targets.float().T
    predicted_similarity = predicted.float() @ targets.float().T
    diagonal = torch.eye(batch, dtype=torch.bool, device=predicted.device)
    valid = (~diagonal) & (teacher_similarity < teacher_ceiling)
    masked = predicted_similarity.masked_fill(~valid, float("-inf"))
    selected = masked.argmax(dim=1)
    fallback = torch.roll(torch.arange(batch, device=predicted.device), shifts=1)
    return torch.where(valid.any(dim=1), selected, fallback)


def _active_chunk_losses(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.shape != targets.shape or predicted.ndim != 3:
        raise ValueError("V36 chunk plans must have matching [B,K,D] shapes")
    if active.shape != predicted.shape[:2]:
        raise ValueError("V36 chunk activity does not align with plans")
    zero = predicted.float().sum() * 0.0
    nce_terms: list[torch.Tensor] = []
    cosine_terms: list[torch.Tensor] = []
    for slot in range(predicted.shape[1]):
        selected = active[:, slot] > 0
        if not bool(selected.any()):
            continue
        slot_predicted = predicted[selected, slot].float()
        slot_targets = targets[selected, slot].float()
        cosine_terms.append(
            (1.0 - F.cosine_similarity(slot_predicted, slot_targets, dim=-1)).mean()
        )
        if len(slot_predicted) >= 2:
            logits = slot_predicted @ slot_targets.T / temperature
            labels = torch.arange(len(slot_predicted), device=predicted.device)
            nce_terms.append(F.cross_entropy(logits, labels))
    chunk_nce = torch.stack(nce_terms).mean() if nce_terms else zero
    chunk_cosine = torch.stack(cosine_terms).mean() if cosine_terms else zero
    return chunk_nce, chunk_cosine, active.float().sum()


def visual_semantic_plan_loss(
    primary: VisualSemanticPlanOutput,
    alternate_view: VisualSemanticPlanOutput,
    targets: VisualSemanticTeacherTargets,
    *,
    weights: VisualSemanticPlanLossWeights = V36_LOSS_WEIGHTS,
) -> VisualSemanticPlanLoss:
    targets.validate()
    if primary.plans.shape != alternate_view.plans.shape:
        raise ValueError("V36 prompt views produce different plan shapes")
    if primary.plans.shape[:2] != (
        targets.global_plan.shape[0],
        V36_PLAN_SLOTS,
    ):
        raise ValueError("V36 student plans do not align with teacher targets")
    if primary.plans.shape[-1] != targets.global_plan.shape[-1]:
        raise ValueError("V36 student and teacher plan dimensions differ")

    predicted_global = primary.plans[:, 0].float()
    target_global = targets.global_plan.float()
    logits = predicted_global @ target_global.T / weights.temperature
    labels = torch.arange(len(predicted_global), device=predicted_global.device)
    global_nce = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    correct_values = F.cosine_similarity(predicted_global, target_global, dim=-1)
    global_cosine = (1.0 - correct_values).mean()

    hard_indices = _hard_negative_indices(
        predicted_global,
        target_global,
        teacher_ceiling=weights.negative_teacher_ceiling,
    )
    hard_targets = target_global[hard_indices]
    hard_values = F.cosine_similarity(predicted_global, hard_targets, dim=-1)
    margin = F.relu(weights.negative_margin - correct_values + hard_values).mean()

    chunk_nce, chunk_cosine, active_chunks = _active_chunk_losses(
        primary.plans[:, 1:],
        targets.chunk_plans,
        targets.chunk_active,
        temperature=weights.temperature,
    )
    view = (
        1.0
        - F.cosine_similarity(
            primary.plans[:, 0].float(),
            alternate_view.plans[:, 0].float(),
            dim=-1,
        )
    ).mean()
    length = F.smooth_l1_loss(primary.length.float(), targets.length.float())
    loss = (
        global_nce
        + weights.global_cosine * global_cosine
        + weights.margin * margin
        + weights.chunk_nce * chunk_nce
        + weights.chunk_cosine * chunk_cosine
        + weights.view * view
        + weights.length * length
    )
    batch_top1 = (logits.argmax(dim=1) == labels).float().mean()
    plan_variance = predicted_global.var(dim=0, unbiased=False).mean()
    return VisualSemanticPlanLoss(
        loss=loss,
        global_nce=global_nce,
        global_cosine=global_cosine,
        margin=margin,
        chunk_nce=chunk_nce,
        chunk_cosine=chunk_cosine,
        view=view,
        length=length,
        batch_top1=batch_top1,
        correct_cosine=correct_values.mean(),
        hard_negative_cosine=hard_values.mean(),
        plan_variance=plan_variance,
        active_chunks=active_chunks,
    )


def set_v36_stage_trainability(
    model: VisualSemanticPlanModel,
    stage: str,
) -> None:
    if stage not in {"plan-alignment", "semantic-adaptation"}:
        raise ValueError("V36 has no such trainability stage")
    model.requires_grad_(False)
    for module in (
        model.memory_projection,
        model.memory_norm,
        model.planner,
        model.plan_projection,
        model.length_head,
    ):
        module.requires_grad_(True)
    model.plan_queries.requires_grad_(True)
    model.plan_scale.requires_grad_(True)
    model.plan_bias.requires_grad_(True)
    if stage == "plan-alignment":
        model.freeze_reader()
    else:
        model.unfreeze_reader_final_blocks(2)


def _normalization_and_bias_ids(model: nn.Module) -> set[int]:
    identifiers: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias" or "norm" in module.__class__.__name__.lower():
                identifiers.add(id(parameter))
    identifiers.update((id(model.plan_scale), id(model.plan_bias)))
    return identifiers


def visual_semantic_plan_optimizer_groups(
    model: VisualSemanticPlanModel,
    *,
    head_learning_rate: float,
    reader_learning_rate: float,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    if min(head_learning_rate, reader_learning_rate, weight_decay) < 0.0:
        raise ValueError("V36 optimizer values must be non-negative")
    if max(head_learning_rate, reader_learning_rate) <= 0.0:
        raise ValueError("V36 optimizer requires a positive learning rate")
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


def set_v36_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    head: float,
    reader: float,
) -> None:
    rates = {"head": float(head), "reader": float(reader)}
    if min(rates.values()) < 0.0:
        raise ValueError("V36 optimizer learning rates must be non-negative")
    for group in optimizer.param_groups:
        role = str(group.get("role", ""))
        if role not in rates:
            raise ValueError("V36 optimizer group has no recognized role")
        group["lr"] = rates[role]


def v36_optimizer_receipt(
    model: VisualSemanticPlanModel,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimized: set[str] = set()
    rows: list[dict[str, Any]] = []
    for group in groups:
        parameters = list(group["params"])
        parameter_names = [names[id(parameter)] for parameter in parameters]
        optimized.update(parameter_names)
        rows.append(
            {
                "role": str(group["role"]),
                "decay": bool(group["decay"]),
                "parameters": sum(parameter.numel() for parameter in parameters),
                "tensors": len(parameters),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    return {
        "groups": rows,
        "optimized_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in optimized
        ),
        "optimized_parameter_names": sorted(optimized),
    }


class SelectiveExponentialMovingAverage:
    def __init__(
        self,
        model: nn.Module,
        parameter_names: Sequence[str],
        *,
        decay: float = 0.999,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("V36 EMA decay must be in [0,1)")
        source = dict(model.named_parameters())
        self.names = tuple(dict.fromkeys(parameter_names))
        missing = set(self.names).difference(source)
        if missing:
            raise ValueError(f"V36 EMA parameters are missing: {sorted(missing)}")
        self.decay = float(decay)
        self.shadow = {
            name: source[name].detach().float().clone() for name in self.names
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].lerp_(source[name].detach().float(), 1.0 - self.decay)

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
        if tuple(state["names"]) != self.names:
            raise ValueError("V36 EMA parameter names differ from the run")
        if float(state["decay"]) != self.decay:
            raise ValueError("V36 EMA decay differs from the run")
        shadow = state["shadow"]
        for name in self.names:
            value = shadow[name]
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"V36 EMA shape differs for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name]))


@dataclass
class VisualSemanticPlanTargetBank:
    identifiers: tuple[str, ...]
    global_plans: torch.Tensor
    chunk_plans: torch.Tensor
    chunk_active: torch.Tensor
    lengths: torch.Tensor
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.identifiers)
        if len(set(self.identifiers)) != count:
            raise ValueError("V36 target bank identifiers must be unique")
        if self.global_plans.ndim != 2 or self.global_plans.shape[0] != count:
            raise ValueError("V36 target bank global plans have an invalid shape")
        if self.chunk_plans.shape != (
            count,
            V36_CHUNKS,
            self.global_plans.shape[-1],
        ):
            raise ValueError("V36 target bank chunk plans have an invalid shape")
        if self.chunk_active.shape != (count, V36_CHUNKS):
            raise ValueError("V36 target bank chunk activity has an invalid shape")
        if self.lengths.shape != (count,):
            raise ValueError("V36 target bank lengths have an invalid shape")
        self._index = {identifier: index for index, identifier in enumerate(self.identifiers)}

    def lookup(
        self,
        identifiers: Sequence[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> VisualSemanticTeacherTargets:
        try:
            indices = torch.tensor(
                [self._index[identifier] for identifier in identifiers],
                dtype=torch.long,
            )
        except KeyError as error:
            raise KeyError(f"V36 target bank lacks {error.args[0]!r}") from error
        targets = VisualSemanticTeacherTargets(
            global_plan=self.global_plans[indices],
            chunk_plans=self.chunk_plans[indices],
            chunk_active=self.chunk_active[indices],
            length=self.lengths[indices],
        )
        return targets.to(device, dtype=dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": "visual-semantic-plan-target-bank-v36",
            "identifiers": list(self.identifiers),
            "global_plans": self.global_plans.detach().cpu(),
            "chunk_plans": self.chunk_plans.detach().cpu(),
            "chunk_active": self.chunk_active.detach().cpu(),
            "lengths": self.lengths.detach().cpu(),
            "receipt": self.receipt,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> VisualSemanticPlanTargetBank:
        if state.get("architecture") != "visual-semantic-plan-target-bank-v36":
            raise ValueError("target bank is not V36")
        return cls(
            identifiers=tuple(str(value) for value in state["identifiers"]),
            global_plans=state["global_plans"],
            chunk_plans=state["chunk_plans"],
            chunk_active=state["chunk_active"],
            lengths=state["lengths"],
            receipt=dict(state["receipt"]),
        )

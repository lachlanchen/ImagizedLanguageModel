from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .direct_visual_patch_lm import DirectVisualPatchLM, DirectVisualPatchOutput


@dataclass(frozen=True)
class DirectVisualPatchLossWeights:
    pixel: float = 1.0
    edge: float = 0.25
    ink: float = 0.25
    stop: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V33 loss weights must be non-negative")


V33_LOSS_WEIGHTS = DirectVisualPatchLossWeights()


@dataclass
class DirectVisualPatchLoss:
    loss: torch.Tensor
    pixel: torch.Tensor
    edge: torch.Tensor
    ink: torch.Tensor
    stop: torch.Tensor
    active_patches: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "pixel": float(self.pixel.detach()),
            "edge": float(self.edge.detach()),
            "ink": float(self.ink.detach()),
            "stop": float(self.stop.detach()),
            "active_patches": float(self.active_patches.detach()),
        }


def stage_cosine_learning_rate(
    update: int,
    *,
    peak: float,
    warmup: int,
    total: int,
    minimum_ratio: float = 0.10,
) -> float:
    if not 1 <= update <= total:
        raise ValueError("V33 learning-rate update must be inside the stage")
    if peak < 0.0 or not 0 <= warmup < total:
        raise ValueError("V33 learning-rate schedule is invalid")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("V33 minimum learning-rate ratio must be in [0,1]")
    if peak == 0.0:
        return 0.0
    if warmup and update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def strip_to_patches(pixels: torch.Tensor, patch_size: int) -> torch.Tensor:
    if pixels.ndim != 4 or pixels.shape[1:3] != (1, patch_size):
        raise ValueError("V33 strip must have shape [B,1,patch,patch*L]")
    if pixels.shape[-1] % patch_size:
        raise ValueError("V33 strip width does not divide into patches")
    return pixels.unfold(-1, patch_size, patch_size).permute(0, 3, 1, 2, 4)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError("V33 loss mask does not align with patch values")
    active = mask.float().sum()
    if not bool(active > 0):
        return values.float().sum() * 0.0
    expanded = mask.float()
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    elements = values[0, 0].numel() if values.ndim > 2 else 1
    return (values.float() * expanded).sum() / (active * elements)


def sobel_edges(patches: torch.Tensor) -> torch.Tensor:
    if patches.ndim != 5 or patches.shape[2] != 1:
        raise ValueError("V33 Sobel input must be [B,L,1,H,W]")
    kernel_x = patches.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    flat = patches.float().reshape(-1, 1, patches.shape[-2], patches.shape[-1])
    horizontal = F.conv2d(flat, kernel_x.float(), padding=1) / 8.0
    vertical = F.conv2d(flat, kernel_y.float(), padding=1) / 8.0
    return torch.cat((horizontal, vertical), dim=1).reshape(
        patches.shape[0],
        patches.shape[1],
        2,
        patches.shape[-2],
        patches.shape[-1],
    )


def direct_patch_losses(
    output: DirectVisualPatchOutput,
    pixels: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = strip_to_patches(pixels, output.patch_logits.shape[-1])
    if mode == "calibration":
        logits = output.patch_logits
        target = targets
        mask = active_mask
    elif mode == "causal":
        logits = output.patch_logits[:, :-1]
        target = targets[:, 1:]
        mask = active_mask[:, :-1]
    else:
        raise ValueError("V33 loss mode must be calibration or causal")
    if logits.shape != target.shape or logits.shape[:2] != mask.shape:
        raise ValueError("V33 patch-loss tensors do not align")
    pixel_values = F.binary_cross_entropy_with_logits(
        logits.float(),
        target.float(),
        reduction="none",
    )
    pixel = _masked_mean(pixel_values, mask)
    probability = logits.float().sigmoid()
    edge = _masked_mean((sobel_edges(probability) - sobel_edges(target)).abs(), mask)
    target_ink = (1.0 - target.float()).flatten(2)
    predicted_ink = (1.0 - probability).flatten(2)
    overlap = 2.0 * (predicted_ink * target_ink).sum(dim=-1)
    scale = predicted_ink.sum(dim=-1) + target_ink.sum(dim=-1)
    ink = _masked_mean(1.0 - (overlap + 1e-6) / (scale + 1e-6), mask)
    return pixel, edge, ink


def direct_visual_patch_loss(
    output: DirectVisualPatchOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    mode: str,
    weights: DirectVisualPatchLossWeights = V33_LOSS_WEIGHTS,
) -> DirectVisualPatchLoss:
    active_key = "reconstruction_mask" if mode == "calibration" else "next_patch_mask"
    active_mask = batch[active_key]
    pixel, edge, ink = direct_patch_losses(
        output,
        batch["pixels"],
        active_mask,
        mode=mode,
    )
    if mode == "causal":
        stop_values = F.binary_cross_entropy_with_logits(
            output.stop_logits.float(),
            batch["stop_targets"].float(),
            reduction="none",
        )
        stop = _masked_mean(stop_values, batch["stop_mask"])
    else:
        stop = output.stop_logits.float().sum() * 0.0
    loss = (
        weights.pixel * pixel
        + weights.edge * edge
        + weights.ink * ink
        + weights.stop * stop
    )
    return DirectVisualPatchLoss(
        loss=loss,
        pixel=pixel,
        edge=edge,
        ink=ink,
        stop=stop,
        active_patches=active_mask.float().sum(),
    )


def set_core_trainable(model: DirectVisualPatchLM, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = bool(trainable)


def _normalization_and_bias_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias" or "norm" in module.__class__.__name__.lower():
                ids.add(id(parameter))
    return ids


def direct_visual_patch_optimizer_groups(
    model: DirectVisualPatchLM,
    *,
    adapter_learning_rate: float,
    core_learning_rate: float,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    if adapter_learning_rate <= 0 or core_learning_rate < 0:
        raise ValueError("V33 optimizer learning rates are invalid")
    if weight_decay < 0:
        raise ValueError("V33 weight decay must be non-negative")
    no_decay = _normalization_and_bias_ids(model)
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("adapter", True): [],
        ("adapter", False): [],
        ("core", True): [],
        ("core", False): [],
    }
    for name, parameter in model.named_parameters():
        role = "core" if name.startswith("backbone.") else "adapter"
        groups[(role, id(parameter) not in no_decay)].append(parameter)
    result: list[dict[str, Any]] = []
    for role, decay in (("adapter", True), ("adapter", False), ("core", True), ("core", False)):
        parameters = groups[(role, decay)]
        if parameters:
            result.append(
                {
                    "params": parameters,
                    "lr": adapter_learning_rate if role == "adapter" else core_learning_rate,
                    "weight_decay": weight_decay if decay else 0.0,
                    "role": role,
                    "decay": decay,
                }
            )
    return result


def set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    adapter: float,
    core: float,
) -> None:
    for group in optimizer.param_groups:
        role = group.get("role")
        if role == "adapter":
            group["lr"] = float(adapter)
        elif role == "core":
            group["lr"] = float(core)
        else:
            raise ValueError("V33 optimizer group lacks a recognized role")


def optimizer_receipt(
    model: DirectVisualPatchLM,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    rows = []
    optimized: set[str] = set()
    for group in groups:
        parameters = list(group["params"])
        optimized.update(names[id(parameter)] for parameter in parameters)
        rows.append(
            {
                "role": group["role"],
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


@torch.no_grad()
def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("V33 EMA decay must be in [0,1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            shadow = self.shadow[name]
            shadow.lerp_(parameter.detach().float(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            parameter.copy_(self.shadow[name].to(parameter))

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {
                name: value.detach().cpu().clone()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("V33 EMA decay differs from checkpoint")
        source = state["shadow"]
        if set(source) != set(self.shadow):
            raise ValueError("V33 EMA parameter names differ from checkpoint")
        for name, value in source.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"V33 EMA tensor shape differs for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name]))

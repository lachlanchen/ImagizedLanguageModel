from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_glyph_flow import CausalGlyphFlowLM, CausalGlyphFlowOutput
from .continuous_glyph_codec_training import glyph_sobel_edges


@dataclass(frozen=True)
class CausalGlyphFlowLossWeights:
    flow: float = 1.0
    anchor: float = 1.0
    visual: float = 0.25
    stop: float = 0.10
    anchor_mse: float = 0.25
    visual_edge: float = 0.25
    visual_ink: float = 0.25

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V35 loss weights must be non-negative")


V35_LOSS_WEIGHTS = CausalGlyphFlowLossWeights()


@dataclass
class VisualInterfaceAlignmentLoss:
    loss: torch.Tensor
    mse: torch.Tensor
    cosine_distance: torch.Tensor
    cosine_similarity: torch.Tensor
    active_patches: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "mse": float(self.mse.detach()),
            "cosine_distance": float(self.cosine_distance.detach()),
            "cosine_similarity": float(self.cosine_similarity.detach()),
            "active_patches": float(self.active_patches.detach()),
        }


@dataclass
class CausalGlyphFlowLoss:
    loss: torch.Tensor
    flow: torch.Tensor
    anchor: torch.Tensor
    anchor_cosine: torch.Tensor
    anchor_mse: torch.Tensor
    visual: torch.Tensor
    visual_pixel: torch.Tensor
    visual_edge: torch.Tensor
    visual_ink: torch.Tensor
    stop: torch.Tensor
    active_patches: torch.Tensor
    density_patches: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "flow": float(self.flow.detach()),
            "anchor": float(self.anchor.detach()),
            "anchor_cosine": float(self.anchor_cosine.detach()),
            "anchor_mse": float(self.anchor_mse.detach()),
            "visual": float(self.visual.detach()),
            "visual_pixel": float(self.visual_pixel.detach()),
            "visual_edge": float(self.visual_edge.detach()),
            "visual_ink": float(self.visual_ink.detach()),
            "stop": float(self.stop.detach()),
            "active_patches": float(self.active_patches.detach()),
            "density_patches": float(self.density_patches),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError("V35 loss mask does not align with sequence values")
    active = mask.float().sum()
    if not bool(active > 0):
        return values.float().sum() * 0.0
    expanded = mask.float()
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    elements = values[0, 0].numel() if values.ndim > 2 else 1
    return (values.float() * expanded).sum() / (active * elements)


def visual_interface_alignment_loss(
    model: CausalGlyphFlowLM,
    pixels: torch.Tensor,
    patch_mask: torch.Tensor,
    teacher_projection: nn.Conv2d,
) -> VisualInterfaceAlignmentLoss:
    model._validate_inputs(pixels, patch_mask)
    if teacher_projection.kernel_size != (
        model.config.patch_size,
        model.config.patch_size,
    ):
        raise ValueError("V35 alignment teacher has the wrong patch shape")
    latents = model.encode_patches(pixels)
    predicted = model.input_adapter(latents)
    normalized = pixels.clamp(0, 1).mul(2.0).sub(1.0)
    with torch.no_grad():
        target = teacher_projection(normalized).squeeze(2).transpose(1, 2)
    if target.shape != predicted.shape:
        raise ValueError("V35 alignment teacher and adapter outputs do not align")
    squared = (predicted.float() - target.float()).square().mean(dim=-1)
    cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1)
    mse = _masked_mean(squared, patch_mask)
    cosine_similarity = _masked_mean(cosine, patch_mask)
    cosine_distance = 1.0 - cosine_similarity
    return VisualInterfaceAlignmentLoss(
        loss=mse + 0.25 * cosine_distance,
        mse=mse,
        cosine_distance=cosine_distance,
        cosine_similarity=cosine_similarity,
        active_patches=patch_mask.float().sum(),
    )


def _selected_active_indices(
    mask: torch.Tensor,
    *,
    maximum: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if mask.ndim != 2 or maximum < 1:
        raise ValueError("V35 density-position selection is invalid")
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if len(indices) < 1:
        raise ValueError("V35 causal loss requires an active next-patch target")
    if len(indices) <= maximum:
        return indices
    order = torch.randperm(
        len(indices),
        device=indices.device,
        generator=generator,
    )
    return indices[order[:maximum]]


def _visual_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: CausalGlyphFlowLossWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape != targets.shape or targets.ndim != 4:
        raise ValueError("V35 decoded visual targets do not align")
    pixel = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
    )
    probability = logits.float().sigmoid()
    edge = (
        glyph_sobel_edges(probability) - glyph_sobel_edges(targets.float())
    ).abs().mean()
    target_ink = (1.0 - targets.float()).flatten(1)
    predicted_ink = (1.0 - probability).flatten(1)
    overlap = 2.0 * (target_ink * predicted_ink).sum(dim=1)
    scale = target_ink.sum(dim=1) + predicted_ink.sum(dim=1)
    ink = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    visual = pixel + weights.visual_edge * edge + weights.visual_ink * ink
    return visual, pixel, edge, ink


def causal_glyph_flow_loss(
    model: CausalGlyphFlowLM,
    output: CausalGlyphFlowOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    generator: torch.Generator,
    maximum_density_patches: int = 128,
    weights: CausalGlyphFlowLossWeights = V35_LOSS_WEIGHTS,
) -> CausalGlyphFlowLoss:
    required = {"pixels", "next_patch_mask", "stop_targets", "stop_mask"}
    if not required.issubset(batch):
        raise ValueError("V35 causal batch lacks required visual tensors")
    pixels = batch["pixels"]
    targets = model.patchify(pixels)
    if output.latents.shape[:2] != targets.shape[:2]:
        raise ValueError("V35 causal output and target sequence do not align")
    active_mask = batch["next_patch_mask"][:, :-1]
    anchor = output.anchor_latents[:, :-1]
    target_latents = output.latents[:, 1:].detach()
    target_patches = targets[:, 1:]
    hidden = output.hidden_states[:, :-1]
    if active_mask.shape != anchor.shape[:2]:
        raise ValueError("V35 next-patch mask does not align with predictions")

    cosine_values = 1.0 - F.cosine_similarity(
        anchor.float(),
        target_latents.float(),
        dim=-1,
    )
    mse_values = (anchor.float() - target_latents.float()).square().mean(dim=-1)
    anchor_cosine = _masked_mean(cosine_values, active_mask)
    anchor_mse = _masked_mean(mse_values, active_mask)
    anchor_loss = anchor_cosine + weights.anchor_mse * anchor_mse

    selected = _selected_active_indices(
        active_mask,
        maximum=maximum_density_patches,
        generator=generator,
    )
    flat_target = target_latents.reshape(-1, model.config.latent_width)[selected]
    flat_hidden = hidden.reshape(-1, model.config.hidden_size)[selected]
    flat_anchor = anchor.reshape(-1, model.config.latent_width)[selected]
    flat_patches = target_patches.reshape(
        -1,
        1,
        model.config.patch_size,
        model.config.patch_size,
    )[selected]

    noise = torch.randn(
        flat_target.shape,
        device=flat_target.device,
        dtype=flat_target.dtype,
        generator=generator,
    )
    times = torch.rand(
        (len(flat_target),),
        device=flat_target.device,
        dtype=flat_target.dtype,
        generator=generator,
    )
    noisy = (1.0 - times[:, None]) * noise + times[:, None] * flat_target
    target_velocity = flat_target - noise
    predicted_velocity = model.flow_velocity(noisy, times, flat_hidden)
    flow = F.mse_loss(predicted_velocity.float(), target_velocity.float())

    decoded = model.decode_latents(flat_anchor)
    visual, visual_pixel, visual_edge, visual_ink = _visual_losses(
        decoded,
        flat_patches,
        weights=weights,
    )

    stop_values = F.binary_cross_entropy_with_logits(
        output.stop_logits.float(),
        batch["stop_targets"].float(),
        reduction="none",
    )
    stop = _masked_mean(stop_values, batch["stop_mask"])
    loss = (
        weights.flow * flow
        + weights.anchor * anchor_loss
        + weights.visual * visual
        + weights.stop * stop
    )
    return CausalGlyphFlowLoss(
        loss=loss,
        flow=flow,
        anchor=anchor_loss,
        anchor_cosine=anchor_cosine,
        anchor_mse=anchor_mse,
        visual=visual,
        visual_pixel=visual_pixel,
        visual_edge=visual_edge,
        visual_ink=visual_ink,
        stop=stop,
        active_patches=active_mask.float().sum(),
        density_patches=len(selected),
    )


def set_v35_stage_trainability(model: CausalGlyphFlowLM, stage: str) -> None:
    if stage not in {"visual-interface-alignment", "causal"}:
        raise ValueError("V35 has no such trainability stage")
    model.requires_grad_(False)
    model.codec.requires_grad_(False)
    if stage == "visual-interface-alignment":
        model.input_adapter.requires_grad_(True)
    else:
        model.backbone.requires_grad_(True)
        model.anchor_head.requires_grad_(True)
        model.flow_head.requires_grad_(True)
        model.stop_head.requires_grad_(True)


def _normalization_and_bias_ids(model: nn.Module) -> set[int]:
    identifiers: set[int] = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias" or "norm" in module.__class__.__name__.lower():
                identifiers.add(id(parameter))
    return identifiers


def causal_glyph_flow_optimizer_groups(
    model: CausalGlyphFlowLM,
    *,
    adapter_learning_rate: float,
    head_learning_rate: float,
    core_learning_rate: float,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    if min(adapter_learning_rate, head_learning_rate, core_learning_rate) < 0:
        raise ValueError("V35 optimizer learning rates must be non-negative")
    if max(adapter_learning_rate, head_learning_rate, core_learning_rate) <= 0:
        raise ValueError("V35 optimizer requires a positive learning rate")
    if weight_decay < 0:
        raise ValueError("V35 weight decay must be non-negative")
    no_decay = _normalization_and_bias_ids(model)
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("codec."):
            continue
        if name.startswith("input_adapter."):
            role = "adapter"
        elif name.startswith("backbone."):
            role = "core"
        else:
            role = "head"
        grouped.setdefault((role, id(parameter) not in no_decay), []).append(parameter)
    learning_rates = {
        "adapter": adapter_learning_rate,
        "head": head_learning_rate,
        "core": core_learning_rate,
    }
    result: list[dict[str, Any]] = []
    for role in ("adapter", "head", "core"):
        for decay in (True, False):
            parameters = grouped.get((role, decay), [])
            if parameters:
                result.append(
                    {
                        "params": parameters,
                        "lr": learning_rates[role],
                        "weight_decay": weight_decay if decay else 0.0,
                        "role": role,
                        "decay": decay,
                    }
                )
    return result


def set_v35_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    adapter: float,
    head: float,
    core: float,
) -> None:
    rates = {"adapter": adapter, "head": head, "core": core}
    if min(rates.values()) < 0:
        raise ValueError("V35 optimizer learning rates must be non-negative")
    for group in optimizer.param_groups:
        role = str(group.get("role", ""))
        if role not in rates:
            raise ValueError("V35 optimizer group lacks a recognized role")
        group["lr"] = float(rates[role])


def v35_optimizer_receipt(
    model: CausalGlyphFlowLM,
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
    codec_names = {name for name, _ in model.codec.named_parameters(prefix="codec")}
    return {
        "groups": rows,
        "optimized_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in optimized
        ),
        "optimized_parameter_names": sorted(optimized),
        "codec_parameter_names_optimized": sorted(codec_names.intersection(optimized)),
    }


from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ink_jepa import InkJEPA


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class FovealWriterConfig:
    fovea_size: int = 32
    condition_dim: int = 256
    base_channels: int = 64
    context_dim: int = 256
    condition_dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 4:
            raise ValueError("fovea_size must be a multiple of four and at least 16")
        if self.base_channels < 16 or self.context_dim < 64:
            raise ValueError("foveal writer is underspecified")


class TimeField(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)
        self.output = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        phase = time.float()[:, None] * frequencies[None] * 1_000.0
        field = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if field.shape[-1] < self.dimension:
            field = F.pad(field, (0, self.dimension - field.shape[-1]))
        return self.output(field.to(time.dtype))


class ConditionedInkBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, context_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(incoming), incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.context = nn.Linear(context_dim, outgoing * 2)
        self.norm2 = nn.GroupNorm(_groups(outgoing), outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, shift = self.context(F.silu(context)).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(image)


class FovealInkFlow(nn.Module):
    """Generate one continuous writing fovea from a continuous retinal state."""

    def __init__(self, config: FovealWriterConfig):
        super().__init__()
        self.config = config
        base = config.base_channels
        self.time = TimeField(config.context_dim)
        self.condition = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.context_dim * 2),
            nn.SiLU(),
            nn.Linear(config.context_dim * 2, config.context_dim),
        )
        self.input = nn.Conv2d(2, base, 3, padding=1)
        self.level0 = ConditionedInkBlock(base, base, config.context_dim)
        self.down0 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.level1 = ConditionedInkBlock(base * 2, base * 2, config.context_dim)
        self.down1 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid0 = ConditionedInkBlock(base * 4, base * 4, config.context_dim)
        self.mid1 = ConditionedInkBlock(base * 4, base * 4, config.context_dim)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.merge1 = ConditionedInkBlock(base * 4, base * 2, config.context_dim)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.merge0 = ConditionedInkBlock(base * 2, base, config.context_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(base), base),
            nn.SiLU(),
            nn.Conv2d(base, 1, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        ink_plan: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (1, self.config.fovea_size, self.config.fovea_size)
        if state.ndim != 4 or tuple(state.shape[1:]) != expected:
            raise ValueError(f"expected foveal state [batch, {expected}]")
        if condition.ndim != 2 or condition.shape[1] != self.config.condition_dim:
            raise ValueError("condition does not match the configured retinal state dimension")
        if ink_plan.shape != state.shape:
            raise ValueError("ink_plan must have the same continuous foveal shape as state")
        if condition_present is None:
            condition_present = torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
        condition_present = condition_present.to(state).reshape(-1, 1)
        context = self.time(time) + self.condition(condition * condition_present)
        visible_plan = ink_plan * condition_present[:, :, None, None]
        level0 = self.level0(self.input(torch.cat((state, visible_plan), dim=1)), context)
        level1 = self.level1(self.down0(level0), context)
        hidden = self.mid1(self.mid0(self.down1(level1), context), context)
        hidden = self.up1(hidden)
        hidden = self.merge1(torch.cat((hidden, level1), dim=1), context)
        hidden = self.up0(hidden)
        hidden = self.merge0(torch.cat((hidden, level0), dim=1), context)
        return self.output(hidden)


def retinal_foveal_prediction(
    foundation: InkJEPA,
    context_image: torch.Tensor,
    hidden_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a blank future location from image context without symbolic IDs."""

    online = foundation.online_encoder(context_image, hidden_mask=hidden_mask)
    prediction = foundation.predictor(online["field"], hidden_mask)
    predicted = prediction["local"]
    weights = target_mask.to(predicted.dtype)
    local = (predicted * weights[..., None]).sum(dim=(1, 2))
    local = local / weights.sum(dim=(1, 2), keepdim=False)[:, None].clamp_min(1.0)
    page = foundation.page_predictor(online["page"])
    patch_logits = prediction["ink_logits"].permute(0, 3, 1, 2)
    ink_plan = F.pixel_shuffle(patch_logits, foundation.config.patch_size)
    return torch.cat((local, page), dim=-1), ink_plan


def flow_training_state(
    target: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = target.shape[0]
    time = torch.rand(batch, device=target.device, dtype=target.dtype, generator=generator)
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    expanded = time[:, None, None, None]
    state = (1.0 - expanded) * target + expanded * noise
    velocity = noise - target
    return state, velocity, time, noise


def foveal_flow_loss(
    prediction: torch.Tensor,
    velocity: torch.Tensor,
    state: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    *,
    endpoint_weight: float = 0.10,
    stroke_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ink = ((target + 1.0) * 0.5).clamp(0, 1)
    weights = 1.0 + stroke_weight * ink
    flow = (weights * (prediction.float() - velocity.float()).square()).mean()
    endpoint = state.float() - time.float()[:, None, None, None] * prediction.float()
    endpoint_l1 = (weights * (endpoint - target.float()).abs()).mean()
    total = flow + endpoint_weight * endpoint_l1
    with torch.no_grad():
        endpoint_ink = (endpoint + 1.0) * 0.5
        binary = endpoint_ink >= 0.5
        target_binary = ink >= 0.5
        true_positive = (binary & target_binary).sum().float()
        f1 = 2.0 * true_positive / (binary.sum() + target_binary.sum()).clamp_min(1)
    return total, {
        "flow_mse": flow.detach(),
        "endpoint_l1": endpoint_l1.detach(),
        "endpoint_ink_f1": f1.detach(),
        "target_ink_fraction": ink.mean().detach(),
    }


@torch.no_grad()
def sample_foveal_ink(
    model: FovealInkFlow,
    condition: torch.Tensor,
    ink_plan: torch.Tensor,
    *,
    steps: int = 8,
    guidance_scale: float = 1.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if ink_plan.shape != (
        condition.shape[0],
        1,
        model.config.fovea_size,
        model.config.fovea_size,
    ):
        raise ValueError("sampling ink plan has the wrong foveal shape")
    state = initial_noise
    if state is None:
        state = torch.randn(
            condition.shape[0],
            1,
            model.config.fovea_size,
            model.config.fovea_size,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
    times = torch.linspace(1.0, 0.0, steps + 1, device=state.device, dtype=state.dtype)
    present = torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
    absent = torch.zeros_like(present)

    def velocity(current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if guidance_scale == 1.0:
            return model(current, value, condition, ink_plan, condition_present=present)
        both_state = torch.cat((current, current), dim=0)
        both_time = torch.cat((value, value), dim=0)
        both_condition = torch.cat((condition, condition), dim=0)
        both_plan = torch.cat((ink_plan, ink_plan), dim=0)
        unconditioned, conditioned = model(
            both_state,
            both_time,
            both_condition,
            both_plan,
            condition_present=torch.cat((absent, present)),
        ).chunk(2)
        return unconditioned + guidance_scale * (conditioned - unconditioned)

    for index in range(steps):
        current_time = times[index].expand(state.shape[0])
        next_time = times[index + 1].expand(state.shape[0])
        delta = next_time[0] - current_time[0]
        first = velocity(state, current_time)
        proposal = state + delta * first
        if index == steps - 1:
            state = proposal
        else:
            second = velocity(proposal, next_time)
            state = state + delta * 0.5 * (first + second)
    return state.clamp(-1, 1)


def foveal_writer_config_payload(config: FovealWriterConfig) -> dict[str, Any]:
    return asdict(config)


def foveal_writer_config_from_payload(payload: dict[str, Any]) -> FovealWriterConfig:
    return FovealWriterConfig(**payload)

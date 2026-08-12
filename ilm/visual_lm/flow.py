from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = time.float()[:, None] * frequencies[None, :] * 1000.0
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return self.mlp(embedding.to(time.dtype))


class FlowResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels * 2)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(x)


class FlowAttention(nn.Module):
    def __init__(self, channels: int, heads: int, window_size: int = 8):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("FlowAttention channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = window_size
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window = self.window_size
        pad_height = (-height) % window
        pad_width = (-width) % window
        normalized = self.norm(x)
        if pad_height or pad_width:
            normalized = F.pad(normalized, (0, pad_width, 0, pad_height), mode="replicate")
        padded_height, padded_width = normalized.shape[-2:]
        rows = padded_height // window
        columns = padded_width // window
        q, k, v = self.qkv(normalized).chunk(3, dim=1)

        def to_windows(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(
                batch,
                self.heads,
                self.head_dim,
                rows,
                window,
                columns,
                window,
            )
            tensor = tensor.permute(0, 3, 5, 1, 4, 6, 2)
            return tensor.reshape(batch * rows * columns, self.heads, window * window, self.head_dim)

        attended = F.scaled_dot_product_attention(to_windows(q), to_windows(k), to_windows(v))
        attended = attended.reshape(
            batch,
            rows,
            columns,
            self.heads,
            window,
            window,
            self.head_dim,
        )
        attended = attended.permute(0, 3, 6, 1, 4, 2, 5).reshape(
            batch, channels, padded_height, padded_width
        )
        attended = attended[:, :, :height, :width]
        return x + self.proj(attended)


@dataclass(frozen=True)
class VisualFlowConfig:
    latent_channels: int = 8
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    blocks_per_level: int = 2
    time_channels: int = 256
    attention_heads: int = 4
    condition_dropout: float = 0.10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "VisualFlowConfig":
        values = dict(values)
        if "channel_multipliers" in values:
            values["channel_multipliers"] = tuple(values["channel_multipliers"])
        return cls(**values)


class ConditionalVisualFlow(nn.Module):
    """Conditional latent rectified-flow network for image-to-image language.

    Both the condition and generated state are continuous visual latents. The
    condition is injected at the input and propagated through every U-Net scale;
    a global visual summary also modulates every residual block.
    """

    def __init__(self, config: VisualFlowConfig | None = None):
        super().__init__()
        self.config = config or VisualFlowConfig()
        cfg = self.config
        widths = [cfg.base_channels * mult for mult in cfg.channel_multipliers]
        model_input_channels = cfg.latent_channels * 2 + 1

        self.time_embedding = TimeEmbedding(cfg.time_channels)
        self.condition_global = nn.Sequential(
            nn.Linear(cfg.latent_channels, cfg.time_channels),
            nn.SiLU(),
            nn.Linear(cfg.time_channels, cfg.time_channels),
        )
        self.input = nn.Conv2d(model_input_channels, widths[0], 3, padding=1)

        self.down_levels = nn.ModuleList()
        channels = widths[0]
        for index, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(cfg.blocks_per_level):
                blocks.append(FlowResidualBlock(channels, width, cfg.time_channels))
                channels = width
            self.down_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "downsample": (
                            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                            if index < len(widths) - 1
                            else nn.Identity()
                        ),
                    }
                )
            )

        self.mid1 = FlowResidualBlock(channels, channels, cfg.time_channels)
        self.mid_attention = FlowAttention(channels, cfg.attention_heads)
        self.mid2 = FlowResidualBlock(channels, channels, cfg.time_channels)

        self.up_levels = nn.ModuleList()
        reversed_widths = list(reversed(widths))
        for index, skip_channels in enumerate(reversed_widths):
            upsample = (
                nn.Identity()
                if index == 0
                else nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv2d(channels, skip_channels, 3, padding=1),
                )
            )
            if index > 0:
                channels = skip_channels
            blocks = nn.ModuleList(
                [FlowResidualBlock(channels + skip_channels, skip_channels, cfg.time_channels)]
            )
            channels = skip_channels
            for _ in range(cfg.blocks_per_level - 1):
                blocks.append(FlowResidualBlock(channels, channels, cfg.time_channels))
            self.up_levels.append(nn.ModuleDict({"upsample": upsample, "blocks": blocks}))

        self.output_norm = nn.GroupNorm(_groups(channels), channels)
        self.output = nn.Conv2d(channels, cfg.latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition.shape != state.shape:
            raise ValueError(
                f"condition shape {tuple(condition.shape)} must match state shape {tuple(state.shape)}"
            )
        batch = state.shape[0]
        if condition_present is None:
            condition_present = torch.ones(batch, device=state.device, dtype=state.dtype)
        condition_present = condition_present.to(device=state.device, dtype=state.dtype).reshape(batch, 1, 1, 1)
        masked_condition = condition * condition_present
        presence_plane = condition_present.expand(batch, 1, state.shape[-2], state.shape[-1])

        time_context = self.time_embedding(time)
        global_condition = masked_condition.mean(dim=(-2, -1))
        time_context = time_context + self.condition_global(global_condition)

        hidden = self.input(torch.cat((state, masked_condition, presence_plane), dim=1))
        skips: list[torch.Tensor] = []
        for level in self.down_levels:
            for block in level["blocks"]:
                hidden = block(hidden, time_context)
            skips.append(hidden)
            hidden = level["downsample"](hidden)

        hidden = self.mid1(hidden, time_context)
        hidden = self.mid_attention(hidden)
        hidden = self.mid2(hidden, time_context)

        for level in self.up_levels:
            hidden = level["upsample"](hidden)
            skip = skips.pop()
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat((hidden, skip), dim=1)
            for block in level["blocks"]:
                hidden = block(hidden, time_context)
        return self.output(F.silu(self.output_norm(hidden)))


def flow_training_pair(
    data: torch.Tensor,
    *,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a rectified-flow state and target velocity.

    The path is z(t) = (1-t) z_data + t z_noise. Sampling integrates the
    learned velocity from t=1 (noise) to t=0 (data).
    """

    batch = data.shape[0]
    if time is None:
        time = torch.rand(batch, device=data.device, dtype=data.dtype, generator=generator)
    if noise is None:
        noise = torch.randn(data.shape, device=data.device, dtype=data.dtype, generator=generator)
    expanded_time = time.reshape(batch, *([1] * (data.ndim - 1)))
    state = (1.0 - expanded_time) * data + expanded_time * noise
    velocity = noise - data
    return state, velocity, time, noise


def condition_keep_mask(
    batch_size: int,
    dropout: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 <= dropout < 1.0:
        raise ValueError("condition dropout must be in [0, 1)")
    return (
        torch.rand(batch_size, device=device, generator=generator) >= dropout
    ).to(dtype=dtype)


@torch.no_grad()
def sample_heun(
    model: ConditionalVisualFlow,
    condition: torch.Tensor,
    *,
    steps: int = 12,
    guidance_scale: float = 2.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    clamp: float | None = 6.0,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    state = (
        initial_noise.clone()
        if initial_noise is not None
        else torch.randn(
            condition.shape,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
    )
    batch = state.shape[0]
    cond_present = torch.ones(batch, device=state.device, dtype=state.dtype)
    uncond_present = torch.zeros_like(cond_present)
    times = torch.linspace(1.0, 0.0, steps + 1, device=state.device, dtype=state.dtype)

    def guided_velocity(current: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        if guidance_scale == 1.0:
            return model(current, current_time, condition, condition_present=cond_present)
        both_state = torch.cat((current, current), dim=0)
        both_time = torch.cat((current_time, current_time), dim=0)
        both_condition = torch.cat((condition, condition), dim=0)
        presence = torch.cat((uncond_present, cond_present), dim=0)
        unconditioned, conditioned = model(
            both_state,
            both_time,
            both_condition,
            condition_present=presence,
        ).chunk(2, dim=0)
        return unconditioned + guidance_scale * (conditioned - unconditioned)

    for index in range(steps):
        current_t = times[index].expand(batch)
        next_t = times[index + 1].expand(batch)
        delta = next_t[0] - current_t[0]
        velocity = guided_velocity(state, current_t)
        predicted = state + delta * velocity
        if index == steps - 1:
            state = predicted
        else:
            next_velocity = guided_velocity(predicted, next_t)
            state = state + delta * 0.5 * (velocity + next_velocity)
        if clamp is not None:
            state = state.clamp(-clamp, clamp)
    return state


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        source_parameters = dict(source.named_parameters())
        for name, target in self.model.named_parameters():
            target.lerp_(source_parameters[name].detach(), 1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, target in self.model.named_buffers():
            target.copy_(source_buffers[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

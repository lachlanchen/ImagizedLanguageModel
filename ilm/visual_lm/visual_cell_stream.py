from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .saccade_lm import FovealRetina, VisualSaccadeConfig


@dataclass(frozen=True)
class VisualCellStreamConfig:
    cell_size: int = 32
    maximum_cells: int = 64
    visual_dim: int = 192
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    retina_base_channels: int = 64
    writer_base_channels: int = 48
    time_dim: int = 64
    context_noise_maximum: float = 0.35
    condition_dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("visual-cell stream currently requires 32x32 images")
        if self.maximum_cells < 4:
            raise ValueError("maximum_cells must contain a causal visual history")
        if self.visual_dim < 64 or self.model_dim < 128:
            raise ValueError("visual-cell state is underspecified")
        if self.layers < 1 or self.heads < 1 or self.model_dim % self.heads:
            raise ValueError("causal layer/head configuration is invalid")
        if self.mlp_ratio < 2.0:
            raise ValueError("mlp_ratio must be at least two")
        if self.retina_base_channels < 8 or self.writer_base_channels < 8:
            raise ValueError("retina and writer channels must be at least eight")
        if self.time_dim < 16 or self.time_dim % 2:
            raise ValueError("time_dim must be an even integer of at least 16")
        if not 0.0 <= self.context_noise_maximum <= 1.0:
            raise ValueError("context_noise_maximum must be in [0, 1]")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = float(epsilon)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        ).to(value.dtype)
        return normalized * self.weight.to(value.dtype)


def _apply_rotary(value: torch.Tensor) -> torch.Tensor:
    """Apply parameter-free relative reading position to [B, heads, T, D]."""

    length = value.shape[-2]
    dimension = value.shape[-1]
    rotary_dimension = dimension - dimension % 2
    if rotary_dimension == 0:
        return value
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(
            0,
            rotary_dimension,
            2,
            device=value.device,
            dtype=torch.float32,
        )
        / rotary_dimension
    )
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos().to(value.dtype)[None, None]
    sine = angles.sin().to(value.dtype)[None, None]
    active = value[..., :rotary_dimension]
    first = active[..., 0::2]
    second = active[..., 1::2]
    rotated = torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    ).flatten(-2)
    return torch.cat((rotated, value[..., rotary_dimension:]), dim=-1)


class CausalVisualAttention(nn.Module):
    def __init__(self, config: VisualCellStreamConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.dropout = float(config.dropout)
        self.qkv = nn.Linear(config.model_dim, config.model_dim * 3, bias=False)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = state.shape
        qkv = self.qkv(state).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = _apply_rotary(query.transpose(1, 2))
        key = _apply_rotary(key.transpose(1, 2))
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, dimension))


class CausalVisualBlock(nn.Module):
    def __init__(self, config: VisualCellStreamConfig) -> None:
        super().__init__()
        hidden = int(config.model_dim * config.mlp_ratio)
        self.attention_norm = RMSNorm(config.model_dim)
        self.attention = CausalVisualAttention(config)
        self.mlp_norm = RMSNorm(config.model_dim)
        self.gate = nn.Linear(config.model_dim, hidden, bias=False)
        self.value = nn.Linear(config.model_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.model_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state + self.dropout(self.attention(self.attention_norm(state)))
        normalized = self.mlp_norm(state)
        update = self.down(F.silu(self.gate(normalized)) * self.value(normalized))
        return state + self.dropout(update)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, output_dim: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(0, time_dim, 2, dtype=torch.float32)
            / time_dim
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.network = nn.Sequential(
            nn.Linear(time_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.float()[..., None] * self.frequencies * (2.0 * math.pi)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.network(embedding.to(self.network[0].weight.dtype))


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class WriterResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return image + hidden


class ContinuousCellFlowWriter(nn.Module):
    """Draw one continuous cell from noise and continuous visual context."""

    def __init__(self, config: VisualCellStreamConfig) -> None:
        super().__init__()
        base = config.writer_base_channels
        self.config = config
        self.time = ContinuousTimeEmbedding(config.time_dim, config.model_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1),
            WriterResidualBlock(base),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            WriterResidualBlock(base * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            WriterResidualBlock(base * 4),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1),
            WriterResidualBlock(base * 4),
        )
        condition_dim = config.model_dim * 2 + config.visual_dim
        self.condition = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, base * 8),
        )
        self.bottleneck = nn.Sequential(
            WriterResidualBlock(base * 4),
            WriterResidualBlock(base * 4),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 4, 4, stride=2, padding=1)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1),
            WriterResidualBlock(base * 4),
        )
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            WriterResidualBlock(base * 2),
        )
        self.up0 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.fuse0 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            WriterResidualBlock(base),
        )
        self.output = nn.Conv2d(base, 1, 3, padding=1)

    def forward(
        self,
        noisy_cell: torch.Tensor,
        time: torch.Tensor,
        context_state: torch.Tensor,
        proposed_visual: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_cell.ndim != 4 or tuple(noisy_cell.shape[1:]) != (1, 32, 32):
            raise ValueError("writer noisy cell must have shape [N,1,32,32]")
        count = noisy_cell.shape[0]
        if time.shape != (count,):
            raise ValueError("writer time must have shape [N]")
        if context_state.shape != (count, self.config.model_dim):
            raise ValueError("writer context state has the wrong shape")
        if proposed_visual.shape != (count, self.config.visual_dim):
            raise ValueError("writer visual proposal has the wrong shape")
        if condition_present is None:
            condition_present = torch.ones(
                (count, 1), device=noisy_cell.device, dtype=noisy_cell.dtype
            )
        if condition_present.shape != (count, 1):
            raise ValueError("condition_present must have shape [N,1]")
        time_state = self.time(time).to(context_state.dtype)
        condition = torch.cat(
            (
                context_state * condition_present,
                proposed_visual * condition_present,
                time_state,
            ),
            dim=-1,
        )
        scale_shift = self.condition(condition)
        scale, shift = scale_shift.chunk(2, dim=-1)

        level0 = self.stem(noisy_cell)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        hidden = self.down3(level2)
        hidden = hidden * (1.0 + 0.1 * scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.bottleneck(hidden)
        hidden = self.fuse2(torch.cat((self.up2(hidden), level2), dim=1))
        hidden = self.fuse1(torch.cat((self.up1(hidden), level1), dim=1))
        hidden = self.fuse0(torch.cat((self.up0(hidden), level0), dim=1))
        return self.output(hidden)


class VisualCellStreamModel(nn.Module):
    """Causal language and pixel generation over visible Chinese cell images."""

    def __init__(
        self,
        config: VisualCellStreamConfig,
        *,
        freeze_retina: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        retina_config = VisualSaccadeConfig(
            fovea_size=config.cell_size,
            visual_dim=config.visual_dim,
            state_dim=config.model_dim,
            state_layers=1,
            retina_base_channels=config.retina_base_channels,
            ink_base_channels=max(16, config.writer_base_channels),
            dropout=0.0,
        )
        self.online_retina = FovealRetina(retina_config)
        self.target_retina = copy.deepcopy(self.online_retina)
        self.target_retina.requires_grad_(False).eval()
        self.freeze_retina = bool(freeze_retina)
        if self.freeze_retina:
            self.online_retina.requires_grad_(False).eval()
        self.context_time = ContinuousTimeEmbedding(config.time_dim, config.model_dim)
        self.visual_input = nn.Linear(config.visual_dim, config.model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.output_norm = RMSNorm(config.model_dim)
        self.proposal = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.visual_dim),
        )
        self.writer = ContinuousCellFlowWriter(config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))
        self._initialize_language_layers()

    def _initialize_language_layers(self) -> None:
        modules: list[nn.Module] = [
            self.context_time,
            self.visual_input,
            self.blocks,
            self.output_norm,
            self.proposal,
        ]
        for root in modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * max(1, self.config.layers))
        for block in self.blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def train(self, mode: bool = True) -> "VisualCellStreamModel":
        super().train(mode)
        self.target_retina.eval()
        if self.freeze_retina:
            self.online_retina.eval()
        return self

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def _validate_cells(self, cells: torch.Tensor, *, maximum: bool = True) -> None:
        if not torch.is_floating_point(cells):
            raise TypeError("visual-cell model accepts floating image tensors only")
        if cells.ndim != 5 or tuple(cells.shape[2:]) != (1, 32, 32):
            raise ValueError("visual cells must have shape [B,T,1,32,32]")
        if cells.shape[1] < 1:
            raise ValueError("visual-cell stream cannot be empty")
        if maximum and cells.shape[1] > self.config.maximum_cells:
            raise ValueError("visual-cell stream exceeds configured context")

    def encode_cells(self, cells: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        self._validate_cells(cells, maximum=False)
        batch, length = cells.shape[:2]
        retina = self.target_retina if target else self.online_retina
        encoded = retina(cells.reshape(batch * length, 1, 32, 32).clamp(0, 1))
        return F.normalize(encoded.float(), dim=-1).reshape(batch, length, -1)

    def language(
        self,
        context: torch.Tensor,
        *,
        context_noise_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_cells(context)
        batch, length = context.shape[:2]
        if context_noise_time is None:
            context_noise_time = torch.zeros(
                (batch, length), device=context.device, dtype=context.dtype
            )
        if context_noise_time.shape != (batch, length):
            raise ValueError("context noise time must have shape [B,T]")
        visual = self.encode_cells(context)
        state = self.visual_input(visual.to(self.visual_input.weight.dtype))
        state = state + self.context_time(context_noise_time).to(state.dtype)
        for block in self.blocks:
            state = block(state)
        state = self.output_norm(state)
        proposal = F.normalize(self.proposal(state).float(), dim=-1)
        return {
            "context_visual": visual,
            "context_state": state,
            "proposed_visual": proposal,
        }

    def forward_language(
        self,
        context: torch.Tensor,
        reference_target: torch.Tensor,
        *,
        context_noise_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_cells(reference_target)
        if reference_target.shape[:2] != context.shape[:2]:
            raise ValueError("reference target must align with context transitions")
        output = self.language(context, context_noise_time=context_noise_time)
        with torch.no_grad():
            output["target_visual"] = self.encode_cells(
                reference_target, target=True
            )
        return output

    @torch.no_grad()
    def update_target_retina(self, momentum: float) -> None:
        if self.freeze_retina:
            return
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0,1]")
        online = dict(self.online_retina.named_parameters())
        for name, target in self.target_retina.named_parameters():
            target.lerp_(online[name], 1.0 - momentum)
        online_buffers = dict(self.online_retina.named_buffers())
        for name, target in self.target_retina.named_buffers():
            target.copy_(online_buffers[name])

    def flow_velocity(
        self,
        noisy_cell: torch.Tensor,
        time: torch.Tensor,
        context_state: torch.Tensor,
        proposed_visual: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.writer(
            noisy_cell,
            time,
            context_state,
            proposed_visual,
            condition_present=condition_present,
        )

    @torch.no_grad()
    def score_image_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluator helper; candidate banks are never used by generate()."""

        output = self.language(context)
        proposal = output["proposed_visual"][:, -1]
        if candidates.ndim == 4:
            candidates = candidates[:, None]
        if candidates.ndim != 5 or tuple(candidates.shape[2:]) != (1, 32, 32):
            raise ValueError("candidates must be [N,V,1,32,32]")
        identities, views = candidates.shape[:2]
        visual = self.encode_cells(candidates, target=True).reshape(
            identities, views, self.config.visual_dim
        )
        return self.contrastive_scale * torch.einsum(
            "bd,nvd->bnv", proposal, visual
        ).amax(dim=2)

    @torch.no_grad()
    def sample_from_state(
        self,
        context_state: torch.Tensor,
        proposed_visual: torch.Tensor,
        *,
        candidates: int,
        flow_steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if candidates < 1 or flow_steps < 1:
            raise ValueError("candidates and flow_steps must be positive")
        batch = context_state.shape[0]
        state = context_state[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, -1
        )
        proposal = proposed_visual[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, -1
        )
        image = torch.randn(
            (batch * candidates, 1, 32, 32),
            device=context_state.device,
            dtype=context_state.dtype,
            generator=generator,
        )
        step_size = 1.0 / flow_steps
        for step in range(flow_steps):
            time_value = 1.0 - step / flow_steps
            time = torch.full(
                (batch * candidates,),
                time_value,
                device=image.device,
                dtype=image.dtype,
            )
            velocity = self.flow_velocity(image, time, state, proposal)
            image = image - step_size * velocity
        candidate_images = image.clamp(0, 1).reshape(
            batch, candidates, 1, 32, 32
        )
        reread = self.encode_cells(candidate_images, target=True)
        scores = torch.einsum("bkd,bd->bk", reread, proposed_visual)
        selected_indices = scores.argmax(dim=1)
        selected = candidate_images[
            torch.arange(batch, device=image.device), selected_indices
        ]
        return selected, {
            "candidate_images": candidate_images,
            "candidate_visual": reread,
            "candidate_scores": scores,
            "selected_indices": selected_indices,
            "selected_visual": reread[
                torch.arange(batch, device=image.device), selected_indices
            ],
        }

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        *,
        new_cells: int,
        candidates: int = 4,
        flow_steps: int = 12,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_cells(prefix, maximum=False)
        if new_cells < 1:
            raise ValueError("new_cells must be positive")
        sequence = prefix
        generated: list[torch.Tensor] = []
        selected_visual: list[torch.Tensor] = []
        for _ in range(new_cells):
            context = sequence[:, -self.config.maximum_cells :]
            language = self.language(context)
            proposal = language["proposed_visual"][:, -1]
            next_cell, trace = self.sample_from_state(
                language["context_state"][:, -1],
                proposal,
                candidates=candidates,
                flow_steps=flow_steps,
                generator=generator,
            )
            generated.append(next_cell)
            selected_visual.append(trace["selected_visual"])
            sequence = torch.cat((sequence, next_cell[:, None]), dim=1)
        return sequence, {
            "generated_cells": torch.stack(generated, dim=1),
            "reread_visual": torch.stack(selected_visual, dim=1),
            "reread_generated_pixels": torch.tensor(
                True, device=sequence.device
            ),
        }


def _multi_positive_contrastive_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: torch.Tensor,
    duplicate_similarity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = F.normalize(predicted.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = scale.float() * predicted @ target.transpose(0, 1)
    target_similarity = target @ target.transpose(0, 1)
    positives = target_similarity >= duplicate_similarity
    positives.fill_diagonal_(True)
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    loss = -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
    predicted_index = logits.argmax(dim=1)
    accuracy = positives[
        torch.arange(predicted.shape[0], device=predicted.device), predicted_index
    ].float().mean()
    return loss, accuracy


def visual_cell_language_loss(
    output: dict[str, torch.Tensor],
    *,
    contrastive_scale: torch.Tensor,
    contrastive_weight: float = 0.50,
    duplicate_similarity: float = 0.985,
    maximum_contrastive: int = 512,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    proposal = F.normalize(output["proposed_visual"].float(), dim=-1)
    target = F.normalize(output["target_visual"].float().detach(), dim=-1)
    cosine = (proposal * target).sum(dim=-1)
    visual = (1.0 - cosine).mean()
    flat_proposal = proposal.flatten(0, 1)
    flat_target = target.flatten(0, 1)
    if flat_proposal.shape[0] > maximum_contrastive:
        indices = torch.randperm(
            flat_proposal.shape[0], device=flat_proposal.device, generator=generator
        )[:maximum_contrastive]
        flat_proposal = flat_proposal[indices]
        flat_target = flat_target[indices]
    contrastive, in_batch_accuracy = _multi_positive_contrastive_loss(
        flat_proposal,
        flat_target,
        scale=contrastive_scale,
        duplicate_similarity=duplicate_similarity,
    )
    total = visual + contrastive_weight * contrastive
    return total, {
        "loss": total.detach(),
        "visual_loss": visual.detach(),
        "contrastive_loss": contrastive.detach(),
        "target_cosine": cosine.mean().detach(),
        "in_batch_visual_accuracy": in_batch_accuracy.detach(),
        "contrastive_scale": contrastive_scale.detach(),
    }


def visual_cell_flow_loss(
    model: VisualCellStreamModel,
    context_state: torch.Tensor,
    proposed_visual: torch.Tensor,
    target_cell: torch.Tensor,
    reference_target_cell: torch.Tensor,
    *,
    flow_weight: float = 1.0,
    endpoint_weight: float = 0.20,
    reread_weight: float = 0.10,
    stroke_weight: float = 4.0,
    condition_dropout: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    count = target_cell.shape[0]
    expected_cell = (count, 1, 32, 32)
    if tuple(target_cell.shape) != expected_cell:
        raise ValueError("flow target must have shape [N,1,32,32]")
    if tuple(reference_target_cell.shape) != expected_cell:
        raise ValueError("flow reference target must align with target")
    if context_state.shape != (count, model.config.model_dim):
        raise ValueError("flow context states do not align with targets")
    if proposed_visual.shape != (count, model.config.visual_dim):
        raise ValueError("flow visual proposals do not align with targets")
    random_time = torch.rand(
        (count,), device=target_cell.device, dtype=target_cell.dtype, generator=generator
    )
    time = random_time.sqrt()
    force_noise = torch.rand(
        (count,), device=target_cell.device, generator=generator
    ) < 0.20
    time = torch.where(force_noise, torch.ones_like(time), time)
    noise = torch.randn(
        target_cell.shape,
        device=target_cell.device,
        dtype=target_cell.dtype,
        generator=generator,
    )
    noisy = (1.0 - time[:, None, None, None]) * target_cell + time[
        :, None, None, None
    ] * noise
    target_velocity = noise - target_cell
    dropout = (
        model.config.condition_dropout
        if condition_dropout is None
        else condition_dropout
    )
    condition_present = (
        torch.rand((count, 1), device=target_cell.device, generator=generator)
        >= dropout
    ).to(target_cell.dtype)
    predicted_velocity = model.flow_velocity(
        noisy,
        time,
        context_state,
        proposed_visual,
        condition_present=condition_present,
    )
    flow = F.mse_loss(predicted_velocity.float(), target_velocity.float())
    endpoint = noisy - time[:, None, None, None] * predicted_velocity
    pixel_weight = 1.0 + stroke_weight * target_cell
    endpoint_loss = (
        (endpoint.float() - target_cell.float()).abs() * pixel_weight.float()
    ).sum() / pixel_weight.float().sum().clamp_min(1.0)
    reread_image = endpoint.clamp(0, 1)
    reread_visual = model.encode_cells(reread_image[:, None], target=True)[:, 0]
    with torch.no_grad():
        target_visual = model.encode_cells(
            reference_target_cell[:, None], target=True
        )[:, 0]
    reread_cosine = (reread_visual * target_visual).sum(dim=-1)
    reread = (1.0 - reread_cosine).mean()
    total = flow_weight * flow + endpoint_weight * endpoint_loss + reread_weight * reread
    with torch.no_grad():
        predicted_ink = reread_image >= 0.5
        target_ink = target_cell >= 0.5
        true_positive = (predicted_ink & target_ink).flatten(1).sum(dim=1).float()
        precision = true_positive / predicted_ink.flatten(1).sum(dim=1).float().clamp_min(1.0)
        recall = true_positive / target_ink.flatten(1).sum(dim=1).float().clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
    return total, {
        "loss": total.detach(),
        "flow_loss": flow.detach(),
        "endpoint_loss": endpoint_loss.detach(),
        "reread_loss": reread.detach(),
        "reread_target_cosine": reread_cosine.mean().detach(),
        "endpoint_pixel_f1": f1.mean().detach(),
        "mean_flow_time": time.mean().detach(),
        "condition_present_fraction": condition_present.mean().detach(),
    }


def visual_cell_model_config_payload(
    config: VisualCellStreamConfig,
) -> dict[str, Any]:
    return asdict(config)


def visual_cell_model_config_from_payload(
    payload: dict[str, Any],
) -> VisualCellStreamConfig:
    return VisualCellStreamConfig(**payload)


def visual_cell_model_boundary_receipt(
    config: VisualCellStreamConfig,
) -> dict[str, bool | str | int | list[int]]:
    return {
        "architecture": "visual-cell-stream-v25",
        "input_shape": [config.maximum_cells, 1, 32, 32],
        "output_shape": [1, 32, 32],
        "visual_time_volume_axes": ["time", "channel", "height", "width"],
        "each_time_slice_is_a_clean_2d_cell": True,
        "geometric_depth_is_one": True,
        "input_is_continuous_image_stream": True,
        "output_is_continuous_image": True,
        "causal_over_visual_time": True,
        "uses_continuous_flow_time": True,
        "rereads_generated_pixels": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_embedding": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }

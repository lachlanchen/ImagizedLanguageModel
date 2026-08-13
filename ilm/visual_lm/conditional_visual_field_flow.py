from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense_visual_future_energy import ResidualSemanticAdapter
from .saccade_lm import FovealRetina, VisualSaccadeConfig
from .spatial_visual_next_field import model_state_sha256
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V31_ARCHITECTURE = "conditional-visual-field-flow-v31"
V31_SPATIAL_ROUTE = "spatial-field"
V31_GLOBAL_ROUTE = "global-control"
V31_ROUTES = (V31_SPATIAL_ROUTE, V31_GLOBAL_ROUTE)
V31_SUFFIX_CELLS = 4
V31_FIELD_SIZE = 4
V31_FIELD_CELLS = V31_FIELD_SIZE**2
V31_SPATIAL_PERMUTATION = tuple(reversed(range(V31_FIELD_CELLS)))
V31_TRAIN_PROBE_TIMES = (0.10, 0.35)
V31_AUDIT_PROBE_TIMES = (0.03, 0.07, 0.12, 0.20, 0.30, 0.42, 0.55, 0.70)


@dataclass(frozen=True)
class ConditionalVisualFieldFlowConfig:
    cell_size: int = 32
    maximum_cells: int = 64
    suffix_cells: int = V31_SUFFIX_CELLS
    visual_dim: int = 192
    semantic_dim: int = 192
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    retina_base_channels: int = 64
    semantic_hidden_dim: int = 384
    semantic_residual_scale: float = 0.10
    field_size: int = V31_FIELD_SIZE
    field_channels: int = 192
    velocity_hidden_channels: int = 192
    velocity_blocks: int = 4
    velocity_kernel_size: int = 3
    velocity_mlp_ratio: float = 2.0
    velocity_dropout: float = 0.05
    time_embedding_dim: int = 128
    initial_path_temperature: float = 0.25
    score_chunk_size: int = 32
    route_mode: str = V31_SPATIAL_ROUTE

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V31 requires 32x32 visual cells")
        if self.maximum_cells != 64 or self.suffix_cells != V31_SUFFIX_CELLS:
            raise ValueError("V31 fixes 64 context cells and a suffix of four")
        if self.visual_dim < 24 or self.semantic_dim != self.visual_dim:
            raise ValueError("V31 raw and semantic dimensions must align")
        if self.retina_base_channels * 3 != self.field_channels:
            raise ValueError("V31 field channels must match the retinal field")
        if self.field_channels != self.visual_dim:
            raise ValueError("V31 spatial and global field widths must match")
        if self.model_dim < 64 or self.model_dim % self.heads:
            raise ValueError("V31 causal width must divide into causal heads")
        if self.layers < 1 or self.mlp_ratio < 2.0:
            raise ValueError("V31 causal field configuration is invalid")
        if self.semantic_hidden_dim < 8:
            raise ValueError("V31 semantic adapter is underspecified")
        if not 0.0 < self.semantic_residual_scale <= 1.0:
            raise ValueError("V31 semantic residual scale must be in (0,1]")
        if self.field_size != V31_FIELD_SIZE:
            raise ValueError("V31 fixes a 4x4 retinal field")
        if self.velocity_hidden_channels != self.field_channels:
            raise ValueError("V31 fixes equal field and velocity widths")
        if self.velocity_blocks < 1:
            raise ValueError("V31 needs at least one velocity block")
        if self.velocity_kernel_size != 3 or self.velocity_mlp_ratio != 2.0:
            raise ValueError("V31 velocity topology differs from the protocol")
        if self.time_embedding_dim < 16 or self.time_embedding_dim % 2:
            raise ValueError("V31 time embedding must be positive and even")
        for value in (self.dropout, self.velocity_dropout):
            if not 0.0 <= value < 1.0:
                raise ValueError("V31 dropout must be in [0,1)")
        if not 0.01 <= self.initial_path_temperature <= 1.0:
            raise ValueError("V31 path temperature must be in [0.01,1]")
        if self.score_chunk_size < 1:
            raise ValueError("V31 score chunk must be positive")
        if self.route_mode not in V31_ROUTES:
            raise ValueError(f"V31 route must be one of {V31_ROUTES}")


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError("V31 channel normalization expects [B,C,H,W]")
        return self.norm(field.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def sinusoidal_time_embedding(times: torch.Tensor, width: int) -> torch.Tensor:
    if times.ndim != 1 or width % 2:
        raise ValueError("V31 times must be [B] and embedding width must be even")
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=times.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    angles = times.float()[:, None] * frequencies[None] * (2.0 * math.pi)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ConditionalVelocityBlock(nn.Module):
    def __init__(self, config: ConditionalVisualFieldFlowConfig) -> None:
        super().__init__()
        channels = config.velocity_hidden_channels
        hidden = int(channels * config.velocity_mlp_ratio)
        self.spatial_norm = ChannelLayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            config.velocity_kernel_size,
            padding=config.velocity_kernel_size // 2,
            groups=channels,
        )
        self.channel_norm = ChannelLayerNorm2d(channels)
        self.condition = nn.Linear(config.model_dim, channels * 2)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dropout = nn.Dropout(config.velocity_dropout)
        self.contract = nn.Conv2d(hidden, channels, 1)

    def forward(
        self,
        field: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[0] != field.shape[0]:
            raise ValueError("V31 block condition must align with field batch")
        spatial = field + self.depthwise(self.spatial_norm(field))
        normalized = self.channel_norm(spatial)
        gamma, beta = self.condition(F.silu(condition)).chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * gamma.tanh()
        modulated = normalized * gamma[:, :, None, None] + beta[:, :, None, None]
        update = self.contract(self.dropout(F.silu(self.expand(modulated))))
        return spatial + update


class ConditionalFieldVelocity(nn.Module):
    def __init__(self, config: ConditionalVisualFieldFlowConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.velocity_hidden_channels
        self.field_input = nn.Conv2d(config.field_channels, channels, 1)
        self.position = nn.Parameter(
            torch.empty(1, channels, config.field_size, config.field_size)
        )
        self.context_norm = nn.LayerNorm(config.model_dim)
        self.context_projection = nn.Linear(config.model_dim, config.model_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.blocks = nn.ModuleList(
            ConditionalVelocityBlock(config) for _ in range(config.velocity_blocks)
        )
        self.output_norm = ChannelLayerNorm2d(channels)
        self.output = nn.Conv2d(channels, config.field_channels, 1)

    def forward(
        self,
        context: torch.Tensor,
        noisy_field: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        if context.ndim != 2 or context.shape[-1] != config.model_dim:
            raise ValueError("V31 velocity context must be [B,model_dim]")
        if noisy_field.ndim != 3 or tuple(noisy_field.shape[1:]) != (
            V31_FIELD_CELLS,
            config.field_channels,
        ):
            raise ValueError("V31 noisy field must be [B,16,C]")
        if times.shape != context.shape[:1]:
            raise ValueError("V31 velocity times must have one value per row")
        field = noisy_field.permute(0, 2, 1).reshape(
            noisy_field.shape[0],
            config.field_channels,
            config.field_size,
            config.field_size,
        )
        hidden = self.field_input(field) + self.position
        time = sinusoidal_time_embedding(times, config.time_embedding_dim)
        condition = self.context_projection(self.context_norm(context))
        condition = condition + self.time_projection(
            time.to(self.time_projection[0].weight.dtype)
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        velocity = self.output(self.output_norm(hidden))
        return velocity.flatten(2).transpose(1, 2).float()


class ConditionalVisualFieldFlowModel(nn.Module):
    """A bank-free conditional flow over coherent continuous writing fields."""

    _BACKBONE_MODULES = (
        "retina",
        "semantic_adapter",
        "target_semantic_adapter",
        "context_input",
        "context_blocks",
        "context_norm",
    )

    def __init__(self, config: ConditionalVisualFieldFlowConfig) -> None:
        super().__init__()
        self.config = config
        retina_config = VisualSaccadeConfig(
            fovea_size=config.cell_size,
            visual_dim=config.visual_dim,
            state_dim=config.model_dim,
            state_layers=1,
            retina_base_channels=config.retina_base_channels,
            ink_base_channels=32,
            dropout=0.0,
        )
        self.retina = FovealRetina(retina_config)
        self.semantic_adapter = ResidualSemanticAdapter(
            config.visual_dim,
            config.semantic_hidden_dim,
            config.semantic_residual_scale,
        )
        self.target_semantic_adapter = copy.deepcopy(self.semantic_adapter)
        self.context_input = nn.Linear(
            config.visual_dim + config.semantic_dim,
            config.model_dim,
            bias=False,
        )
        self.context_blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.context_norm = RMSNorm(config.model_dim)
        self.velocity_decoder = ConditionalFieldVelocity(config)
        self.log_path_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_path_temperature))
        )
        self._initialize_new_path()
        self._freeze_perception()

    def _initialize_new_path(self) -> None:
        nn.init.normal_(self.velocity_decoder.position, std=0.02)
        for module in self.velocity_decoder.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.velocity_blocks)
        for block in self.velocity_decoder.blocks:
            block.contract.weight.data.mul_(residual_scale)

    def _freeze_perception(self) -> None:
        for module in (
            self.retina,
            self.semantic_adapter,
            self.target_semantic_adapter,
        ):
            module.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> ConditionalVisualFieldFlowModel:
        super().train(mode)
        self._freeze_perception()
        return self

    @property
    def path_score_scale(self) -> torch.Tensor:
        return self.log_path_scale.exp().clamp(max=100.0)

    @staticmethod
    def _validate_images(images: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(images):
            raise TypeError(f"V31 {name} must be a floating image tensor")
        if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V31 {name} must end in [1,32,32]")

    @torch.no_grad()
    def encode_image_parts(
        self,
        images: torch.Tensor,
        *,
        target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_images(images, name="images")
        leading = images.shape[:-3]
        flat = images.reshape(-1, 1, 32, 32).clamp(0, 1)
        raw, field = self.retina.forward_with_field(flat)
        raw = F.normalize(raw.float(), dim=-1)
        adapter = self.target_semantic_adapter if target else self.semantic_adapter
        semantic = F.normalize(adapter(raw).float(), dim=-1)
        cells = (
            field.float()
            .permute(0, 2, 3, 1)
            .reshape(flat.shape[0], V31_FIELD_CELLS, self.config.field_channels)
        )
        cells = F.normalize(cells, dim=-1)
        return (
            raw.reshape(*leading, self.config.visual_dim),
            semantic.reshape(*leading, self.config.semantic_dim),
            cells.reshape(*leading, V31_FIELD_CELLS, self.config.field_channels),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_images(context, name="context")
        if context.ndim != 5:
            raise ValueError("V31 context must have shape [B,T,1,32,32]")
        if not 1 <= context.shape[1] <= self.config.maximum_cells:
            raise ValueError("V31 context length must be in [1,64]")
        raw, semantic, _ = self.encode_image_parts(context, target=False)
        state = self.context_input(
            torch.cat((raw, semantic), dim=-1).to(self.context_input.weight.dtype)
        )
        for block in self.context_blocks:
            state = block(state)
        return self.context_norm(state)

    def context_condition(self, context: torch.Tensor) -> torch.Tensor:
        return self.encode_context(context)[:, -1]

    def encode_route_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        _, semantic, spatial = self.encode_image_parts(candidates, target=True)
        if self.config.route_mode == V31_SPATIAL_ROUTE:
            return spatial
        return semantic.unsqueeze(-2).expand(
            *semantic.shape[:-1], V31_FIELD_CELLS, self.config.field_channels
        )

    def velocity(
        self,
        condition: torch.Tensor,
        noisy_field: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity_decoder(condition, noisy_field, times)

    def make_coherent_base(
        self,
        rows: int,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if rows < 1:
            raise ValueError("V31 coherent base needs at least one row")
        if device is None:
            device = self.log_path_scale.device
        vectors = torch.randn(
            rows,
            self.config.field_channels,
            device=device,
            generator=generator,
            dtype=torch.float32,
        )
        vectors = F.normalize(vectors, dim=-1)
        return vectors[:, None].expand(-1, V31_FIELD_CELLS, -1).clone()

    @staticmethod
    def _validate_probes(
        probe_bases: torch.Tensor,
        probe_times: torch.Tensor,
        channels: int,
    ) -> None:
        if probe_bases.ndim != 3 or tuple(probe_bases.shape[1:]) != (
            V31_FIELD_CELLS,
            channels,
        ):
            raise ValueError("V31 probe bases must be [M,16,C]")
        if probe_times.shape != probe_bases.shape[:1]:
            raise ValueError("V31 probe times must be [M]")
        if not torch.all((probe_times > 0) & (probe_times < 1)):
            raise ValueError("V31 probe times must be strictly inside (0,1)")

    def path_score_encoded_shared(
        self,
        condition: torch.Tensor,
        candidate_fields: torch.Tensor,
        probe_bases: torch.Tensor,
        probe_times: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.config.model_dim:
            raise ValueError("V31 shared score condition must be [B,D]")
        if candidate_fields.ndim != 3 or tuple(candidate_fields.shape[1:]) != (
            V31_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V31 shared candidates must be [N,16,C]")
        self._validate_probes(probe_bases, probe_times, self.config.field_channels)
        width = self.config.score_chunk_size if chunk_size is None else chunk_size
        if width < 1:
            raise ValueError("V31 score chunk must be positive")
        batch = condition.shape[0]
        probes = probe_bases.shape[0]
        scores: list[torch.Tensor] = []
        for start in range(0, candidate_fields.shape[0], width):
            candidate = candidate_fields[start : start + width]
            count = candidate.shape[0]
            target = candidate[None, :, None].expand(batch, -1, probes, -1, -1)
            base = probe_bases[None, None].expand(batch, count, -1, -1, -1)
            times = probe_times[None, None, :].expand(batch, count, -1)
            noisy = (1.0 - times[..., None, None]) * base + (
                times[..., None, None] * target
            )
            expanded_condition = condition[:, None, None].expand(-1, count, probes, -1)
            velocity = self.velocity(
                expanded_condition.reshape(-1, self.config.model_dim),
                noisy.reshape(-1, V31_FIELD_CELLS, self.config.field_channels),
                times.reshape(-1),
            ).reshape(batch, count, probes, V31_FIELD_CELLS, -1)
            residual = velocity - (target - base)
            error = residual.float().square().sum(dim=-1).mean(dim=(-1, -2))
            scores.append(-self.path_score_scale.float() * error)
        return torch.cat(scores, dim=1)

    def path_score_encoded_batched(
        self,
        condition: torch.Tensor,
        candidate_fields: torch.Tensor,
        probe_bases: torch.Tensor,
        probe_times: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim != 3:
            raise ValueError("V31 paired condition must be [B,Q,D]")
        if candidate_fields.ndim != 4 or (
            candidate_fields.shape[0] != condition.shape[0]
        ):
            raise ValueError("V31 paired candidates must be [B,K,16,C]")
        self._validate_probes(probe_bases, probe_times, self.config.field_channels)
        batch, queries = condition.shape[:2]
        candidates = candidate_fields.shape[1]
        probes = probe_bases.shape[0]
        target = candidate_fields[:, None, :, None].expand(
            -1, queries, -1, probes, -1, -1
        )
        base = probe_bases[None, None, None].expand(
            batch, queries, candidates, -1, -1, -1
        )
        times = probe_times[None, None, None].expand(batch, queries, candidates, -1)
        noisy = (1.0 - times[..., None, None]) * base + (
            times[..., None, None] * target
        )
        expanded_condition = condition[:, :, None, None].expand(
            -1, -1, candidates, probes, -1
        )
        velocity = self.velocity(
            expanded_condition.reshape(-1, self.config.model_dim),
            noisy.reshape(-1, V31_FIELD_CELLS, self.config.field_channels),
            times.reshape(-1),
        ).reshape(batch, queries, candidates, probes, V31_FIELD_CELLS, -1)
        residual = velocity - (target - base)
        error = residual.float().square().sum(dim=-1).mean(dim=(-1, -2))
        return -self.path_score_scale.float() * error

    def path_score_shared_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        probe_bases: torch.Tensor,
        probe_times: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        return self.path_score_encoded_shared(
            self.context_condition(context),
            self.encode_route_candidates(candidates),
            probe_bases,
            probe_times,
            chunk_size=chunk_size,
        )

    def path_score_paired_candidates(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
        probe_bases: torch.Tensor,
        probe_times: torch.Tensor,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or candidates.ndim != 5:
            raise ValueError("V31 pairs require contexts [B,Q,T,1,32,32]")
        if contexts.shape[0] != candidates.shape[0]:
            raise ValueError("V31 pair batches do not align")
        batch, queries = contexts.shape[:2]
        condition = self.context_condition(
            contexts.reshape(batch * queries, *contexts.shape[2:])
        ).reshape(batch, queries, self.config.model_dim)
        fields = self.encode_route_candidates(candidates)
        return self.path_score_encoded_batched(
            condition, fields, probe_bases, probe_times
        )

    def path_score_exact_suffix_paired(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
        probe_bases: torch.Tensor,
        probe_times: torch.Tensor,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or contexts.shape[1] != 2:
            raise ValueError("V31 exact suffix contexts must be [B,2,T,1,32,32]")
        suffix = contexts[:, :, -self.config.suffix_cells :]
        if not torch.equal(suffix[:, 0], suffix[:, 1]):
            raise ValueError("V31 exact suffix pixels differ between pair rows")
        score = self.path_score_paired_candidates(
            suffix[:, :1], candidates, probe_bases, probe_times
        )
        return score.expand(-1, 2, -1)

    def sample_encoded(
        self,
        condition: torch.Tensor,
        base_vectors: torch.Tensor,
        *,
        steps: int = 8,
        solver: str = "heun",
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.config.model_dim:
            raise ValueError("V31 sampling condition must be [B,D]")
        if base_vectors.ndim == 2:
            base_vectors = base_vectors[None].expand(condition.shape[0], -1, -1)
        if base_vectors.ndim != 3 or base_vectors.shape[0] != condition.shape[0]:
            raise ValueError("V31 base vectors must be [K,C] or [B,K,C]")
        if base_vectors.shape[-1] != self.config.field_channels:
            raise ValueError("V31 base vector width is invalid")
        if steps < 1 or solver not in {"euler", "heun"}:
            raise ValueError("V31 sampling requires Euler/Heun and positive steps")
        batch, samples = base_vectors.shape[:2]
        normalized = F.normalize(base_vectors.float(), dim=-1)
        field = normalized[:, :, None].expand(-1, -1, V31_FIELD_CELLS, -1).clone()
        flat_condition = (
            condition[:, None]
            .expand(-1, samples, -1)
            .reshape(batch * samples, self.config.model_dim)
        )
        field = field.reshape(
            batch * samples, V31_FIELD_CELLS, self.config.field_channels
        )
        step_size = 1.0 / steps
        for index in range(steps):
            start = index * step_size
            time = torch.full(
                (batch * samples,), start, device=field.device, dtype=torch.float32
            )
            first = self.velocity(flat_condition, field, time)
            if solver == "euler":
                field = field + step_size * first
                continue
            proposal = field + step_size * first
            end_time = torch.full_like(time, (index + 1) * step_size)
            second = self.velocity(flat_condition, proposal, end_time)
            field = field + 0.5 * step_size * (first + second)
        field = F.normalize(field.float(), dim=-1)
        return field.reshape(
            batch,
            samples,
            V31_FIELD_CELLS,
            self.config.field_channels,
        )

    def sample(
        self,
        context: torch.Tensor,
        base_vectors: torch.Tensor,
        *,
        steps: int = 8,
        solver: str = "heun",
    ) -> torch.Tensor:
        return self.sample_encoded(
            self.context_condition(context),
            base_vectors,
            steps=steps,
            solver=solver,
        )

    @staticmethod
    def sample_score_encoded_shared(
        samples: torch.Tensor,
        candidate_fields: torch.Tensor,
        *,
        kernel_scale: float = 16.0,
    ) -> torch.Tensor:
        if samples.ndim != 4 or samples.shape[-2] != V31_FIELD_CELLS:
            raise ValueError("V31 samples must be [B,K,16,C]")
        if (
            candidate_fields.ndim != 3
            or candidate_fields.shape[-2:] != samples.shape[-2:]
        ):
            raise ValueError("V31 sample candidates must be [N,16,C]")
        similarity = (
            torch.einsum("bkpc,npc->bkn", samples.float(), candidate_fields.float())
            / V31_FIELD_CELLS
        )
        return torch.logsumexp(kernel_scale * similarity, dim=1) - math.log(
            samples.shape[1]
        )

    @staticmethod
    def sample_score_encoded_batched(
        samples: torch.Tensor,
        candidate_fields: torch.Tensor,
        *,
        kernel_scale: float = 16.0,
    ) -> torch.Tensor:
        if samples.ndim != 5 or candidate_fields.ndim != 4:
            raise ValueError("V31 paired samples/candidates have invalid rank")
        if samples.shape[0] != candidate_fields.shape[0]:
            raise ValueError("V31 paired sample batches do not align")
        similarity = (
            torch.einsum(
                "bqspc,bkpc->bqsk",
                samples.float(),
                candidate_fields.float(),
            )
            / V31_FIELD_CELLS
        )
        return torch.logsumexp(kernel_scale * similarity, dim=2) - math.log(
            samples.shape[2]
        )

    def load_v30_backbone_state(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        loaded: dict[str, int] = {}
        for name in self._BACKBONE_MODULES:
            prefix = f"{name}."
            module_state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not module_state:
                raise ValueError(f"V31 V30 source contains no {name} state")
            getattr(self, name).load_state_dict(module_state, strict=True)
            loaded[name] = len(module_state)
        self._freeze_perception()
        return {
            "source_architecture": "spatial-visual-next-field-v30",
            "source_route": "global-control",
            "loaded_modules": loaded,
            "discarded_v30_field_decoder": True,
            "discarded_v30_logit_scale": True,
        }


def spatially_permute_v31_fields(
    fields: torch.Tensor,
    permutation: tuple[int, ...] = V31_SPATIAL_PERMUTATION,
) -> torch.Tensor:
    if fields.ndim < 3 or fields.shape[-2] != V31_FIELD_CELLS:
        raise ValueError("V31 fields must contain 16 retinal cells")
    if tuple(sorted(permutation)) != tuple(range(V31_FIELD_CELLS)):
        raise ValueError("V31 spatial permutation is not bijective")
    index = torch.tensor(permutation, device=fields.device)
    return fields.index_select(-2, index)


def conditional_visual_field_flow_config_payload(
    config: ConditionalVisualFieldFlowConfig,
) -> dict[str, Any]:
    return asdict(config)


def conditional_visual_field_flow_config_from_payload(
    payload: Mapping[str, Any],
) -> ConditionalVisualFieldFlowConfig:
    return ConditionalVisualFieldFlowConfig(**dict(payload))


def conditional_visual_field_flow_boundary_receipt(
    config: ConditionalVisualFieldFlowConfig,
) -> dict[str, Any]:
    return {
        "architecture": V31_ARCHITECTURE,
        "route_mode": config.route_mode,
        "context_shape": [config.maximum_cells, 1, 32, 32],
        "generated_field_shape": [V31_FIELD_CELLS, config.field_channels],
        "candidate_shape": [1, 32, 32],
        "input_is_continuous_image_stream": True,
        "output_is_candidate_independent_continuous_distribution": True,
        "autonomous_sampler_requires_candidates": False,
        "candidate_is_arbitrary_image": True,
        "coherent_base_width": config.field_channels,
        "independent_patch_noise": False,
        "retina_is_frozen": True,
        "semantic_adapters_are_frozen": True,
        "candidate_bank_deployed": False,
        "candidate_bank_in_model_state": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_embedding": False,
        "uses_vocabulary_output": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
    }


__all__ = [
    "ConditionalVisualFieldFlowConfig",
    "ConditionalVisualFieldFlowModel",
    "V31_ARCHITECTURE",
    "V31_AUDIT_PROBE_TIMES",
    "V31_FIELD_CELLS",
    "V31_FIELD_SIZE",
    "V31_GLOBAL_ROUTE",
    "V31_ROUTES",
    "V31_SPATIAL_PERMUTATION",
    "V31_SPATIAL_ROUTE",
    "V31_SUFFIX_CELLS",
    "V31_TRAIN_PROBE_TIMES",
    "conditional_visual_field_flow_boundary_receipt",
    "conditional_visual_field_flow_config_from_payload",
    "conditional_visual_field_flow_config_payload",
    "model_state_sha256",
    "sinusoidal_time_embedding",
    "spatially_permute_v31_fields",
]

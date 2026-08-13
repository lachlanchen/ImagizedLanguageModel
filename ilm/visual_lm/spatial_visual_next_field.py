from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense_visual_future_energy import ResidualSemanticAdapter
from .joint_visual_compatibility import paired_assignment_loss
from .saccade_lm import FovealRetina, VisualSaccadeConfig
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V30_ARCHITECTURE = "spatial-visual-next-field-v30"
V30_SPATIAL_ROUTE = "spatial-field"
V30_GLOBAL_ROUTE = "global-control"
V30_ROUTES = (V30_SPATIAL_ROUTE, V30_GLOBAL_ROUTE)
V30_SUFFIX_CELLS = 4
V30_FIELD_SIZE = 4
V30_FIELD_CELLS = V30_FIELD_SIZE**2
V30_SPATIAL_PERMUTATION = tuple(reversed(range(V30_FIELD_CELLS)))


@dataclass(frozen=True)
class SpatialVisualNextFieldConfig:
    cell_size: int = 32
    maximum_cells: int = 64
    suffix_cells: int = V30_SUFFIX_CELLS
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
    field_size: int = V30_FIELD_SIZE
    field_channels: int = 192
    decoder_hidden_channels: int = 192
    decoder_blocks: int = 2
    decoder_kernel_size: int = 3
    decoder_mlp_ratio: float = 2.0
    decoder_dropout: float = 0.05
    initial_temperature: float = 0.08
    score_chunk_size: int = 128
    route_mode: str = V30_SPATIAL_ROUTE

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V30 requires 32x32 visual cells")
        if self.maximum_cells != 64 or self.suffix_cells != V30_SUFFIX_CELLS:
            raise ValueError("V30 fixes 64 context cells and a suffix of four")
        if self.visual_dim < 24 or self.semantic_dim != self.visual_dim:
            raise ValueError("V30 raw and semantic dimensions must align")
        if self.retina_base_channels * 3 != self.field_channels:
            raise ValueError("V30 field channels must match the retinal field")
        if self.field_channels != self.visual_dim:
            raise ValueError("V30 spatial and global candidate widths must match")
        if self.model_dim < 64 or self.model_dim % self.heads:
            raise ValueError("V30 causal width must divide into causal heads")
        if self.layers < 1 or self.mlp_ratio < 2.0:
            raise ValueError("V30 causal field configuration is invalid")
        if self.semantic_hidden_dim < 8:
            raise ValueError("V30 semantic adapter is underspecified")
        if not 0.0 < self.semantic_residual_scale <= 1.0:
            raise ValueError("V30 semantic residual scale must be in (0,1]")
        if self.field_size != V30_FIELD_SIZE:
            raise ValueError("V30 fixes a 4x4 retinal field")
        if self.decoder_hidden_channels < 24 or self.decoder_blocks != 2:
            raise ValueError("V30 fixes two adequately sized decoder blocks")
        if self.decoder_kernel_size != 3 or self.decoder_mlp_ratio != 2.0:
            raise ValueError("V30 decoder topology differs from the protocol")
        for value in (self.dropout, self.decoder_dropout):
            if not 0.0 <= value < 1.0:
                raise ValueError("V30 dropout must be in [0,1)")
        if not 0.01 <= self.initial_temperature <= 1.0:
            raise ValueError("V30 initial temperature must be in [0.01,1]")
        if self.score_chunk_size < 1:
            raise ValueError("V30 score chunk must be positive")
        if self.route_mode not in V30_ROUTES:
            raise ValueError(f"V30 route must be one of {V30_ROUTES}")


class ChannelLayerNorm2d(nn.Module):
    """Normalize channels independently at each retinal location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError("V30 channel normalization expects [B,C,H,W]")
        return self.norm(field.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpatialFieldResidualBlock(nn.Module):
    """Mix neighboring cells, then transform each cell channel-wise."""

    def __init__(self, config: SpatialVisualNextFieldConfig) -> None:
        super().__init__()
        channels = config.decoder_hidden_channels
        hidden = int(channels * config.decoder_mlp_ratio)
        self.spatial_norm = ChannelLayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            config.decoder_kernel_size,
            padding=config.decoder_kernel_size // 2,
            groups=channels,
        )
        self.channel_norm = ChannelLayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.contract = nn.Conv2d(hidden, channels, 1)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        spatial = field + self.depthwise(self.spatial_norm(field))
        update = self.contract(
            self.dropout(F.silu(self.expand(self.channel_norm(spatial))))
        )
        return field + update


class SpatialNextFieldDecoder(nn.Module):
    """Decode one causal language state into a continuous 4x4 visual field."""

    def __init__(self, config: SpatialVisualNextFieldConfig) -> None:
        super().__init__()
        hidden = config.decoder_hidden_channels
        self.config = config
        self.seed_norm = nn.LayerNorm(config.model_dim)
        self.seed = nn.Linear(
            config.model_dim,
            hidden * config.field_size * config.field_size,
        )
        self.blocks = nn.ModuleList(
            SpatialFieldResidualBlock(config)
            for _ in range(config.decoder_blocks)
        )
        self.output_norm = ChannelLayerNorm2d(hidden)
        self.output = nn.Conv2d(hidden, config.field_channels, 1)

    def forward(self, causal_state: torch.Tensor) -> torch.Tensor:
        if causal_state.ndim != 2 or causal_state.shape[-1] != self.config.model_dim:
            raise ValueError("V30 decoder state must be [B,model_dim]")
        batch = causal_state.shape[0]
        field = self.seed(self.seed_norm(causal_state)).reshape(
            batch,
            self.config.decoder_hidden_channels,
            self.config.field_size,
            self.config.field_size,
        )
        for block in self.blocks:
            field = block(field)
        field = self.output(self.output_norm(field))
        cells = field.permute(0, 2, 3, 1).reshape(
            batch,
            self.config.field_size**2,
            self.config.field_channels,
        )
        return F.normalize(cells.float(), dim=-1)


class SpatialVisualNextFieldModel(nn.Module):
    """Predict and score a continuous next-writing field from image history."""

    _BACKBONE_MODULES = (
        "retina",
        "semantic_adapter",
        "target_semantic_adapter",
        "context_input",
        "context_blocks",
        "context_norm",
    )

    def __init__(self, config: SpatialVisualNextFieldConfig) -> None:
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
        self.field_decoder = SpatialNextFieldDecoder(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )
        self._initialize_new_path()
        self._freeze_perception()

    def _initialize_new_path(self) -> None:
        for module in self.field_decoder.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.decoder_blocks)
        for block in self.field_decoder.blocks:
            block.contract.weight.data.mul_(residual_scale)

    def _freeze_perception(self) -> None:
        for module in (
            self.retina,
            self.semantic_adapter,
            self.target_semantic_adapter,
        ):
            module.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> SpatialVisualNextFieldModel:
        super().train(mode)
        self._freeze_perception()
        return self

    @property
    def score_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    @staticmethod
    def _validate_images(images: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(images):
            raise TypeError(f"V30 {name} must be a floating image tensor")
        if images.ndim < 4 or tuple(images.shape[-3:]) != (1, 32, 32):
            raise ValueError(f"V30 {name} must end in [1,32,32]")

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
        cells = field.float().permute(0, 2, 3, 1).reshape(
            flat.shape[0], V30_FIELD_CELLS, self.config.field_channels
        )
        cells = F.normalize(cells, dim=-1)
        return (
            raw.reshape(*leading, self.config.visual_dim),
            semantic.reshape(*leading, self.config.semantic_dim),
            cells.reshape(*leading, V30_FIELD_CELLS, self.config.field_channels),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_images(context, name="context")
        if context.ndim != 5:
            raise ValueError("V30 context must have shape [B,T,1,32,32]")
        if not 1 <= context.shape[1] <= self.config.maximum_cells:
            raise ValueError("V30 context length must be in [1,64]")
        raw, semantic, _ = self.encode_image_parts(context, target=False)
        combined = torch.cat((raw, semantic), dim=-1)
        state = self.context_input(combined.to(self.context_input.weight.dtype))
        for block in self.context_blocks:
            state = block(state)
        return self.context_norm(state)

    def predict_encoded_field(self, context_state: torch.Tensor) -> torch.Tensor:
        if context_state.ndim != 3:
            raise ValueError("V30 context state must be [B,T,D]")
        return self.field_decoder(context_state[:, -1])

    def predict_field(self, context: torch.Tensor) -> torch.Tensor:
        return self.predict_encoded_field(self.encode_context(context))

    def encode_route_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Encode arbitrary candidate images under the fixed experimental route."""

        _, semantic, spatial = self.encode_image_parts(candidates, target=True)
        if self.config.route_mode == V30_SPATIAL_ROUTE:
            return spatial
        return semantic.unsqueeze(-2).expand(
            *semantic.shape[:-1], V30_FIELD_CELLS, self.config.field_channels
        )

    @staticmethod
    def _validate_field(field: torch.Tensor, *, name: str) -> None:
        if field.ndim < 3 or tuple(field.shape[-2:]) != (
            V30_FIELD_CELLS,
            field.shape[-1],
        ):
            raise ValueError(f"V30 {name} must end in [16,C]")
        if not torch.is_floating_point(field):
            raise TypeError(f"V30 {name} must be floating point")

    def score_encoded_shared(
        self,
        predicted: torch.Tensor,
        candidate_fields: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if predicted.ndim != 3 or predicted.shape[1:] != (
            V30_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V30 predicted field must be [B,16,C]")
        if candidate_fields.ndim != 3 or candidate_fields.shape[1:] != (
            V30_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V30 shared candidate fields must be [N,16,C]")
        width = self.config.score_chunk_size if chunk_size is None else chunk_size
        if width < 1:
            raise ValueError("V30 score chunk must be positive")
        scores: list[torch.Tensor] = []
        for start in range(0, candidate_fields.shape[0], width):
            candidates = candidate_fields[start : start + width]
            score = torch.einsum(
                "bpc,npc->bn",
                predicted.float(),
                candidates.float(),
            ) / V30_FIELD_CELLS
            scores.append(self.score_scale.float() * score)
        return torch.cat(scores, dim=1)

    def score_encoded_batched(
        self,
        predicted: torch.Tensor,
        candidate_fields: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if predicted.ndim != 3 or predicted.shape[1:] != (
            V30_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V30 batched prediction must be [B,16,C]")
        if candidate_fields.ndim != 4 or candidate_fields.shape[0] != predicted.shape[0]:
            raise ValueError("V30 batched candidates must be [B,N,16,C]")
        if candidate_fields.shape[2:] != (
            V30_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V30 batched candidate field width is invalid")
        width = self.config.score_chunk_size if chunk_size is None else chunk_size
        if width < 1:
            raise ValueError("V30 score chunk must be positive")
        scores: list[torch.Tensor] = []
        for start in range(0, candidate_fields.shape[1], width):
            candidates = candidate_fields[:, start : start + width]
            score = torch.einsum(
                "bpc,bnpc->bn",
                predicted.float(),
                candidates.float(),
            ) / V30_FIELD_CELLS
            scores.append(self.score_scale.float() * score)
        return torch.cat(scores, dim=1)

    def score_encoded_paired(
        self,
        predicted: torch.Tensor,
        candidate_fields: torch.Tensor,
    ) -> torch.Tensor:
        if predicted.ndim != 4 or predicted.shape[2:] != (
            V30_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V30 paired prediction must be [B,Q,16,C]")
        if candidate_fields.ndim != 4 or candidate_fields.shape[0] != predicted.shape[0]:
            raise ValueError("V30 paired candidates must be [B,K,16,C]")
        if candidate_fields.shape[2:] != (
            V30_FIELD_CELLS,
            self.config.field_channels,
        ):
            raise ValueError("V30 paired candidate field width is invalid")
        score = torch.einsum(
            "bqpc,bkpc->bqk",
            predicted.float(),
            candidate_fields.float(),
        ) / V30_FIELD_CELLS
        return self.score_scale.float() * score

    def score_shared_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError("V30 shared candidates must be [N,1,32,32]")
        predicted = self.predict_field(context)
        fields = self.encode_route_candidates(candidates)
        return self.score_encoded_shared(predicted, fields, chunk_size=chunk_size)

    def predict_paired_fields(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.ndim != 6:
            raise ValueError("V30 paired contexts must be [B,Q,T,1,32,32]")
        batch, queries, length = contexts.shape[:3]
        state = self.encode_context(
            contexts.reshape(batch * queries, *contexts.shape[2:])
        )
        predicted = self.predict_encoded_field(state)
        return predicted.reshape(
            batch,
            queries,
            V30_FIELD_CELLS,
            self.config.field_channels,
        )

    def score_paired_candidates(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or candidates.ndim != 5:
            raise ValueError(
                "V30 pairs must be [B,Q,T,1,32,32] and [B,K,1,32,32]"
            )
        if contexts.shape[0] != candidates.shape[0]:
            raise ValueError("V30 pair batches do not align")
        predicted = self.predict_paired_fields(contexts)
        fields = self.encode_route_candidates(candidates)
        return self.score_encoded_paired(predicted, fields)

    def score_exact_suffix_paired(
        self,
        contexts: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if contexts.ndim != 6 or contexts.shape[1] != 2:
            raise ValueError("V30 exact suffix contexts must be [B,2,T,1,32,32]")
        suffix = contexts[:, :, -self.config.suffix_cells :]
        if not torch.equal(suffix[:, 0], suffix[:, 1]):
            raise ValueError("V30 exact suffix pixels differ between pair rows")
        score = self.score_paired_candidates(suffix[:, :1], candidates)
        return score.expand(-1, 2, -1)

    def load_v29_backbone_state(
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
                raise ValueError(f"V30 V29 source contains no {name} state")
            module = getattr(self, name)
            module.load_state_dict(module_state, strict=True)
            loaded[name] = len(module_state)
        self._freeze_perception()
        return {
            "source_architecture": "conditional-visual-density-ratio-v29",
            "loaded_modules": loaded,
            "discarded_candidate_critic": True,
        }


def spatially_permute_candidate_fields(
    fields: torch.Tensor,
    permutation: tuple[int, ...] = V30_SPATIAL_PERMUTATION,
) -> torch.Tensor:
    if fields.ndim < 3 or fields.shape[-2] != V30_FIELD_CELLS:
        raise ValueError("V30 candidate fields must contain 16 retinal cells")
    if tuple(sorted(permutation)) != tuple(range(V30_FIELD_CELLS)):
        raise ValueError("V30 spatial permutation is not bijective")
    index = torch.tensor(permutation, device=fields.device)
    return fields.index_select(-2, index)


def per_row_assignment_margin(
    logits: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 2):
        raise ValueError("V30 assignment logits must be [B,2,2]")
    if assignments.shape != logits.shape[:2] or assignments.dtype != torch.long:
        raise ValueError("V30 assignments must be int64 [B,2]")
    correct = logits.gather(2, assignments[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignments)[:, :, None])[:, :, 0]
    return correct.float() - other.float()


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def spatial_visual_next_field_config_payload(
    config: SpatialVisualNextFieldConfig,
) -> dict[str, Any]:
    return asdict(config)


def spatial_visual_next_field_config_from_payload(
    payload: Mapping[str, Any],
) -> SpatialVisualNextFieldConfig:
    return SpatialVisualNextFieldConfig(**dict(payload))


def spatial_visual_next_field_boundary_receipt(
    config: SpatialVisualNextFieldConfig,
) -> dict[str, bool | str | list[int]]:
    return {
        "architecture": V30_ARCHITECTURE,
        "route_mode": config.route_mode,
        "context_shape": [config.maximum_cells, 1, 32, 32],
        "predicted_field_shape": [V30_FIELD_CELLS, config.field_channels],
        "candidate_shape": [1, 32, 32],
        "input_is_continuous_image_stream": True,
        "output_is_candidate_independent_continuous_field": True,
        "candidate_is_arbitrary_image": True,
        "candidate_reduction_occurs_after_local_interaction": (
            config.route_mode == V30_SPATIAL_ROUTE
        ),
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
    "ChannelLayerNorm2d",
    "SpatialFieldResidualBlock",
    "SpatialNextFieldDecoder",
    "SpatialVisualNextFieldConfig",
    "SpatialVisualNextFieldModel",
    "V30_ARCHITECTURE",
    "V30_FIELD_CELLS",
    "V30_FIELD_SIZE",
    "V30_GLOBAL_ROUTE",
    "V30_ROUTES",
    "V30_SPATIAL_PERMUTATION",
    "V30_SPATIAL_ROUTE",
    "V30_SUFFIX_CELLS",
    "model_state_sha256",
    "paired_assignment_loss",
    "per_row_assignment_margin",
    "spatial_visual_next_field_boundary_receipt",
    "spatial_visual_next_field_config_from_payload",
    "spatial_visual_next_field_config_payload",
    "spatially_permute_candidate_fields",
]

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    OrthonormalGlyphField,
)
from .visual_cell_stream import CausalVisualBlock, RMSNorm


HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE = "hierarchical-raster-language-v50"


@dataclass(frozen=True)
class HierarchicalRasterLanguageConfig:
    cell_size: int = 32
    patch_size: int = 8
    coarse_size: int = 8
    local_dim: int = 256
    local_layers: int = 4
    local_heads: int = 8
    local_mlp_ratio: float = 3.0
    local_dropout: float = 0.05
    mixtures: int = 8

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V50 currently requires 32x32 writing cells")
        if self.patch_size < 2 or self.cell_size % self.patch_size:
            raise ValueError("V50 patch size must divide the writing cell")
        if self.coarse_size != self.patch_size:
            raise ValueError(
                "V50 coarse image and detail patches must share one raster width"
            )
        if self.local_dim < 64 or self.local_dim % self.local_heads:
            raise ValueError("V50 local width must divide into attention heads")
        if self.local_layers < 1 or self.local_mlp_ratio < 2.0:
            raise ValueError("V50 local autoregressor is underspecified")
        if not 0.0 <= self.local_dropout < 1.0:
            raise ValueError("V50 local dropout must lie in [0,1)")
        if self.mixtures < 2:
            raise ValueError("V50 requires at least two raster mixture components")

    @property
    def patches_per_side(self) -> int:
        return self.cell_size // self.patch_size

    @property
    def patches_per_cell(self) -> int:
        return self.patches_per_side**2

    @property
    def pixels_per_patch(self) -> int:
        return self.patch_size**2

    @property
    def generation_units(self) -> int:
        return 1 + self.patches_per_cell


class CausalRasterReader(nn.Module):
    """The measured V42 image reader with its failed sampler removed."""

    def __init__(self, config: CanonicalGlyphLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.field = OrthonormalGlyphField(
            size=config.cell_size,
            binary_threshold=config.binary_threshold,
            decoder_sharpness=config.decoder_sharpness,
        )
        self.input_projection = nn.Linear(
            config.field_dim,
            config.model_dim,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.output_norm = RMSNorm(config.model_dim)
        self.anchor_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.field_dim),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def load_v42_state(self, state: Mapping[str, torch.Tensor]) -> None:
        own = self.state_dict()
        transferred = {name: state[name] for name in own if name in state}
        missing = sorted(set(own) - set(transferred))
        if missing:
            raise ValueError(f"V42 reader state is missing tensors: {missing}")
        self.load_state_dict(transferred, strict=True)

    def _validate_context(self, context: torch.Tensor) -> None:
        self.field._validate_pixels(context)
        if context.ndim != 5:
            raise ValueError("V50 context must be [B,T,1,32,32]")
        if not 1 <= context.shape[1] <= self.config.maximum_cells:
            raise ValueError("V50 context length is outside the reader range")

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_context(context)
        visual = self.field.encode_unit(context)
        state = self.input_projection(visual.to(self.input_projection.weight.dtype))
        for block in self.blocks:
            state = block(state)
        hidden = self.output_norm(state)
        anchor = F.normalize(self.anchor_head(hidden).float(), dim=-1)
        return {
            "context_visual": visual,
            "hidden_states": hidden,
            "anchor_fields": anchor,
        }


@dataclass
class RasterMixtureOutput:
    mixture_logits: torch.Tensor
    pixel_logits: torch.Tensor
    hidden_states: torch.Tensor


def raster_to_micro_patches(
    cells: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if not cells.is_floating_point() or cells.ndim != 4:
        raise ValueError("raster cells must be floating [N,1,H,W]")
    if cells.shape[1] != 1 or cells.shape[-2] != cells.shape[-1]:
        raise ValueError("raster cells must be square single-channel images")
    if cells.shape[-1] % patch_size:
        raise ValueError("micro-patch size must divide the cell")
    patches = F.unfold(cells.float(), kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous()


def micro_patches_to_raster(
    patches: torch.Tensor,
    *,
    cell_size: int,
    patch_size: int,
) -> torch.Tensor:
    expected = (cell_size // patch_size) ** 2
    if patches.ndim != 3 or patches.shape[1:] != (
        expected,
        patch_size**2,
    ):
        raise ValueError("micro patches do not tile the requested raster cell")
    return F.fold(
        patches.transpose(1, 2).contiguous(),
        output_size=(cell_size, cell_size),
        kernel_size=patch_size,
        stride=patch_size,
    )


def raster_to_generation_units(
    cells: torch.Tensor,
    *,
    config: HierarchicalRasterLanguageConfig,
) -> torch.Tensor:
    """Return one coarse whole-glyph image followed by local detail patches."""

    details = raster_to_micro_patches(cells, patch_size=config.patch_size)
    pooling = config.cell_size // config.coarse_size
    coarse = F.max_pool2d(
        cells.float(),
        kernel_size=pooling,
        stride=pooling,
    )
    coarse = (coarse >= 0.5).float().flatten(1)
    if coarse.shape[-1] != config.pixels_per_patch:
        raise RuntimeError("V50 coarse raster does not share the patch bit width")
    return torch.cat((coarse[:, None], details), dim=1)


class LocalRasterAutoregressor(nn.Module):
    """Draw one writing cell patch by patch from a visual-language state."""

    def __init__(
        self,
        reader_config: CanonicalGlyphLanguageConfig,
        config: HierarchicalRasterLanguageConfig,
    ) -> None:
        super().__init__()
        self.reader_config = reader_config
        self.config = config
        condition_dim = reader_config.model_dim + reader_config.field_dim
        self.condition = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, config.local_dim),
            nn.SiLU(),
            nn.Linear(config.local_dim, config.local_dim),
        )
        self.patch_input = nn.Linear(config.pixels_per_patch, config.local_dim)
        self.start = nn.Parameter(torch.zeros(config.local_dim))
        self.position = nn.Parameter(
            torch.zeros(config.generation_units, config.local_dim)
        )
        local_block_config = type(
            "LocalRasterBlockConfig",
            (),
            {
                "model_dim": config.local_dim,
                "heads": config.local_heads,
                "mlp_ratio": config.local_mlp_ratio,
                "dropout": config.local_dropout,
            },
        )()
        self.blocks = nn.ModuleList(
            [CausalVisualBlock(local_block_config) for _ in range(config.local_layers)]
        )
        self.output_norm = RMSNorm(config.local_dim)
        output_dim = config.mixtures * (1 + config.pixels_per_patch)
        self.output = nn.Linear(config.local_dim, output_dim)
        self._initialize()

    def _initialize(self) -> None:
        nn.init.normal_(self.start, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        for root in (self.condition, self.patch_input, self.blocks, self.output):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * self.config.local_layers)
        for block in self.blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def _validate_condition(
        self,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        target_patches: torch.Tensor,
    ) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != self.reader_config.model_dim:
            raise ValueError("V50 writer hidden state must be [N,reader_dim]")
        if anchor.shape != (hidden.shape[0], self.reader_config.field_dim):
            raise ValueError("V50 writer anchor does not align with hidden state")
        if target_patches.shape != (
            hidden.shape[0],
            self.config.generation_units,
            self.config.pixels_per_patch,
        ):
            raise ValueError("V50 writer units must contain coarse and detail rasters")

    def forward(
        self,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        target_patches: torch.Tensor,
    ) -> RasterMixtureOutput:
        self._validate_condition(hidden, anchor, target_patches)
        condition = self.condition(
            torch.cat((hidden.float(), anchor.float()), dim=-1).to(
                self.condition[1].weight.dtype
            )
        )
        previous = target_patches[:, :-1].to(self.patch_input.weight.dtype)
        prefix = self.patch_input(previous)
        start = self.start[None, None].expand(len(hidden), 1, -1)
        state = torch.cat((start, prefix), dim=1)
        state = state + condition[:, None] + self.position[None]
        for block in self.blocks:
            state = block(state)
        state = self.output_norm(state)
        raw = self.output(state).float()
        raw = raw.reshape(
            len(hidden),
            self.config.generation_units,
            self.config.mixtures,
            1 + self.config.pixels_per_patch,
        )
        return RasterMixtureOutput(
            mixture_logits=raw[..., 0],
            pixel_logits=raw[..., 1:],
            hidden_states=state,
        )

    @torch.no_grad()
    def generate(
        self,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        *,
        stochastic: bool,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0.0:
            raise ValueError("V50 sampling temperature must be positive")
        patches = torch.zeros(
            len(hidden),
            self.config.generation_units,
            self.config.pixels_per_patch,
            device=hidden.device,
            dtype=torch.float32,
        )
        log_probability = torch.zeros(len(hidden), device=hidden.device)
        rows = torch.arange(len(hidden), device=hidden.device)
        for index in range(self.config.generation_units):
            output = self(hidden, anchor, patches)
            mixture_logits = output.mixture_logits[:, index] / temperature
            if stochastic:
                component = torch.multinomial(
                    mixture_logits.softmax(dim=-1),
                    1,
                    generator=generator,
                ).squeeze(1)
            else:
                component = mixture_logits.argmax(dim=-1)
            pixel_logits = output.pixel_logits[rows, index, component] / temperature
            probability = pixel_logits.sigmoid()
            if stochastic:
                uniforms = torch.rand(
                    probability.shape,
                    device=probability.device,
                    generator=generator,
                )
                patch = (uniforms < probability).float()
            else:
                patch = (probability >= 0.5).float()
            patches[:, index] = patch
            component_log_probability = mixture_logits.log_softmax(dim=-1)[
                rows, component
            ]
            pixel_log_probability = -F.binary_cross_entropy_with_logits(
                pixel_logits,
                patch,
                reduction="none",
            ).sum(dim=-1)
            log_probability += component_log_probability + pixel_log_probability
        raster = micro_patches_to_raster(
            patches[:, 1:],
            cell_size=self.config.cell_size,
            patch_size=self.config.patch_size,
        )
        return raster, log_probability


class HierarchicalRasterLanguageModel(nn.Module):
    def __init__(
        self,
        reader_config: CanonicalGlyphLanguageConfig,
        config: HierarchicalRasterLanguageConfig,
    ) -> None:
        super().__init__()
        self.reader_config = reader_config
        self.config = config
        self.reader = CausalRasterReader(reader_config)
        self.writer = LocalRasterAutoregressor(reader_config, config)

    def load_v42_reader(self, state: Mapping[str, torch.Tensor]) -> None:
        self.reader.load_v42_state(state)

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.reader(context)

    def teacher_force(
        self,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        target_cells: torch.Tensor,
    ) -> RasterMixtureOutput:
        patches = raster_to_micro_patches(
            target_cells, patch_size=self.config.patch_size
        )
        units = raster_to_generation_units(target_cells, config=self.config)
        if not torch.equal(patches, units[:, 1:]):
            raise RuntimeError("V50 detail-unit construction is inconsistent")
        return self.writer(hidden, anchor, units)

    @torch.no_grad()
    def sample_next(
        self,
        context: torch.Tensor,
        *,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        language = self.language(context)
        raster, log_probability = self.writer.generate(
            language["hidden_states"][:, -1],
            language["anchor_fields"][:, -1],
            stochastic=stochastic,
            generator=generator,
            temperature=temperature,
        )
        return raster, {
            "log_probability": log_probability,
            "anchor_fields": language["anchor_fields"][:, -1],
        }


def hierarchical_raster_language_config_payload(
    config: HierarchicalRasterLanguageConfig,
) -> dict[str, Any]:
    return asdict(config)


def hierarchical_raster_language_config_from_payload(
    payload: Mapping[str, Any],
) -> HierarchicalRasterLanguageConfig:
    return HierarchicalRasterLanguageConfig(**dict(payload))


def hierarchical_raster_language_boundary_receipt(
    model: HierarchicalRasterLanguageModel,
) -> dict[str, Any]:
    return {
        "architecture": HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE,
        "primary_input": "continuous visible writing rasters",
        "primary_output": "autoregressively generated binary writing rasters",
        "text_tokenizer": False,
        "unicode_or_character_ids_in_model": False,
        "ocr_in_model": False,
        "glyph_vocabulary_or_candidate_bank_in_model": False,
        "vector_quantizer_or_codebook": False,
        "external_model_at_runtime": False,
        "patch_distribution": "learned mixture of Bernoulli raster fields",
        "global_units": "32x32 visible raster cells",
        "local_units": (
            f"{model.config.patch_size}x{model.config.patch_size} raw raster patches"
        ),
        "patches_per_cell": model.config.patches_per_cell,
        "coarse_generation_units": 1,
        "total_generation_units": model.config.generation_units,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reader_parameters": sum(
            parameter.numel() for parameter in model.reader.parameters()
        ),
        "writer_parameters": sum(
            parameter.numel() for parameter in model.writer.parameters()
        ),
    }

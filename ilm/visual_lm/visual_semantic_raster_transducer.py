from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from .visual_semantic_raster_data import (
    V32_ANSWER_CELL,
    V32_ARCHITECTURE,
    V32_MAX_ANSWER_CELLS,
    V32_MAX_PROMPT_PATCHES,
    V32_PROMPT_PATCH,
)


PIXEL_M4_REPOSITORY = "Team-PIXEL/pixel-m4"
PIXEL_M4_REVISION = "56bfcbf71e98f613ee00f8efb7a607bf0058f1e6"
PIXEL_M4_WEIGHT_SHA256 = (
    "6aa0642d46fe211727fefc5ac6e0bc28efa8511d1f0d9e5eee1987fa821600bc"
)
PIXEL_M4_CONFIG_SHA256 = (
    "90789708a8b064d848977d256ce4e3e20ba51f57d6803a59295bf401d514f83f"
)


@dataclass(frozen=True)
class VisualSemanticRasterConfig:
    prompt_patch_size: int = V32_PROMPT_PATCH
    maximum_prompt_patches: int = V32_MAX_PROMPT_PATCHES
    answer_cell_size: int = V32_ANSWER_CELL
    maximum_answer_cells: int = V32_MAX_ANSWER_CELLS
    reader_hidden_size: int = 768
    reader_layers: int = 12
    reader_heads: int = 12
    reader_intermediate_size: int = 3072
    reader_dropout: float = 0.1
    planner_dim: int = 512
    planner_layers: int = 6
    planner_heads: int = 8
    planner_mlp_dim: int = 1536
    planner_dropout: float = 0.05
    cell_retina_channels: int = 64
    target_width: int = 256
    target_blocks: int = 3
    latent_dim: int = 32
    decoder_width: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_mlp_dim: int = 768
    decoder_dropout: float = 0.05
    feedback_minimum_t: float = 0.65
    feedback_noise_probability: float = 0.90
    feedback_ground_truth_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.prompt_patch_size != V32_PROMPT_PATCH:
            raise ValueError("V32 fixes 16-pixel prompt patches")
        if not 8 <= self.maximum_prompt_patches <= V32_MAX_PROMPT_PATCHES:
            raise ValueError("V32 prompt patches must be in [8,192]")
        if self.answer_cell_size != V32_ANSWER_CELL:
            raise ValueError("V32 fixes 24-pixel answer cells")
        if not 1 <= self.maximum_answer_cells <= V32_MAX_ANSWER_CELLS:
            raise ValueError("V32 answer cells must be in [1,32]")
        if self.reader_hidden_size % self.reader_heads:
            raise ValueError("V32 reader width must divide into reader heads")
        if self.reader_layers < 1 or self.reader_intermediate_size < self.reader_hidden_size:
            raise ValueError("V32 reader configuration is invalid")
        if self.planner_dim % self.planner_heads or self.planner_layers < 1:
            raise ValueError("V32 planner configuration is invalid")
        if self.planner_mlp_dim < self.planner_dim:
            raise ValueError("V32 planner MLP is underspecified")
        if self.target_width < self.latent_dim or self.target_blocks < 1:
            raise ValueError("V32 target encoder is underspecified")
        if self.decoder_width % self.decoder_heads or self.decoder_layers < 1:
            raise ValueError("V32 raster decoder configuration is invalid")
        if self.decoder_mlp_dim < self.decoder_width:
            raise ValueError("V32 raster decoder MLP is underspecified")
        if not 4 <= self.latent_dim <= 128:
            raise ValueError("V32 continuous glyph state must be in [4,128]")
        for value in (self.reader_dropout, self.planner_dropout, self.decoder_dropout):
            if not 0.0 <= value < 1.0:
                raise ValueError("V32 dropout values must be in [0,1)")
        if not 0.0 < self.feedback_minimum_t <= 1.0:
            raise ValueError("V32 feedback minimum t must be in (0,1]")
        for value in (
            self.feedback_noise_probability,
            self.feedback_ground_truth_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("V32 feedback probabilities must be in [0,1]")

    @property
    def prompt_width(self) -> int:
        return self.prompt_patch_size * self.maximum_prompt_patches

    @property
    def planner_positions(self) -> int:
        return self.maximum_answer_cells + 1

    @property
    def production_reader(self) -> bool:
        return (
            self.reader_hidden_size == 768
            and self.reader_layers == 12
            and self.reader_heads == 12
            and self.reader_intermediate_size == 3072
            and self.prompt_patch_size == 16
        )


@dataclass
class VisualSemanticRasterOutput:
    target_states: torch.Tensor
    state_mean: torch.Tensor
    state_log_scale: torch.Tensor
    raster_logits: torch.Tensor
    feedback_cells: torch.Tensor
    stop_logits: torch.Tensor
    clean_hidden: torch.Tensor
    feedback_hidden: torch.Tensor


@dataclass
class VisualSemanticRasterGeneration:
    cells: torch.Tensor
    cell_mask: torch.Tensor
    lengths: torch.Tensor
    stop_probabilities: torch.Tensor
    latent_states: torch.Tensor

    def strips(self) -> torch.Tensor:
        batch, cells, channels, height, width = self.cells.shape
        return self.cells.permute(0, 2, 3, 1, 4).reshape(
            batch,
            channels,
            height,
            cells * width,
        )


class AnswerCellRetina(nn.Module):
    def __init__(self, config: VisualSemanticRasterConfig) -> None:
        super().__init__()
        base = config.cell_retina_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1),
            nn.GroupNorm(8 if base % 8 == 0 else 1, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8 if (base * 2) % 8 == 0 else 1, base * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(base * 2, config.planner_dim)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 5 or tuple(cells.shape[-3:]) != (1, 24, 24):
            raise ValueError("V32 answer retina expects [B,A,1,24,24]")
        batch, count = cells.shape[:2]
        hidden = self.features(cells.reshape(batch * count, 1, 24, 24))
        return self.output(hidden.flatten(1)).reshape(batch, count, -1)


class TargetResidualBlock(nn.Module):
    def __init__(self, width: int, context_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.context = nn.Linear(context_dim, width * 2)
        self.expand = nn.Linear(width, width * 2)
        self.contract = nn.Linear(width * 2, width)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.context(F.silu(context)).chunk(2, dim=-1)
        normalized = self.norm(hidden)
        modulated = normalized * (1.0 + 0.1 * gamma.tanh()) + beta
        return hidden + self.contract(F.silu(self.expand(modulated)))


class ContinuousGlyphTargetEncoder(nn.Module):
    def __init__(self, config: VisualSemanticRasterConfig) -> None:
        super().__init__()
        width = config.target_width
        base = config.cell_retina_channels
        self.cell_features = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.context_projection = nn.Linear(config.planner_dim, width)
        self.blocks = nn.ModuleList(
            TargetResidualBlock(width, width) for _ in range(config.target_blocks)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.latent_dim)

    def forward(self, cells: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 5 or tuple(cells.shape[-3:]) != (1, 24, 24):
            raise ValueError("V32 target encoder expects [B,A,1,24,24]")
        if context.shape[:2] != cells.shape[:2]:
            raise ValueError("V32 target context does not align with answer cells")
        batch, count = cells.shape[:2]
        hidden = self.cell_features(cells.reshape(batch * count, 1, 24, 24))
        hidden = hidden.flatten(1).reshape(batch, count, -1)
        condition = self.context_projection(context)
        for block in self.blocks:
            hidden = block(hidden, condition)
        states = self.output(self.output_norm(hidden))
        return F.layer_norm(states.float(), (states.shape[-1],)).to(states.dtype)


class CausalRasterDecoder(nn.Module):
    def __init__(self, config: VisualSemanticRasterConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.latent_dim, config.decoder_width)
        self.position = nn.Parameter(
            torch.empty(1, config.maximum_answer_cells, config.decoder_width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_width,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_mlp_dim,
            dropout=config.decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.decoder_width),
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.Linear(config.decoder_width, config.decoder_width),
            nn.SiLU(),
            nn.Linear(config.decoder_width, config.answer_cell_size**2),
        )
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or states.shape[-1] != self.config.latent_dim:
            raise ValueError("V32 raster decoder expects [B,A,latent_dim]")
        if not 1 <= states.shape[1] <= self.config.maximum_answer_cells:
            raise ValueError("V32 raster decoder received an invalid sequence length")
        hidden = self.input(states) + self.position[:, : states.shape[1]]
        hidden = self.blocks(hidden, mask=self._causal_mask(states.shape[1], states.device))
        logits = self.output(hidden)
        return logits.reshape(
            states.shape[0],
            states.shape[1],
            1,
            self.config.answer_cell_size,
            self.config.answer_cell_size,
        )


class VisualSemanticRasterTransducer(nn.Module):
    def __init__(self, config: VisualSemanticRasterConfig) -> None:
        super().__init__()
        self.config = config
        reader_config = ViTConfig(
            hidden_size=config.reader_hidden_size,
            num_hidden_layers=config.reader_layers,
            num_attention_heads=config.reader_heads,
            intermediate_size=config.reader_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=config.reader_dropout,
            attention_probs_dropout_prob=config.reader_dropout,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=(config.prompt_patch_size, config.prompt_width),
            patch_size=config.prompt_patch_size,
            num_channels=3,
            qkv_bias=True,
        )
        self.reader = ViTModel(reader_config, add_pooling_layer=False)
        self.memory_projection = nn.Linear(config.reader_hidden_size, config.planner_dim)
        self.cell_retina = AnswerCellRetina(config)
        self.answer_start = nn.Parameter(torch.empty(1, 1, config.planner_dim))
        self.answer_position = nn.Parameter(
            torch.empty(1, config.planner_positions, config.planner_dim)
        )
        planner_layer = nn.TransformerDecoderLayer(
            d_model=config.planner_dim,
            nhead=config.planner_heads,
            dim_feedforward=config.planner_mlp_dim,
            dropout=config.planner_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.planner = nn.TransformerDecoder(
            planner_layer,
            num_layers=config.planner_layers,
            norm=nn.LayerNorm(config.planner_dim),
        )
        self.target_encoder = ContinuousGlyphTargetEncoder(config)
        self.raster_decoder = CausalRasterDecoder(config)
        self.state_head = nn.Linear(config.planner_dim, config.latent_dim * 2)
        self.stop_head = nn.Linear(config.planner_dim, 1)
        self._reader_trainable_blocks: int | None = None
        nn.init.normal_(self.answer_start, std=0.02)
        nn.init.normal_(self.answer_position, std=0.02)

    @staticmethod
    def _validate_float(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V32 {name} must be a floating tensor")

    def _validate_prompt(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> None:
        self._validate_float("prompt_pixels", prompt_pixels)
        self._validate_float("prompt_mask", prompt_mask)
        expected = (3, self.config.prompt_patch_size, self.config.prompt_width)
        if prompt_pixels.ndim != 4 or tuple(prompt_pixels.shape[1:]) != expected:
            raise ValueError(f"V32 prompt pixels must end in {expected}")
        if prompt_mask.shape != (
            prompt_pixels.shape[0],
            self.config.maximum_prompt_patches,
        ):
            raise ValueError("V32 prompt mask does not align with prompt pixels")

    def _validate_answer(self, answer_cells: torch.Tensor) -> None:
        self._validate_float("answer_cells", answer_cells)
        expected = (
            self.config.maximum_answer_cells,
            1,
            self.config.answer_cell_size,
            self.config.answer_cell_size,
        )
        if answer_cells.ndim != 5 or tuple(answer_cells.shape[1:]) != expected:
            raise ValueError(f"V32 answer cells must end in {expected}")

    def freeze_reader(self) -> None:
        self._reader_trainable_blocks = 0
        self.reader.requires_grad_(False).eval()

    def unfreeze_reader_final_blocks(self, count: int = 2) -> None:
        if not 1 <= count <= len(self.reader.encoder.layer):
            raise ValueError("V32 reader block count is invalid")
        self.reader.requires_grad_(False)
        for block in self.reader.encoder.layer[-count:]:
            block.requires_grad_(True)
        self.reader.layernorm.requires_grad_(True)
        self._reader_trainable_blocks = count

    def train(self, mode: bool = True) -> VisualSemanticRasterTransducer:
        super().train(mode)
        if mode and self._reader_trainable_blocks is not None:
            self.reader.eval()
            if self._reader_trainable_blocks:
                for block in self.reader.encoder.layer[-self._reader_trainable_blocks :]:
                    block.train()
                self.reader.layernorm.train()
        return self

    def reader_trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.reader.parameters() if parameter.requires_grad)

    def encode_prompt(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_prompt(prompt_pixels, prompt_mask)
        normalized = (prompt_pixels.clamp(0, 1) - 0.5) / 0.5
        memory = self.reader(normalized).last_hidden_state
        memory = self.memory_projection(memory)
        cls_padding = torch.zeros(
            prompt_mask.shape[0],
            1,
            dtype=torch.bool,
            device=prompt_mask.device,
        )
        padding_mask = torch.cat((cls_padding, prompt_mask <= 0.0), dim=1)
        return memory, padding_mask

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def plan_from_memory(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        previous_cells: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_float("previous_cells", previous_cells)
        if previous_cells.ndim != 5 or tuple(previous_cells.shape[-3:]) != (
            1,
            self.config.answer_cell_size,
            self.config.answer_cell_size,
        ):
            raise ValueError("V32 previous cells must end in [1,24,24]")
        if previous_cells.shape[1] > self.config.maximum_answer_cells:
            raise ValueError("V32 previous-cell sequence is too long")
        batch = memory.shape[0]
        if previous_cells.shape[0] != batch:
            raise ValueError("V32 prompt and answer batches do not align")
        if previous_cells.shape[1]:
            previous = self.cell_retina(previous_cells)
            planner_input = torch.cat((self.answer_start.expand(batch, -1, -1), previous), dim=1)
        else:
            planner_input = self.answer_start.expand(batch, -1, -1)
        length = planner_input.shape[1]
        planner_input = planner_input + self.answer_position[:, :length]
        return self.planner(
            planner_input,
            memory,
            tgt_mask=self._causal_mask(length, planner_input.device),
            memory_key_padding_mask=memory_padding_mask,
        )

    def plan(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
        previous_cells: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_padding_mask = self.encode_prompt(prompt_pixels, prompt_mask)
        return self.plan_from_memory(memory, memory_padding_mask, previous_cells)

    def predict_state(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_scale = self.state_head(hidden).chunk(2, dim=-1)
        return mean.float(), log_scale.float().clamp(-4.0, 2.0)

    def encode_target_states(
        self,
        answer_cells: torch.Tensor,
        planner_context: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_answer(answer_cells)
        if planner_context.shape != (
            answer_cells.shape[0],
            answer_cells.shape[1],
            self.config.planner_dim,
        ):
            raise ValueError("V32 target planner context has an invalid shape")
        return self.target_encoder(answer_cells, planner_context)

    def perturb_target_states(
        self,
        states: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        minimum_t: float | None = None,
    ) -> torch.Tensor:
        minimum = self.config.feedback_minimum_t if minimum_t is None else float(minimum_t)
        if not 0.0 < minimum <= 1.0:
            raise ValueError("V32 feedback minimum t must be in (0,1]")
        noise = torch.randn(
            states.shape,
            dtype=states.dtype,
            device=states.device,
            generator=generator,
        )
        times = minimum + (1.0 - minimum) * torch.rand(
            states.shape[:2],
            dtype=states.dtype,
            device=states.device,
            generator=generator,
        )
        selected = torch.rand(
            states.shape[:2],
            dtype=states.dtype,
            device=states.device,
            generator=generator,
        ) < self.config.feedback_noise_probability
        selected = selected & (active_mask > 0)
        mixed = times[..., None] * states + (1.0 - times[..., None]) * noise
        return torch.where(selected[..., None], mixed, states)

    def forward(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_cells: torch.Tensor,
        answer_mask: torch.Tensor,
        *,
        feedback_mode: str = "decoded",
        generator: torch.Generator | None = None,
    ) -> VisualSemanticRasterOutput:
        self._validate_answer(answer_cells)
        self._validate_float("answer_mask", answer_mask)
        if answer_mask.shape != answer_cells.shape[:2]:
            raise ValueError("V32 answer mask does not align with answer cells")
        if feedback_mode not in {"decoded", "clean"}:
            raise ValueError("V32 feedback mode must be decoded or clean")

        memory, memory_padding_mask = self.encode_prompt(prompt_pixels, prompt_mask)
        clean_hidden = self.plan_from_memory(memory, memory_padding_mask, answer_cells)
        target_states = self.encode_target_states(
            answer_cells,
            clean_hidden[:, :-1].detach(),
        )
        if feedback_mode == "decoded":
            decoder_states = self.perturb_target_states(
                target_states,
                answer_mask,
                generator=generator,
            )
            raster_logits = self.raster_decoder(decoder_states)
            feedback_cells = raster_logits.sigmoid().detach()
            if self.training and self.config.feedback_ground_truth_probability > 0:
                replacement = torch.rand(
                    answer_mask.shape,
                    device=answer_mask.device,
                    generator=generator,
                ) < self.config.feedback_ground_truth_probability
                replacement = replacement & (answer_mask > 0)
                bounded_ground_truth = (
                    answer_cells
                    + torch.randn(
                        answer_cells.shape,
                        device=answer_cells.device,
                        dtype=answer_cells.dtype,
                        generator=generator,
                    )
                    * 0.02
                ).clamp(0, 1)
                feedback_cells = torch.where(
                    replacement[..., None, None, None],
                    bounded_ground_truth,
                    feedback_cells,
                )
        else:
            raster_logits = self.raster_decoder(target_states)
            feedback_cells = answer_cells.detach()

        feedback_hidden = self.plan_from_memory(
            memory,
            memory_padding_mask,
            feedback_cells,
        )
        state_mean, state_log_scale = self.predict_state(feedback_hidden[:, :-1])
        stop_logits = self.stop_head(feedback_hidden).squeeze(-1).float()
        return VisualSemanticRasterOutput(
            target_states=target_states.float(),
            state_mean=state_mean,
            state_log_scale=state_log_scale,
            raster_logits=raster_logits.float(),
            feedback_cells=feedback_cells.float(),
            stop_logits=stop_logits,
            clean_hidden=clean_hidden,
            feedback_hidden=feedback_hidden,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_pixels: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        stop_threshold: float = 0.5,
        minimum_cells: int = 1,
        maximum_cells: int | None = None,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> VisualSemanticRasterGeneration:
        if not 0.0 < stop_threshold < 1.0:
            raise ValueError("V32 stop threshold must be in (0,1)")
        maximum = self.config.maximum_answer_cells if maximum_cells is None else int(maximum_cells)
        if not 1 <= minimum_cells <= maximum <= self.config.maximum_answer_cells:
            raise ValueError("V32 generation length bounds are invalid")
        memory, memory_padding_mask = self.encode_prompt(prompt_pixels, prompt_mask)
        batch = prompt_pixels.shape[0]
        cells = prompt_pixels.new_zeros(
            batch,
            maximum,
            1,
            self.config.answer_cell_size,
            self.config.answer_cell_size,
        )
        mask = prompt_pixels.new_zeros(batch, maximum)
        stop_probabilities = prompt_pixels.new_zeros(batch, maximum + 1)
        latent_states = prompt_pixels.new_zeros(batch, maximum, self.config.latent_dim)
        active = torch.ones(batch, dtype=torch.bool, device=prompt_pixels.device)
        lengths = torch.full(
            (batch,),
            maximum,
            dtype=torch.long,
            device=prompt_pixels.device,
        )
        prefix_cells = cells[:, :0]
        prefix_states = latent_states[:, :0]

        for step in range(maximum + 1):
            hidden = self.plan_from_memory(memory, memory_padding_mask, prefix_cells)
            current = hidden[:, -1]
            stop_probability = self.stop_head(current).squeeze(-1).sigmoid()
            stop_probabilities[:, step] = stop_probability
            if step >= minimum_cells:
                stopping = active & (stop_probability > stop_threshold)
                lengths[stopping] = step
                active = active & ~stopping
            if step == maximum or not bool(active.any()):
                break

            mean, log_scale = self.predict_state(current)
            if sample:
                noise = torch.randn(
                    mean.shape,
                    device=mean.device,
                    dtype=mean.dtype,
                    generator=generator,
                )
                state = mean + log_scale.exp() * noise
            else:
                state = mean
            state = F.layer_norm(state, (self.config.latent_dim,))
            state = torch.where(active[:, None], state, torch.zeros_like(state))
            latent_states[:, step] = state.to(latent_states.dtype)
            prefix_states = latent_states[:, : step + 1]
            raster_logits = self.raster_decoder(prefix_states)
            current_cell = raster_logits[:, -1].sigmoid().to(cells.dtype)
            current_cell = torch.where(
                active[:, None, None, None],
                current_cell,
                torch.zeros_like(current_cell),
            )
            cells[:, step] = current_cell
            mask[:, step] = active.to(mask.dtype)
            prefix_cells = cells[:, : step + 1]

        return VisualSemanticRasterGeneration(
            cells=cells,
            cell_mask=mask,
            lengths=lengths,
            stop_probabilities=stop_probabilities,
            latent_states=latent_states,
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pixel_m4_checkpoint(
    path: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> Path:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            PIXEL_M4_REPOSITORY,
            "pytorch_model.bin",
            revision=PIXEL_M4_REVISION,
            local_files_only=local_files_only,
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = file_sha256(resolved)
    if digest != PIXEL_M4_WEIGHT_SHA256:
        raise ValueError(
            f"V32 PIXEL-M4 checkpoint hash mismatch: expected {PIXEL_M4_WEIGHT_SHA256}, got {digest}"
        )
    return resolved


def load_pixel_m4_reader(
    model: VisualSemanticRasterTransducer,
    checkpoint: str | Path,
) -> dict[str, Any]:
    if not model.config.production_reader:
        raise ValueError("V32 PIXEL-M4 weights require the production reader shape")
    path = resolve_pixel_m4_checkpoint(checkpoint)
    source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(source, Mapping):
        raise TypeError("V32 PIXEL-M4 checkpoint must contain a state mapping")
    selected = {
        key.removeprefix("vit."): value
        for key, value in source.items()
        if isinstance(key, str) and key.startswith("vit.")
    }
    upstream_reader_parameters = sum(value.numel() for value in selected.values())
    position_key = "embeddings.position_embeddings"
    required_positions = model.config.maximum_prompt_patches + 1
    if position_key not in selected or selected[position_key].shape[1] < required_positions:
        raise ValueError("V32 PIXEL-M4 checkpoint has insufficient prompt positions")
    selected[position_key] = selected[position_key][:, :required_positions].clone()
    missing, unexpected = model.reader.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"V32 PIXEL-M4 mapping failed: missing={missing}, unexpected={unexpected}"
        )
    deployed_reader_parameters = sum(value.numel() for value in selected.values())
    return {
        "repository": PIXEL_M4_REPOSITORY,
        "revision": PIXEL_M4_REVISION,
        "checkpoint": str(path),
        "sha256": PIXEL_M4_WEIGHT_SHA256,
        "selected_state_prefix": "vit.*",
        "upstream_reader_parameters": upstream_reader_parameters,
        "deployed_reader_parameters": deployed_reader_parameters,
        "selected_positions": required_positions,
        "missing_keys": [],
        "unexpected_keys": [],
    }


def visual_semantic_raster_boundary_receipt(
    model: VisualSemanticRasterTransducer,
) -> dict[str, Any]:
    parameter_names = tuple(name for name, _ in model.named_parameters())
    forbidden_fragments = (
        "token_embed",
        "word_embed",
        "vocab",
        "unicode",
        "character_id",
        "glyph_id",
        "codebook",
        "candidate_bank",
    )
    forbidden_names = [
        name
        for name in parameter_names
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    ]
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "architecture": V32_ARCHITECTURE,
        "model_config": asdict(model.config),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_cap": 125_000_000,
        "parameter_cap_pass": total < 125_000_000,
        "primary_input": "prompt raster",
        "primary_output": "generated answer raster",
        "continuous_glyph_state_dimension": model.config.latent_dim,
        "forbidden_parameter_names": forbidden_names,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_logits": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
        "generation_is_autoregressive_raster_feedback": True,
    }

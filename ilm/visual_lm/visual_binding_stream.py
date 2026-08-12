from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retinal_topology_router import ChannelLayerNorm2d


QUERY_AWARE_ROUTE = "query_aware"
QUERY_BLIND_ROUTE = "query_blind"
ROUTE_MODES = (QUERY_AWARE_ROUTE, QUERY_BLIND_ROUTE)


class SpatialVisualRetina(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...

    def forward_with_field(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class VisualBindingStreamConfig:
    fovea_size: int = 32
    prompt_length: int = 6
    answer_length: int = 1
    visual_dim: int = 192
    spatial_channels: int = 192
    field_size: int = 4
    model_dim: int = 256
    transformer_blocks: int = 4
    attention_heads: int = 8
    feedforward_dim: int = 768
    writer_hidden_channels: int = 128
    writer_context_dim: int = 128
    writer_blocks: int = 3
    writer_patch_size: int = 12
    writer_stride: int = 8
    writer_padding: int = 2
    dropout: float = 0.05
    route_mode: str = QUERY_AWARE_ROUTE

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("binding fovea size must be a multiple of eight")
        if self.prompt_length != 6:
            raise ValueError("V22 fixes prompt_length=6")
        if self.answer_length != 1:
            raise ValueError("V22 fixes answer_length=1")
        if self.visual_dim != self.spatial_channels:
            raise ValueError("V22 global and local visual dimensions must match")
        if self.field_size != 4:
            raise ValueError("V22 fixes a 4x4 retinal field")
        if self.model_dim % self.attention_heads:
            raise ValueError("model dimension must be divisible by attention heads")
        if self.transformer_blocks < 1 or self.writer_blocks < 1:
            raise ValueError("binding and writer blocks must be positive")
        if self.writer_stride * self.field_size != self.fovea_size:
            raise ValueError("writer stride and field size must span the fovea")
        output_size = (
            (self.field_size - 1) * self.writer_stride
            - 2 * self.writer_padding
            + self.writer_patch_size
        )
        if output_size != self.fovea_size:
            raise ValueError("overlap writer geometry does not produce the fovea size")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"route_mode must be one of {ROUTE_MODES}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class VisualTransformerBlock(nn.Module):
    def __init__(self, config: VisualBindingStreamConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim),
        )
        self.dropout = config.dropout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = hidden + F.dropout(
            attended,
            p=self.dropout,
            training=self.training,
        )
        update = self.feedforward(self.feedforward_norm(hidden))
        return hidden + F.dropout(update, p=self.dropout, training=self.training)


class PointwiseWriterBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm2d(channels)
        self.input = nn.Conv2d(channels, channels * 2, 1)
        self.context = nn.Linear(context_dim, channels * 4)
        self.output = nn.Conv2d(channels * 2, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        field: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input(F.silu(self.norm(field)))
        scale, shift = self.context(F.silu(context)).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.output(self.dropout(F.silu(hidden)))
        return field + hidden


def positive_overlap_window(size: int) -> torch.Tensor:
    if size < 2:
        raise ValueError("overlap window size must be at least two")
    phase = math.pi * (torch.arange(size, dtype=torch.float32) + 0.5) / size
    one_dimensional = phase.sin()
    return torch.outer(one_dimensional, one_dimensional)


class OverlapLocalWriter(nn.Module):
    """Blend shared local patches without a global spatial drawing seed."""

    def __init__(self, config: VisualBindingStreamConfig) -> None:
        super().__init__()
        self.config = config
        self.context = nn.Sequential(
            nn.LayerNorm(config.visual_dim + config.model_dim),
            nn.Linear(
                config.visual_dim + config.model_dim,
                config.writer_context_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.writer_context_dim, config.writer_context_dim),
            nn.LayerNorm(config.writer_context_dim),
        )
        self.source_norm = ChannelLayerNorm2d(config.spatial_channels)
        self.source_projection = nn.Conv2d(
            config.spatial_channels,
            config.writer_hidden_channels,
            1,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            PointwiseWriterBlock(
                config.writer_hidden_channels,
                config.writer_context_dim,
                config.dropout,
            )
            for _ in range(config.writer_blocks)
        )
        self.output_norm = ChannelLayerNorm2d(config.writer_hidden_channels)
        self.patch_output = nn.Conv2d(
            config.writer_hidden_channels,
            config.writer_patch_size**2,
            1,
        )
        window = positive_overlap_window(config.writer_patch_size)
        columns = window.reshape(1, -1, 1).repeat(
            1,
            1,
            config.field_size**2,
        )
        overlap = F.fold(
            columns,
            output_size=(config.fovea_size, config.fovea_size),
            kernel_size=config.writer_patch_size,
            stride=config.writer_stride,
            padding=config.writer_padding,
        )
        self.register_buffer("patch_window", window, persistent=True)
        self.register_buffer("overlap_weights", overlap, persistent=True)

    def logits_with_trace(
        self,
        visual: torch.Tensor,
        field: torch.Tensor,
        query_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = visual.shape[0]
        if tuple(visual.shape) != (batch, self.config.visual_dim):
            raise ValueError("writer visual state has the wrong shape")
        expected_field = (
            batch,
            self.config.spatial_channels,
            self.config.field_size,
            self.config.field_size,
        )
        if tuple(field.shape) != expected_field:
            raise ValueError(f"writer field must have shape {expected_field}")
        if tuple(query_context.shape) != (batch, self.config.model_dim):
            raise ValueError("writer query context has the wrong shape")
        context = self.context(torch.cat((visual, query_context), dim=-1))
        hidden = self.source_projection(self.source_norm(field.float()))
        for block in self.blocks:
            hidden = block(hidden, context)
        patch_logits = self.patch_output(F.silu(self.output_norm(hidden))).float()
        columns = patch_logits.flatten(2)
        window = self.patch_window.reshape(1, -1, 1)
        numerator = F.fold(
            columns * window,
            output_size=(self.config.fovea_size, self.config.fovea_size),
            kernel_size=self.config.writer_patch_size,
            stride=self.config.writer_stride,
            padding=self.config.writer_padding,
        )
        logits = numerator / self.overlap_weights.clamp_min(1e-8)
        return logits, {
            "local_source": field,
            "global_context": context,
            "patch_logits": patch_logits,
            "overlap_weights": self.overlap_weights,
            "combined_logits": logits,
        }

    def forward(
        self,
        visual: torch.Tensor,
        field: torch.Tensor,
        query_context: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.logits_with_trace(visual, field, query_context)
        return logits


class VisualBindingStream(nn.Module):
    """Read an image prompt and emit a requested answer image stream."""

    def __init__(
        self,
        config: VisualBindingStreamConfig,
        retina: SpatialVisualRetina,
    ) -> None:
        super().__init__()
        self.config = config
        self.retina = retina
        if isinstance(retina, nn.Module):
            retina.requires_grad_(False).eval()
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.model_dim),
        )
        self.positions = nn.Parameter(
            torch.zeros(1, config.prompt_length, config.model_dim)
        )
        nn.init.normal_(self.positions, std=0.02)
        self.blocks = nn.ModuleList(
            VisualTransformerBlock(config)
            for _ in range(config.transformer_blocks)
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.selection_query = nn.Linear(config.model_dim, config.model_dim)
        self.selection_key = nn.Linear(config.model_dim, config.model_dim)
        self.global_value = nn.Linear(config.visual_dim, config.visual_dim)
        self.field_value = nn.Conv2d(
            config.spatial_channels,
            config.spatial_channels,
            1,
        )
        self.null_visual = nn.Parameter(torch.zeros(config.visual_dim))
        self.null_field = nn.Parameter(
            torch.zeros(config.spatial_channels, config.field_size, config.field_size)
        )
        nn.init.normal_(self.null_visual, std=0.02)
        nn.init.normal_(self.null_field, std=0.02)
        self.writer = OverlapLocalWriter(config)

    def train(self, mode: bool = True) -> "VisualBindingStream":
        super().train(mode)
        if isinstance(self.retina, nn.Module):
            self.retina.eval()
        return self

    def _validate_prompt(self, prompt: torch.Tensor) -> None:
        expected = (
            prompt.shape[0],
            self.config.prompt_length,
            1,
            self.config.fovea_size,
            self.config.fovea_size,
        )
        if tuple(prompt.shape) != expected:
            raise ValueError(f"prompt stream must have shape {expected}")
        if not torch.is_floating_point(prompt):
            raise TypeError("visual binding accepts continuous image tensors only")

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (images.shape[0], 1, self.config.fovea_size, self.config.fovea_size)
        if tuple(images.shape) != expected:
            raise ValueError(f"retinal images must have shape {expected}")
        visual, field = self.retina.forward_with_field(images.float())
        return (
            F.normalize(visual.float(), dim=-1),
            F.normalize(field.float(), dim=1),
        )

    def _prompt_states(
        self,
        prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = prompt.shape[:2]
        visual, field = self.encode_images(
            prompt.reshape(batch * length, *prompt.shape[2:])
        )
        visual = visual.reshape(batch, length, self.config.visual_dim)
        field = field.reshape(
            batch,
            length,
            self.config.spatial_channels,
            self.config.field_size,
            self.config.field_size,
        )
        if self.config.route_mode == QUERY_BLIND_ROUTE:
            visual = torch.cat(
                (
                    visual[:, :-1],
                    F.normalize(self.null_visual, dim=0)[None, None].expand(
                        batch,
                        1,
                        -1,
                    ),
                ),
                dim=1,
            )
            field = torch.cat(
                (
                    field[:, :-1],
                    F.normalize(self.null_field, dim=0)[None, None].expand(
                        batch,
                        1,
                        -1,
                        -1,
                        -1,
                    ),
                ),
                dim=1,
            )
        return visual, field

    def logits_with_trace(
        self,
        prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_prompt(prompt)
        visual, field = self._prompt_states(prompt)
        hidden = self.visual_projection(visual) + self.positions
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        query_context = hidden[:, -1]
        query = self.selection_query(query_context)
        keys = self.selection_key(hidden)
        scores = torch.einsum("bd,btd->bt", query, keys) / math.sqrt(
            self.config.model_dim
        )
        attention = scores.softmax(dim=1)
        global_values = self.global_value(visual)
        bound_visual = F.normalize(
            torch.einsum("bt,btd->bd", attention, global_values),
            dim=-1,
        )
        batch, length = field.shape[:2]
        field_values = self.field_value(
            field.reshape(
                batch * length,
                self.config.spatial_channels,
                self.config.field_size,
                self.config.field_size,
            )
        ).reshape_as(field)
        bound_field = F.normalize(
            torch.einsum("bt,btchw->bchw", attention, field_values),
            dim=1,
        )
        logits, writer_trace = self.writer.logits_with_trace(
            bound_visual,
            bound_field,
            query_context,
        )
        return logits[:, None], {
            "prompt_visual": visual,
            "prompt_field": field,
            "hidden": hidden,
            "query_context": query_context,
            "selection_scores": scores,
            "selection_attention": attention,
            "bound_visual": bound_visual,
            "bound_field": bound_field,
            **{f"writer_{key}": value for key, value in writer_trace.items()},
        }

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        logits, _ = self.logits_with_trace(prompt)
        return logits.sigmoid()

    def write_reference(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        visual, field = self.encode_images(reference)
        query_context = visual.new_zeros(reference.shape[0], self.config.model_dim)
        logits, trace = self.writer.logits_with_trace(
            visual,
            field,
            query_context,
        )
        return logits, trace

    def boundary_receipt(self) -> dict[str, bool | str]:
        return {
            "architecture": "visual-binding-stream-v1",
            "route_mode": self.config.route_mode,
            "input_is_continuous_image": True,
            "output_is_continuous_image": True,
            "uses_strings": False,
            "uses_token_ids": False,
            "uses_unicode_ids": False,
            "uses_ocr": False,
            "uses_character_labels": False,
            "uses_operation_ids": False,
            "uses_slot_indices": False,
            "uses_visual_codebook": False,
            "uses_glyph_lookup": False,
            "uses_external_language_model": False,
            "retina_trainable": False,
        }


def _edge_field(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    horizontal = F.conv2d(image.float(), kernel_x.float(), padding=1)
    vertical = F.conv2d(image.float(), kernel_y.float(), padding=1)
    return torch.cat((horizontal, vertical), dim=1)


def _pixel_f1_rows(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_binary = generated >= 0.5
    target_binary = target >= 0.5
    true_positive = (generated_binary & target_binary).sum(dim=(1, 2, 3)).float()
    denominator = (
        generated_binary.sum(dim=(1, 2, 3))
        + target_binary.sum(dim=(1, 2, 3))
    ).clamp_min(1)
    return 2.0 * true_positive / denominator


def _soft_dice_rows(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (generated * target).sum(dim=(1, 2, 3)) + 1.0
    denominator = generated.sum(dim=(1, 2, 3)) + target.sum(
        dim=(1, 2, 3)
    ) + 1.0
    return numerator / denominator


def _topology_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    stroke_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ink = logits.sigmoid()
    pixel_weights = 1.0 + stroke_weight * target.float()
    bce = (
        pixel_weights
        * F.binary_cross_entropy_with_logits(
            logits.float(),
            target.float(),
            reduction="none",
        )
    ).mean()
    dice = _soft_dice_rows(ink, target.float()).mean()
    pixel_l1 = F.l1_loss(ink, target.float())
    edge_l1 = F.l1_loss(_edge_field(ink), _edge_field(target.float()))
    loss = bce + (1.0 - dice) + 0.50 * pixel_l1 + 0.25 * edge_l1
    return loss, {
        "bce": bce.detach(),
        "soft_dice": dice.detach(),
        "pixel_l1": pixel_l1.detach(),
        "edge_l1": edge_l1.detach(),
        "pixel_f1": _pixel_f1_rows(ink, target).mean().detach(),
    }


def _field_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = F.normalize(first.float(), dim=1)
    second = F.normalize(second.float(), dim=1)
    return (1.0 - (first * second).sum(dim=1)).mean()


def visual_binding_stream_loss(
    model: VisualBindingStream,
    prompt: torch.Tensor,
    target: torch.Tensor,
    counterfactual_prompt: torch.Tensor,
    counterfactual_target: torch.Tensor,
    oracle_reference: torch.Tensor,
    counterfactual_oracle_reference: torch.Tensor,
    *,
    stroke_weight: float = 4.0,
    oracle_weight: float = 0.50,
    visual_state_weight: float = 0.20,
    field_state_weight: float = 0.15,
    reread_visual_weight: float = 0.10,
    reread_field_weight: float = 0.10,
    attention_entropy_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prompts = torch.cat((prompt, counterfactual_prompt), dim=0)
    targets = torch.cat((target, counterfactual_target), dim=0).float()
    references = torch.cat(
        (oracle_reference, counterfactual_oracle_reference),
        dim=0,
    )
    logits_stream, trace = model.logits_with_trace(prompts)
    logits = logits_stream[:, 0]
    generated = logits.sigmoid()
    topology, topology_metrics = _topology_loss(
        logits,
        targets,
        stroke_weight=stroke_weight,
    )
    with torch.no_grad():
        target_visual, target_field = model.encode_images(targets)
    bound_visual_loss = (
        1.0 - (trace["bound_visual"] * target_visual).sum(dim=-1)
    ).mean()
    bound_field_loss = _field_distance(trace["bound_field"], target_field)
    generated_visual, generated_field = model.encode_images(generated)
    reread_visual_loss = (
        1.0 - (generated_visual * target_visual).sum(dim=-1)
    ).mean()
    reread_field_loss = _field_distance(generated_field, target_field)
    attention = trace["selection_attention"].clamp_min(1e-8)
    attention_entropy = -(attention * attention.log()).sum(dim=1).mean()

    oracle_logits, oracle_trace = model.write_reference(references)
    oracle_topology, oracle_metrics = _topology_loss(
        oracle_logits,
        targets,
        stroke_weight=stroke_weight,
    )
    total = (
        topology
        + visual_state_weight * bound_visual_loss
        + field_state_weight * bound_field_loss
        + reread_visual_weight * reread_visual_loss
        + reread_field_weight * reread_field_loss
        + attention_entropy_weight * attention_entropy
        + oracle_weight * oracle_topology
    )
    metrics = {
        "loss": total.detach(),
        **{f"full_{key}": value for key, value in topology_metrics.items()},
        **{f"oracle_{key}": value for key, value in oracle_metrics.items()},
        "bound_visual_loss": bound_visual_loss.detach(),
        "bound_field_loss": bound_field_loss.detach(),
        "reread_visual_loss": reread_visual_loss.detach(),
        "reread_field_loss": reread_field_loss.detach(),
        "attention_entropy": attention_entropy.detach(),
        "attention_max": attention.max(dim=1).values.mean().detach(),
        "ink_fraction": generated.mean().detach(),
    }
    return total, metrics, {
        **trace,
        "generated": generated,
        "oracle_generated": oracle_logits.sigmoid(),
        "oracle_overlap_weights": oracle_trace["overlap_weights"],
    }


@torch.no_grad()
def encode_identity_bank(
    model: VisualBindingStream,
    images: torch.Tensor,
) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError("identity bank images must have [objects, views, 1, H, W]")
    objects, views = images.shape[:2]
    visual, _ = model.encode_images(images.flatten(0, 1))
    return visual.reshape(objects, views, -1)


@torch.no_grad()
def visual_binding_batch_metrics(
    model: VisualBindingStream,
    batch: dict[str, Any],
    *,
    bank_visual: torch.Tensor | None = None,
    bank_characters: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    prompt = batch["prompt"]
    counterfactual_prompt = batch["counterfactual_prompt"]
    target = batch["target"].float()
    counterfactual_target = batch["counterfactual_target"].float()
    distractor = batch["distractor_target"].float()
    counterfactual_distractor = batch["counterfactual_distractor_target"].float()
    prompts = torch.cat((prompt, counterfactual_prompt), dim=0)
    targets = torch.cat((target, counterfactual_target), dim=0)
    distractors = torch.cat((distractor, counterfactual_distractor), dim=0)
    logits_stream, trace = model.logits_with_trace(prompts)
    generated = logits_stream[:, 0].sigmoid()
    generated_visual, _ = model.encode_images(generated)
    target_visual, _ = model.encode_images(targets)
    distractor_visual, _ = model.encode_images(distractors)
    target_cosine_rows = (generated_visual * target_visual).sum(dim=-1)
    distractor_cosine_rows = (generated_visual * distractor_visual).sum(dim=-1)
    choice = target_cosine_rows > distractor_cosine_rows
    batch_size = prompt.shape[0]
    paired_choice = choice[:batch_size] & choice[batch_size:]

    query_shuffled = prompt.clone()
    query_shuffled[:, -1] = query_shuffled[:, -1].roll(1, dims=0)
    shuffled_generated = model(query_shuffled)[:, 0]
    shuffled_visual, _ = model.encode_images(shuffled_generated)
    original_generated = generated[:batch_size]
    counterfactual_generated = generated[batch_size:]

    operation_visual, _ = model.encode_images(prompt[:, 4])
    query_visual, _ = model.encode_images(
        torch.cat((prompt[:, 5], counterfactual_prompt[:, 5]), dim=0)
    )
    operation_visual = torch.cat((operation_visual, operation_visual), dim=0)
    target_margin_operation = target_cosine_rows - (
        generated_visual * operation_visual
    ).sum(dim=-1)
    target_margin_query = target_cosine_rows - (
        generated_visual * query_visual
    ).sum(dim=-1)

    heldout = torch.tensor(
        [bool(item["heldout_combination"]) for item in batch["metadata"]],
        device=prompt.device,
    )
    target_prompt_indices = torch.tensor(
        [int(item["target_prompt_index"]) for item in batch["metadata"]]
        + [
            int(item["counterfactual_target_prompt_index"])
            for item in batch["metadata"]
        ],
        device=prompt.device,
    )
    target_attention = trace["selection_attention"].gather(
        1,
        target_prompt_indices[:, None],
    )[:, 0]

    oracle_references = torch.cat(
        (
            batch["oracle_reference"],
            batch["counterfactual_oracle_reference"],
        ),
        dim=0,
    )
    oracle_logits, _ = model.write_reference(oracle_references)
    oracle_generated = oracle_logits.sigmoid()
    metrics: dict[str, torch.Tensor] = {
        "examples": generated.new_tensor(float(generated.shape[0])),
        "pairs": generated.new_tensor(float(batch_size)),
        "heldout_pairs": heldout.float().sum(),
        "seen_pairs": (~heldout).float().sum(),
        "binary_choice_accuracy": choice.float().mean(),
        "counterfactual_switch_accuracy": paired_choice.float().mean(),
        "heldout_combination_switch_accuracy": (
            paired_choice[heldout].float().mean()
            if heldout.any()
            else generated.new_tensor(float("nan"))
        ),
        "seen_combination_switch_accuracy": (
            paired_choice[~heldout].float().mean()
            if (~heldout).any()
            else generated.new_tensor(float("nan"))
        ),
        "target_cosine": target_cosine_rows.mean(),
        "distractor_cosine": distractor_cosine_rows.mean(),
        "pixel_f1": _pixel_f1_rows(generated, targets).mean(),
        "pixel_l1": F.l1_loss(generated, targets),
        "oracle_pixel_f1": _pixel_f1_rows(oracle_generated, targets).mean(),
        "oracle_pixel_l1": F.l1_loss(oracle_generated, targets),
        "paired_output_pixel_l1": F.l1_loss(
            original_generated,
            counterfactual_generated,
        ),
        "query_shuffled_output_pixel_l1": F.l1_loss(
            original_generated,
            shuffled_generated,
        ),
        "target_margin_over_operation": target_margin_operation.mean(),
        "target_margin_over_query_label": target_margin_query.mean(),
        "target_attention_mass": target_attention.mean(),
        "attention_max": trace["selection_attention"].max(dim=1).values.mean(),
        "frozen_images_instantiated": generated.new_zeros(()),
    }
    if bank_visual is not None:
        if bank_characters is None or bank_visual.ndim != 3:
            raise ValueError("identity bank requires characters and [objects, views, dim]")
        if bank_visual.shape[0] != len(bank_characters):
            raise ValueError("identity bank characters and visual states differ")
        scores = torch.einsum(
            "bd,nvd->bnv",
            generated_visual,
            F.normalize(bank_visual.float(), dim=-1),
        ).amax(dim=-1)
        shuffled_scores = torch.einsum(
            "bd,nvd->bnv",
            shuffled_visual,
            F.normalize(bank_visual.float(), dim=-1),
        ).amax(dim=-1)
        bank_index = {character: index for index, character in enumerate(bank_characters)}
        target_indices = torch.tensor(
            [bank_index[item["target_character"]] for item in batch["metadata"]]
            + [
                bank_index[item["counterfactual_target_character"]]
                for item in batch["metadata"]
            ],
            device=prompt.device,
        )
        metrics["identity_top1"] = (
            scores.argmax(dim=1) == target_indices
        ).float().mean()
        metrics["query_shuffled_identity_top1"] = (
            shuffled_scores.argmax(dim=1) == target_indices[:batch_size]
        ).float().mean()
    return metrics


def visual_binding_config_payload(
    config: VisualBindingStreamConfig,
) -> dict[str, Any]:
    return asdict(config)


def visual_binding_config_from_payload(
    payload: dict[str, Any],
) -> VisualBindingStreamConfig:
    return VisualBindingStreamConfig(**payload)

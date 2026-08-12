from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


RELATION_AWARE_ROUTE = "relation_aware"
QUERY_BLIND_ROUTE = "query_blind"
OPERATION_BLIND_ROUTE = "operation_blind"
ROUTE_MODES = (
    RELATION_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
)


class VisualRetina(Protocol):
    def forward_with_field(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class CanonicalizerResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(8, channels)
        self.conv_1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm_2 = nn.GroupNorm(8, channels)
        self.conv_2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        update = self.conv_1(F.silu(self.norm_1(image)))
        update = self.conv_2(F.silu(self.norm_2(update)))
        return image + update


class VisualCanonicalizer(nn.Module):
    """Preserve glyph topology while normalizing a source image's visual face."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 32, 3, padding=1)
        self.encoder_32 = nn.Sequential(
            CanonicalizerResidualBlock(32),
            CanonicalizerResidualBlock(32, dilation=2),
        )
        self.down_16 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.encoder_16 = nn.Sequential(
            CanonicalizerResidualBlock(64),
            CanonicalizerResidualBlock(64, dilation=2),
        )
        self.down_8 = nn.Conv2d(64, 96, 4, stride=2, padding=1)
        self.middle = nn.Sequential(
            CanonicalizerResidualBlock(96),
            CanonicalizerResidualBlock(96, dilation=2),
            CanonicalizerResidualBlock(96),
        )
        self.up_16 = nn.Conv2d(160, 64, 3, padding=1)
        self.decoder_16 = nn.Sequential(
            CanonicalizerResidualBlock(64),
            CanonicalizerResidualBlock(64),
        )
        self.up_32 = nn.Conv2d(96, 32, 3, padding=1)
        self.decoder_32 = nn.Sequential(
            CanonicalizerResidualBlock(32),
            CanonicalizerResidualBlock(32),
        )
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    def _validate(self, image: torch.Tensor) -> None:
        expected = (image.shape[0], 1, 32, 32)
        if tuple(image.shape) != expected:
            raise ValueError(f"canonicalizer input must have shape {expected}")
        if not torch.is_floating_point(image):
            raise TypeError("canonicalizer accepts continuous image tensors only")

    def logits_with_trace(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate(image)
        level_32 = self.encoder_32(self.stem(image))
        level_16 = self.encoder_16(self.down_16(level_32))
        middle = self.middle(self.down_8(level_16))
        decoded_16 = F.interpolate(
            middle,
            size=level_16.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_16 = self.decoder_16(
            self.up_16(torch.cat((decoded_16, level_16), dim=1))
        )
        decoded_32 = F.interpolate(
            decoded_16,
            size=level_32.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_32 = self.decoder_32(
            self.up_32(torch.cat((decoded_32, level_32), dim=1))
        )
        residual_logits = self.output(decoded_32)
        input_logits = 2.0 * (image.float() - 0.5)
        logits = residual_logits.float() + input_logits
        return logits, {
            "input_logits": input_logits,
            "residual_logits": residual_logits,
            "level_32": level_32,
            "level_16": level_16,
            "middle": middle,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _ = self.logits_with_trace(image)
        return logits.sigmoid()

    def boundary_receipt(self) -> dict[str, bool | str]:
        return {
            "architecture": "visual-canonicalizer-v23",
            "input_is_continuous_image": True,
            "output_is_continuous_image": True,
            "uses_strings": False,
            "uses_token_ids": False,
            "uses_unicode_ids": False,
            "uses_ocr": False,
            "uses_character_labels": False,
            "uses_font_ids": False,
            "uses_visual_codebook": False,
            "uses_glyph_lookup": False,
            "uses_external_language_model": False,
        }


@dataclass(frozen=True)
class VisualRelationCircuitConfig:
    fovea_size: int = 32
    prompt_length: int = 6
    answer_length: int = 1
    visual_dim: int = 192
    operation_hidden_dim: int = 128
    minimum_temperature: float = 1.0
    maximum_temperature: float = 32.0
    initial_temperature: float = 8.0
    route_mode: str = RELATION_AWARE_ROUTE

    def __post_init__(self) -> None:
        if self.fovea_size != 32:
            raise ValueError("V23 fixes the fovea size to 32")
        if self.prompt_length != 6 or self.answer_length != 1:
            raise ValueError("V23 fixes six prompt frames and one answer frame")
        if self.visual_dim < 1 or self.operation_hidden_dim < 1:
            raise ValueError("V23 dimensions must be positive")
        if not (
            self.minimum_temperature
            < self.initial_temperature
            < self.maximum_temperature
        ):
            raise ValueError("initial temperature must lie inside its bounds")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"unknown V23 route mode {self.route_mode!r}")


class VisualRelationCircuit(nn.Module):
    """Compose visual equality and operation states, then route source pixels."""

    def __init__(
        self,
        config: VisualRelationCircuitConfig,
        retina: VisualRetina,
        canonicalizer: VisualCanonicalizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.retina = retina
        self.canonicalizer = canonicalizer
        if isinstance(retina, nn.Module):
            retina.requires_grad_(False).eval()
        canonicalizer.requires_grad_(False).eval()
        self.null_query = nn.Parameter(torch.empty(config.visual_dim))
        self.null_operation = nn.Parameter(torch.empty(config.visual_dim))
        nn.init.normal_(self.null_query, std=0.02)
        nn.init.normal_(self.null_operation, std=0.02)
        self.operation_reader = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.operation_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.operation_hidden_dim, 1),
        )
        fraction = (
            config.initial_temperature - config.minimum_temperature
        ) / (config.maximum_temperature - config.minimum_temperature)
        self.raw_temperature = nn.Parameter(
            torch.tensor(math.log(fraction / (1.0 - fraction)))
        )

    def train(self, mode: bool = True) -> "VisualRelationCircuit":
        super().train(mode)
        if isinstance(self.retina, nn.Module):
            self.retina.eval()
        self.canonicalizer.eval()
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
            raise ValueError(f"relation prompt must have shape {expected}")
        if not torch.is_floating_point(prompt):
            raise TypeError("visual relation circuit accepts image tensors only")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        expected = (
            images.shape[0],
            1,
            self.config.fovea_size,
            self.config.fovea_size,
        )
        if tuple(images.shape) != expected:
            raise ValueError(f"retinal input must have shape {expected}")
        visual, _ = self.retina.forward_with_field(images.float())
        return F.normalize(visual.float(), dim=-1)

    def temperature(self) -> torch.Tensor:
        span = self.config.maximum_temperature - self.config.minimum_temperature
        return self.config.minimum_temperature + span * self.raw_temperature.sigmoid()

    def logits_with_trace(
        self, prompt: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_prompt(prompt)
        batch = prompt.shape[0]
        visual = self.encode_images(
            prompt.reshape(batch * self.config.prompt_length, 1, 32, 32)
        ).reshape(batch, self.config.prompt_length, self.config.visual_dim)
        query = visual[:, 5]
        operation = visual[:, 4]
        if self.config.route_mode == QUERY_BLIND_ROUTE:
            query = F.normalize(self.null_query, dim=0)[None].expand(batch, -1)
        if self.config.route_mode == OPERATION_BLIND_ROUTE:
            operation = F.normalize(self.null_operation, dim=0)[None].expand(
                batch, -1
            )

        # Two-slot relation algebra is accumulated in FP64 so exchanging the
        # visible pairs remains inside the preregistered 1e-6 output tolerance.
        label_states = visual[:, (0, 2)]
        match_cosine = torch.einsum(
            "bd,bid->bi", query.double(), label_states.double()
        )
        match_weights = (
            self.temperature().double() * match_cosine
        ).softmax(dim=1)
        same_gate = self.operation_reader(operation).sigmoid().double()
        route_weights = same_gate * match_weights + (1.0 - same_gate) * (
            1.0 - match_weights
        )
        source_images = prompt[:, (1, 3)]
        routed_source = torch.einsum(
            "bi,bichw->bchw", route_weights, source_images.double()
        ).float()
        logits, writer_trace = self.canonicalizer.logits_with_trace(routed_source)
        return logits[:, None], {
            "prompt_visual": visual,
            "query_visual": query,
            "operation_visual": operation,
            "match_cosine": match_cosine,
            "match_weights": match_weights,
            "same_gate": same_gate,
            "route_weights": route_weights,
            "source_images": source_images,
            "routed_source": routed_source,
            **{f"writer_{key}": value for key, value in writer_trace.items()},
        }

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        logits, _ = self.logits_with_trace(prompt)
        return logits.sigmoid()

    def boundary_receipt(self) -> dict[str, bool | str]:
        return {
            "architecture": "visual-relation-circuit-v23",
            "route_mode": self.config.route_mode,
            "input_is_continuous_image": True,
            "output_is_continuous_image": True,
            "uses_frame_positions": True,
            "uses_strings": False,
            "uses_token_ids": False,
            "uses_unicode_ids": False,
            "uses_ocr": False,
            "uses_character_labels": False,
            "uses_operation_ids": False,
            "uses_slot_indices": False,
            "uses_target_indices": False,
            "uses_visual_codebook": False,
            "uses_glyph_lookup": False,
            "uses_external_language_model": False,
            "retina_trainable": False,
            "canonicalizer_trainable": False,
        }


def _edge_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_x = generated[..., :, 1:] - generated[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    generated_y = generated[..., 1:, :] - generated[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return (generated_x - target_x).abs().mean() + (
        generated_y - target_y
    ).abs().mean()


def pixel_f1_rows(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_ink = generated > 0.5
    target_ink = target > 0.5
    dimensions = tuple(range(1, generated.ndim))
    intersection = (generated_ink & target_ink).sum(dim=dimensions).float()
    denominator = generated_ink.sum(dim=dimensions) + target_ink.sum(
        dim=dimensions
    )
    return (2.0 * intersection + 1e-6) / (denominator.float() + 1e-6)


def topology_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    stroke_weight: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generated = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        weight=1.0 + (stroke_weight - 1.0) * target,
    )
    dimensions = tuple(range(1, target.ndim))
    dice = 1.0 - (
        2.0 * (generated * target).sum(dim=dimensions) + 1.0
    ) / ((generated + target).sum(dim=dimensions) + 1.0)
    pixel_l1 = (generated - target).abs().mean()
    edge_l1 = _edge_loss(generated, target)
    loss = bce + dice.mean() + 0.5 * pixel_l1 + 0.2 * edge_l1
    return loss, {
        "bce": bce.detach(),
        "dice": dice.mean().detach(),
        "pixel_l1": pixel_l1.detach(),
        "edge_l1": edge_l1.detach(),
        "pixel_f1": pixel_f1_rows(generated, target).mean().detach(),
        "ink_fraction": generated.mean().detach(),
    }


def canonicalizer_loss(
    model: VisualCanonicalizer,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    stroke_weight: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, _ = model.logits_with_trace(source)
    return topology_loss(logits, target, stroke_weight=stroke_weight)


def relation_circuit_config_payload(
    config: VisualRelationCircuitConfig,
) -> dict[str, int | float | str]:
    return asdict(config)


def relation_circuit_config_from_payload(
    payload: dict[str, int | float | str],
) -> VisualRelationCircuitConfig:
    return VisualRelationCircuitConfig(**payload)

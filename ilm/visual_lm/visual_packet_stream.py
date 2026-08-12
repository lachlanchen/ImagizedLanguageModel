from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_relation_circuit import VisualCanonicalizer


PACKET_AWARE_ROUTE = "packet_aware"
HEADER_BLIND_ROUTE = "header_blind"
QUERY_BLIND_ROUTE = "query_blind"
OPERATION_BLIND_ROUTE = "operation_blind"
HISTORY_BLIND_ROUTE = "history_blind"
ROUTE_MODES = (
    PACKET_AWARE_ROUTE,
    HEADER_BLIND_ROUTE,
    QUERY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
    HISTORY_BLIND_ROUTE,
)
PAIR_ROLE = 0
OPERATION_ROLE = 1
QUERY_ROLE = 2


class VisualRetina(Protocol):
    def forward_with_field(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class VisualPacketStreamConfig:
    fovea_size: int = 32
    visual_dim: int = 192
    frames_per_packet: int = 3
    minimum_packets: int = 5
    maximum_packets: int = 8
    answer_length: int = 2
    minimum_role_temperature: float = 1.0
    maximum_role_temperature: float = 32.0
    initial_role_temperature: float = 8.0
    route_mode: str = PACKET_AWARE_ROUTE

    def __post_init__(self) -> None:
        if self.fovea_size != 32:
            raise ValueError("V24 fixes the fovea size to 32")
        if self.visual_dim != 192:
            raise ValueError("V24 fixes the visual dimension to 192")
        if self.frames_per_packet != 3:
            raise ValueError("V24 fixes three frames per packet")
        if self.minimum_packets != 5 or self.maximum_packets != 8:
            raise ValueError("V24 fixes five through eight packets")
        if self.answer_length != 2:
            raise ValueError("V24 fixes two answer frames")
        if not (
            self.minimum_role_temperature
            < self.initial_role_temperature
            < self.maximum_role_temperature
        ):
            raise ValueError("initial role temperature must lie inside bounds")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"unknown V24 route mode {self.route_mode!r}")


def _straight_through_topk(
    logits: torch.Tensor,
    *,
    selections: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("role logits must have shape [batch, packets]")
    if not 1 <= selections <= logits.shape[1]:
        raise ValueError("invalid hard-role selection count")
    soft = logits.softmax(dim=1)
    indices = logits.topk(selections, dim=1, sorted=False).indices
    hard = torch.zeros_like(soft).scatter_(1, indices, 1.0 / selections)
    return hard + soft - soft.detach(), indices


class VisualPacketRereadStream(nn.Module):
    """Locate visible packets, write frame one, and reread it for frame two."""

    def __init__(
        self,
        config: VisualPacketStreamConfig,
        retina: VisualRetina,
        canonicalizer: VisualCanonicalizer,
        operation_reader: nn.Module,
        *,
        match_temperature: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(match_temperature) or match_temperature <= 0:
            raise ValueError("match temperature must be finite and positive")
        self.config = config
        self.retina = retina
        self.canonicalizer = canonicalizer
        self.operation_reader = operation_reader
        if isinstance(retina, nn.Module):
            retina.requires_grad_(False).eval()
        canonicalizer.requires_grad_(False).eval()
        operation_reader.requires_grad_(False).eval()

        self.role_prototypes = nn.Parameter(torch.empty(3, config.visual_dim))
        self.null_header = nn.Parameter(torch.empty(config.visual_dim))
        self.null_query = nn.Parameter(torch.empty(config.visual_dim))
        self.null_operation = nn.Parameter(torch.empty(config.visual_dim))
        self.null_history = nn.Parameter(torch.empty(config.visual_dim))
        nn.init.normal_(self.role_prototypes, std=0.02)
        for parameter in (
            self.null_header,
            self.null_query,
            self.null_operation,
            self.null_history,
        ):
            nn.init.normal_(parameter, std=0.02)
        fraction = (
            config.initial_role_temperature - config.minimum_role_temperature
        ) / (config.maximum_role_temperature - config.minimum_role_temperature)
        raw = math.log(fraction / (1.0 - fraction))
        self.raw_role_temperatures = nn.Parameter(torch.full((3,), raw))
        self.register_buffer(
            "match_temperature",
            torch.tensor(float(match_temperature)),
            persistent=True,
        )

    def train(self, mode: bool = True) -> "VisualPacketRereadStream":
        super().train(mode)
        if isinstance(self.retina, nn.Module):
            self.retina.eval()
        self.canonicalizer.eval()
        self.operation_reader.eval()
        return self

    def _validate_prompt(self, prompt: torch.Tensor) -> None:
        if prompt.ndim != 5:
            raise ValueError("packet prompt must have five dimensions")
        batch, frames, channels, height, width = prompt.shape
        if batch < 1 or channels != 1 or (height, width) != (32, 32):
            raise ValueError(
                "packet prompt must have shape [B,T,1,32,32] with B positive"
            )
        minimum = self.config.minimum_packets * self.config.frames_per_packet
        maximum = self.config.maximum_packets * self.config.frames_per_packet
        if not minimum <= frames <= maximum or frames % 3:
            raise ValueError(
                f"packet prompt length must be packet aligned in [{minimum},{maximum}]"
            )
        if not torch.is_floating_point(prompt):
            raise TypeError("visual packet stream accepts image tensors only")

    def _validate_override(
        self,
        override: torch.Tensor,
        batch: int,
    ) -> None:
        if tuple(override.shape) != (batch, 1, 32, 32):
            raise ValueError("frame-one override must have shape [B,1,32,32]")
        if not torch.is_floating_point(override):
            raise TypeError("frame-one override must be a continuous image")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 32, 32):
            raise ValueError("retinal input must have shape [N,1,32,32]")
        visual, _ = self.retina.forward_with_field(images.float())
        return F.normalize(visual.float(), dim=-1)

    def role_temperatures(self) -> torch.Tensor:
        span = (
            self.config.maximum_role_temperature
            - self.config.minimum_role_temperature
        )
        return self.config.minimum_role_temperature + span * (
            self.raw_role_temperatures.sigmoid()
        )

    @staticmethod
    def _weighted_sum(
        weights: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bp,bp...->b...", weights.double(), values.double()
        ).float()

    @staticmethod
    def _gather_packets(
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_indices = torch.arange(values.shape[0], device=values.device)[:, None]
        return values[batch_indices, indices]

    def _role_route(
        self,
        header_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = header_states.shape[0]
        if self.config.route_mode == HEADER_BLIND_ROUTE:
            header_states = F.normalize(self.null_header, dim=0)[None, None].expand(
                batch, header_states.shape[1], -1
            )
        prototypes = F.normalize(self.role_prototypes, dim=-1)
        logits = torch.einsum(
            "bpd,rd->bpr", header_states, prototypes
        ) * self.role_temperatures()[None, None]
        pair_weights, pair_indices = _straight_through_topk(
            logits[:, :, PAIR_ROLE], selections=2
        )
        operation_weights, operation_indices = _straight_through_topk(
            logits[:, :, OPERATION_ROLE], selections=1
        )
        query_weights, query_indices = _straight_through_topk(
            logits[:, :, QUERY_ROLE], selections=1
        )
        return logits, {
            "header_visual": header_states,
            "pair_role_weights": pair_weights,
            "operation_role_weights": operation_weights,
            "query_role_weights": query_weights,
            "pair_indices": pair_indices,
            "operation_indices": operation_indices,
            "query_indices": query_indices,
        }

    def logits_with_trace(
        self,
        prompt: torch.Tensor,
        *,
        first_frame_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_prompt(prompt)
        batch, frames = prompt.shape[:2]
        packets = frames // self.config.frames_per_packet
        if first_frame_override is not None:
            self._validate_override(first_frame_override, batch)

        visual = self.encode_images(prompt.reshape(batch * frames, 1, 32, 32))
        visual = visual.reshape(batch, packets, 3, self.config.visual_dim)
        header_states = visual[:, :, 0]
        content_a_states = visual[:, :, 1]
        content_b_states = visual[:, :, 2]
        content_a_images = prompt.reshape(batch, packets, 3, 1, 32, 32)[:, :, 1]
        content_b_images = prompt.reshape(batch, packets, 3, 1, 32, 32)[:, :, 2]

        role_logits, role_trace = self._role_route(header_states)
        pair_indices = role_trace["pair_indices"]
        pair_weights = role_trace["pair_role_weights"]
        query = self._weighted_sum(
            role_trace["query_role_weights"], content_a_states
        )
        operation = self._weighted_sum(
            role_trace["operation_role_weights"], content_a_states
        )
        if self.config.route_mode == QUERY_BLIND_ROUTE:
            query = F.normalize(self.null_query, dim=0)[None].expand(batch, -1)
        if self.config.route_mode == OPERATION_BLIND_ROUTE:
            operation = F.normalize(self.null_operation, dim=0)[None].expand(
                batch, -1
            )

        pair_label_states = self._gather_packets(content_a_states, pair_indices)
        pair_glyph_states = self._gather_packets(content_b_states, pair_indices)
        pair_label_images = self._gather_packets(content_a_images, pair_indices)
        pair_glyph_images = self._gather_packets(content_b_images, pair_indices)
        selected_pair_prior = self._gather_packets(pair_weights, pair_indices)
        selected_pair_prior = selected_pair_prior / selected_pair_prior.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

        match_cosine = torch.einsum(
            "bd,bkd->bk", query.double(), pair_label_states.double()
        )
        match_logits = self.match_temperature.double() * match_cosine + (
            selected_pair_prior.double().clamp_min(1e-8).log()
        )
        match_weights = match_logits.softmax(dim=1)
        same_gate = self.operation_reader(operation).sigmoid().double()
        route_weights = same_gate * match_weights + (1.0 - same_gate) * (
            1.0 - match_weights
        )
        routed_source = self._weighted_sum(route_weights, pair_glyph_images)
        first_logits, first_writer_trace = self.canonicalizer.logits_with_trace(
            routed_source
        )
        first_frame = first_logits.sigmoid()

        reread_frame = (
            first_frame if first_frame_override is None else first_frame_override
        )
        history = self.encode_images(reread_frame)
        if self.config.route_mode == HISTORY_BLIND_ROUTE:
            history = F.normalize(self.null_history, dim=0)[None].expand(batch, -1)
        history_cosine = torch.einsum(
            "bd,bkd->bk", history.double(), pair_glyph_states.double()
        )
        history_logits = self.match_temperature.double() * history_cosine + (
            selected_pair_prior.double().clamp_min(1e-8).log()
        )
        history_weights = history_logits.softmax(dim=1)
        routed_label = self._weighted_sum(history_weights, pair_label_images)
        second_logits, second_writer_trace = self.canonicalizer.logits_with_trace(
            routed_label
        )

        output_logits = torch.stack((first_logits, second_logits), dim=1)
        return output_logits, {
            "prompt_visual": visual,
            "role_logits": role_logits,
            **role_trace,
            "query_visual": query,
            "operation_visual": operation,
            "pair_label_visual": pair_label_states,
            "pair_glyph_visual": pair_glyph_states,
            "match_cosine": match_cosine,
            "match_weights": match_weights,
            "same_gate": same_gate,
            "route_weights": route_weights,
            "routed_source": routed_source,
            "generated_first_frame": first_frame,
            "reread_frame": reread_frame,
            "history_visual": history,
            "history_cosine": history_cosine,
            "history_weights": history_weights,
            "routed_label": routed_label,
            "routed_query_image": self._weighted_sum(
                role_trace["query_role_weights"], content_a_images
            ),
            "routed_operation_image": self._weighted_sum(
                role_trace["operation_role_weights"], content_a_images
            ),
            "routed_pair_label_mean": self._weighted_sum(
                pair_weights, content_a_images
            ),
            "routed_pair_glyph_mean": self._weighted_sum(
                pair_weights, content_b_images
            ),
            **{
                f"first_writer_{key}": value
                for key, value in first_writer_trace.items()
            },
            **{
                f"second_writer_{key}": value
                for key, value in second_writer_trace.items()
            },
        }

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        logits, _ = self.logits_with_trace(prompt)
        return logits.sigmoid()

    def rollout_with_first_frame(
        self,
        prompt: torch.Tensor,
        first_frame: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.logits_with_trace(
            prompt, first_frame_override=first_frame
        )
        return logits.sigmoid()

    def boundary_receipt(self) -> dict[str, bool | str]:
        return {
            "architecture": "visual-packet-reread-stream-v24",
            "route_mode": self.config.route_mode,
            "input_is_continuous_image_stream": True,
            "output_is_continuous_image_stream": True,
            "uses_absolute_frame_roles": False,
            "uses_relative_packet_offsets": True,
            "uses_visible_packet_headers": True,
            "uses_padding_mask": False,
            "uses_active_lengths": False,
            "uses_strings": False,
            "uses_token_ids": False,
            "uses_unicode_ids": False,
            "uses_ocr": False,
            "uses_character_labels": False,
            "uses_operation_ids": False,
            "uses_role_ids_as_input": False,
            "uses_packet_indices_as_input": False,
            "uses_target_indices": False,
            "uses_visual_codebook": False,
            "uses_glyph_lookup": False,
            "uses_external_language_model": False,
            "retina_trainable": False,
            "canonicalizer_trainable": False,
            "operation_reader_trainable": False,
            "rereads_generated_pixels": True,
        }


def visual_packet_stream_config_payload(
    config: VisualPacketStreamConfig,
) -> dict[str, int | float | str]:
    return asdict(config)


def visual_packet_stream_config_from_payload(
    payload: dict[str, int | float | str],
) -> VisualPacketStreamConfig:
    return VisualPacketStreamConfig(**payload)

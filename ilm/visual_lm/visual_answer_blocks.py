"""A normalized posterior over variable-length blocks in observed writing."""

from __future__ import annotations

import math

import torch
from torch import nn

from .exact_raster_density_v54 import ExactRasterDensityConfig
from .visual_copy_language import (
    VisualCopyLanguage,
    VisualCopyReaderConfig,
    raster_signatures,
)

ARCHITECTURE = "visual-answer-blocks-v58"


class VisualAnswerBlocks(VisualCopyLanguage):
    maximum_answer_cells = 16

    def __init__(self, reader_config, density_config, *, block_enabled=True):
        super().__init__(reader_config, density_config, copy_enabled=True)
        self.block_enabled = bool(block_enabled)
        d = reader_config.model_dim
        for name in (
            "block_start_query",
            "block_start_key",
            "block_end_query",
            "block_end_key",
        ):
            layer = nn.Linear(d, d, bias=False)
            layer.requires_grad_(block_enabled)
            setattr(self, name, layer)
        self.block_length_bias = nn.Parameter(
            torch.zeros(self.maximum_answer_cells), requires_grad=block_enabled
        )

    def initialize_block_heads(self):
        for layer in (self.block_start_query, self.block_end_query):
            layer.load_state_dict(self.copy_query.state_dict())
        for layer in (self.block_start_key, self.block_end_key):
            layer.load_state_dict(self.copy_key.state_dict())

    def block_distribution(self, prompt_images, prompt_lengths):
        states = self.reader.language(prompt_images)["hidden_states"].float()
        if prompt_lengths.shape != (len(states),) or not bool(
            ((prompt_lengths > 0) & (prompt_lengths <= states.shape[1])).all()
        ):
            raise ValueError("invalid prompt packing")
        b = torch.arange(len(states), device=states.device)
        final = states[b, prompt_lengths - 1]
        start = torch.einsum(
            "bd,bld->bl", self.block_start_query(final), self.block_start_key(states)
        ).float() / math.sqrt(states.shape[-1])
        end = torch.einsum(
            "bd,bld->bl", self.block_end_query(final), self.block_end_key(states)
        ).float() / math.sqrt(states.shape[-1])
        starts = torch.arange(states.shape[1], device=states.device)[:, None]
        ends = (
            starts + torch.arange(self.maximum_answer_cells, device=states.device)[None]
        )
        scores = (
            start[:, :, None]
            + end[:, ends.clamp(max=states.shape[1] - 1)]
            + self.block_length_bias
        )
        valid = ends[None] < prompt_lengths[:, None, None]
        logp = (
            scores.masked_fill(~valid, -torch.inf)
            .flatten(1)
            .log_softmax(-1)
            .reshape_as(scores)
        )
        return logp

    def block_nll(
        self, logp, source_images, prompt_lengths, target_images, target_lengths
    ):
        """Exact image-block mass, marginalizing all matching source positions."""
        if not bool(
            ((target_lengths > 0) & (target_lengths <= self.maximum_answer_cells)).all()
        ):
            raise ValueError("unsupported target block length")
        source = raster_signatures(source_images)
        target = raster_signatures(target_images)
        starts = torch.arange(source.shape[1], device=source.device)
        steps = torch.arange(target.shape[1], device=source.device)
        addresses = (starts[:, None] + steps[None]).clamp(max=source.shape[1] - 1)
        observed = source[:, addresses]
        equal = (observed == target[:, None]).all(-1)
        equal = equal | (steps[None, None] >= target_lengths[:, None, None])
        matches = equal.all(-1) & (
            starts[None] + target_lengths[:, None] <= prompt_lengths[:, None]
        )
        if not bool(matches.any(-1).all()):
            raise ValueError("reference image block absent from visible prompt")
        lengths = (target_lengths - 1)[:, None, None].expand(-1, source.shape[1], 1)
        reference_logp = logp.gather(2, lengths).squeeze(-1)
        return -reference_logp.masked_fill(~matches, -torch.inf).logsumexp(-1)

    @torch.no_grad()
    def generate_block(self, prompt):
        if prompt.ndim != 4 or not 1 <= len(prompt) <= 480:
            raise ValueError("native prompt must be [1..480,1,32,32]")
        prompt = (prompt >= 0.5).float()
        logp = self.block_distribution(
            prompt[None], torch.tensor([len(prompt)], device=prompt.device)
        )[0]
        probability = logp.exp().cpu().double()
        # Anonymous addresses are derived from this prompt's pixels only. No
        # persistent writing inventory, string or reference answer is consulted.
        keys = [tuple(row) for row in raster_signatures(prompt).cpu().tolist()]
        mass = {}
        representative = {}
        for start in range(len(prompt)):
            for length in range(
                1, min(self.maximum_answer_cells, len(prompt) - start) + 1
            ):
                key = tuple(keys[start : start + length])
                if key not in mass:
                    mass[key] = 0.0
                    representative[key] = (start, length)
                mass[key] += float(probability[start, length - 1])
        chosen = max(mass, key=mass.get)
        start, length = representative[chosen]
        return {
            "images": prompt[start : start + length].clone(),
            "stopped": True,
            "capped": False,
            "events": [
                {
                    "route": "observed_block",
                    "source_start": start,
                    "source_end": start + length,
                    "marginal_block_probability": mass[chosen],
                }
            ],
            "termination": "predicted image-block end boundary",
        }

    def boundary_receipt(self):
        receipt = super().boundary_receipt()
        receipt.update(
            architecture=ARCHITECTURE,
            block_enabled=self.block_enabled,
            reading_probability="normalized dynamic observed-image-block distribution",
            continuation_probability="learned stop hazard and copy/exact-raster mixture",
            reading_limitation="extractive; answer must be in current prompt",
            runtime_parameters=sum(p.numel() for p in self.parameters()),
        )
        return receipt


def model_from_foundation(checkpoint, device, *, block_enabled):
    model = VisualAnswerBlocks(
        VisualCopyReaderConfig(**checkpoint["reader_config"]),
        ExactRasterDensityConfig(**checkpoint["model_config"]),
        block_enabled=block_enabled,
    )
    result = model.load_state_dict(
        checkpoint.get("ema_model", checkpoint.get("model")), strict=False
    )
    expected = {
        "block_start_query.weight",
        "block_start_key.weight",
        "block_end_query.weight",
        "block_end_key.weight",
        "block_length_bias",
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError(f"foundation mismatch: {result}")
    model.initialize_block_heads()
    return model.to(device)


def model_from_checkpoint(checkpoint, device):
    if checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("not a V58 checkpoint")
    model = VisualAnswerBlocks(
        VisualCopyReaderConfig(**checkpoint["reader_config"]),
        ExactRasterDensityConfig(**checkpoint["model_config"]),
        block_enabled=checkpoint["arm"] == "block",
    )
    model.load_state_dict(
        checkpoint.get("ema_model", checkpoint.get("model")), strict=True
    )
    return model.to(device).requires_grad_(False).eval()

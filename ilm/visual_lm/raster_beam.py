"""Beam search through raw binary pixels; no inventory of written forms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RasterBeamResult:
    images: torch.Tensor
    log_probability: torch.Tensor
    beams: torch.Tensor
    beam_log_probabilities: torch.Tensor


@torch.no_grad()
def raster_beam_search(model, hidden, fields, *, beam_width=8):
    """Approximate the mode of the model's exact 32×32 conditional raster law.

    States and scores always describe the emitted pixels. Equal scores prefer
    ink, then the earlier parent, matching the original greedy >=0.5 rule at
    width one. Row states depend only on completed earlier rows and can be
    gathered along beam parents while the current row is being expanded.
    """
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam width must be a positive integer")
    if getattr(model, "training", False):
        raise ValueError("raster beam search requires evaluation mode")
    if hidden.ndim != 2 or fields.ndim != 2 or len(hidden) != len(fields):
        raise ValueError("conditions must have matching [batch, features] shapes")
    if hidden.device != fields.device or not len(hidden):
        raise ValueError("nonempty conditions must share a device")
    batch, device = len(hidden), hidden.device
    width = beam_width
    h = hidden[:, None].expand(-1, width, -1).reshape(batch * width, -1)
    f = fields[:, None].expand(-1, width, -1).reshape(batch * width, -1)
    pixels = torch.zeros(batch * width, 32, 32, device=device)
    scores = torch.full((batch, width), -torch.inf, device=device)
    scores[:, 0] = 0
    offsets = torch.arange(batch, device=device)[:, None] * width
    for row in range(32):
        states, plan = model.row_states(pixels[:, : row + 1], h, f)
        condition = states[:, -1]
        bias = model.config.plan_logit_scale * plan[:, -1]
        for column in range(32):
            logits = (bias + model.pixel_density(pixels[:, row], condition))[
                :, column
            ].float()
            logits = logits.reshape(batch, width)
            # Ink first gives deterministic, exact-probability tie breaking.
            branches = torch.stack(
                (F.logsigmoid(logits), F.logsigmoid(-logits)), dim=-1
            )
            candidates = (scores[:, :, None] + branches).flatten(1)
            ranked, indices = torch.sort(
                candidates, descending=True, stable=True, dim=-1
            )
            scores, indices = ranked[:, :width], indices[:, :width]
            parents = torch.div(indices, 2, rounding_mode="floor")
            gather = (offsets + parents).flatten()
            pixels = pixels.index_select(0, gather)
            condition = condition.index_select(0, gather)
            bias = bias.index_select(0, gather)
            pixels[:, row, column] = (1 - indices.remainder(2)).flatten().float()
    beams = pixels.reshape(batch, width, 1, 32, 32)
    return RasterBeamResult(beams[:, 0], scores[:, 0], beams, scores)

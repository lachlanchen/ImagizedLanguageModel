"""Propose distinct person-shaped source spans and ground them with a pictured question."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.book_participant_reader import (
    WIDTH,
    VisualContext,
    end_values,
    valid_spans,
)


class PersonDetector(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim), nn.GELU())
        self.context = VisualContext(dim)
        self.start = nn.Linear(dim, WIDTH)
        self.end = nn.Linear(dim, WIDTH)
        self.left = nn.Linear(dim, 16)
        self.right = nn.Linear(dim, 16)

    def forward(self, pixels, page):
        h = self.context(self.ink(pixels)[page], page != 0)
        # Gather an end state for every (start, width) combination.
        tail = F.pad(self.right(h), (0, 0, 0, WIDTH - 1)).unfold(1, WIDTH, 1)
        agreement = (self.left(h).unsqueeze(-1) * tail).sum(2) / 4
        end = (
            F.pad(self.end(h), (0, 0, 0, WIDTH - 1))
            .unfold(1, WIDTH, 1)
            .diagonal(dim1=2, dim2=3)
        )
        return (self.start(h) + end + agreement).masked_fill(~valid_spans(page), -1e4)


def propose(logits, groups, k=16):
    flat = logits.flatten()
    ranked = [(max(flat[i] for i in spans), name) for name, spans in groups.items()]
    # Stable ordering among ties follows the physical order of first occurrence.
    ranked.sort(key=lambda v: v[0], reverse=True)
    return [name for _, name in ranked[:k]]


def select_reply(scores, groups, candidates):
    flat = scores.flatten().astype(np.float64)

    def logmass(name):
        return np.logaddexp.reduce(flat[groups[name]])

    return max(candidates, key=logmass)

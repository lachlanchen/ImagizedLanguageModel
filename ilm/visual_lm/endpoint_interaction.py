"""Question-conditioned endpoint corrections on the existing physical interval writer."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.paired_evidence import EvidenceWriter, LENGTH


class EndpointWriter(EvidenceWriter):
    def __init__(self, mode, width=80, rank=8):
        super().__init__(width)
        if mode not in ("additive", "product"):
            raise ValueError("Expected additive or product endpoint scores")
        self.mode = mode
        self.left = nn.Linear(width, rank)
        self.right = nn.Linear(width, rank)
        self.gain = nn.Parameter(torch.zeros(()))

    def correction(self, key):
        left = self.left(key).tanh()
        right = self.right(key).tanh()
        # [batch, start, length - 1, coordinate]
        ends = (
            F.pad(right, (0, 0, 0, LENGTH - 1)).unfold(1, LENGTH, 1).transpose(-1, -2)
        )
        if self.mode == "product":
            pairs = left.unsqueeze(2) * ends
        else:
            pairs = left.unsqueeze(2) + ends
        return pairs.sum(-1).float() * left.size(-1) ** -0.5

    def interval_scores(self, key, ev, summary):
        return super().interval_scores(key, ev, summary) + self.gain * self.correction(
            key
        )

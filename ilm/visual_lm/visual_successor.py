"""Causal compatibility of a visible history and a proposed successor bitmap."""
import torch
from torch import nn
from torch.nn import functional as F


class VisualSuccessor(nn.Module):
    history_length = 32

    def __init__(self):
        super().__init__()
        self.shape = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 96))
        self.context = nn.Sequential(nn.Linear(5 * 96, 128), nn.GELU(), nn.Linear(128, 64))
        self.candidate = nn.Linear(96, 64)

    @staticmethod
    def pool(ink):
        """Pool binary foreground without a font, character map or vocabulary."""
        if ink.shape[-2:] != (32, 32):
            raise ValueError('Expected 32 by 32 glyph images')
        pooled = F.avg_pool2d(ink.reshape(-1, 1, 32, 32).float(), 2)
        return pooled.reshape(*ink.shape[:-2], 256)

    def queries(self, glyph_features):
        """One query after each causal 32-cell window; zero features pad short histories."""
        if glyph_features.ndim != 3 or glyph_features.shape[1] < self.history_length:
            raise ValueError('Supply at least one 32-cell window, left-padded if needed')
        prefix = F.pad(glyph_features.cumsum(1), (0, 0, 1, 0))
        end = self.history_length
        ranges = [(prefix[:, end:] - prefix[:, end-width:-width]) / width for width in (4, 16, 32)]
        features = torch.cat([glyph_features[:, end-1:], glyph_features[:, end-2:-1], *ranges], dim=-1)
        return F.normalize(self.context(features), dim=-1)

    def keys(self, glyph_features):
        return F.normalize(self.candidate(glyph_features), dim=-1)

    def forward(self, histories, candidates):
        """Score every candidate for each independent, complete history in the batch."""
        history = self.shape(self.pool(histories))
        key = self.keys(self.shape(self.pool(candidates)))
        return 10 * (self.queries(history)[:, -1:] @ key.transpose(-1, -2)).squeeze(1)

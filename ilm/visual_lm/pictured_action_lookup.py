"""Compare a Chinese predicate in context with the ink in an English request."""

import torch
from torch import nn
from torch.nn import functional as F


class PicturedActionLookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.letters = nn.Linear(256, 96)
        self.short = nn.Conv1d(96, 96, 5, padding=2)
        self.long = nn.Conv1d(96, 96, 5, padding=2)
        self.source = nn.Sequential(nn.Linear(160, 128), nn.GELU(), nn.Linear(128, 96))

    def questions(self, features, lengths):
        mask = (
            torch.arange(features.shape[1], device=features.device) < lengths[:, None]
        )
        x = F.gelu(self.letters(features)) * mask[..., None]
        for layer in (self.short, self.long):
            x = (x + F.gelu(layer(x.transpose(1, 2)).transpose(1, 2))) * mask[..., None]
        return F.normalize(x, dim=-1), mask

    def predicates(self, states):
        return F.normalize(self.source(states), dim=-1)

    def scores(self, states, features, lengths):
        letters, mask = self.questions(features, lengths)
        sim = torch.einsum("id,jkd->ijk", self.predicates(states), letters)
        return sim.masked_fill(~mask, -torch.inf).amax(-1)

"""Distil word meaning into optical features and use it to select a Chinese line."""

import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.pictured_action_lookup import PicturedActionLookup


class OpticalWordMeaning(nn.Module):
    width = 24

    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(256, 32, bias=False)
        self.readout = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.width * 32, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 50),
        )

    def forward(self, projected):
        return F.normalize(self.readout(projected), dim=-1)

    def sentence(self, features, lengths):
        # Blank cells separate English words. Punctuation remains visible ink.
        offsets = torch.arange(features.size(1), device=features.device)[None]
        ink = features.abs().any(-1) & (offsets < lengths[:, None])
        start = ink & ~F.pad(ink[:, :-1], (1, 0))
        word = start.cumsum(1) - 1
        position = offsets - (offsets * start).cummax(1).values
        counts = start.sum(1)
        width = max(1, int(counts.max()))
        packed = features.new_zeros(
            len(features) * width, self.width, self.cell.out_features
        )
        b, t = torch.where(ink & (position < self.width))
        packed[b * width + word[b, t], position[b, t]] = self.cell(features[b, t])
        return self(packed).reshape(len(features), width, 50), counts


class VisualActionMeaning(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = PicturedActionLookup()
        self.words = OpticalWordMeaning()
        self.source = nn.Sequential(
            nn.Linear(160, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 50)
        )
        self.context = nn.Conv1d(50, 96, 3, padding=1)
        self.salience = nn.Linear(96, 1)
        self.residual = nn.Linear(96, 50)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def meaning(self, words, counts):
        mask = torch.arange(words.size(1), device=words.device)[None] < counts[:, None]
        hidden = F.gelu(
            self.context((words * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        )
        att = (
            self.salience(hidden).squeeze(-1).masked_fill(~mask, -1e4).softmax(1) * mask
        )
        return F.normalize(
            ((words + self.residual(hidden)) * att[..., None]).sum(1), dim=-1
        )

    def questions(self, features, lengths):
        old, mask = self.parent.questions(features, lengths)
        shared = self.meaning(*self.words.sentence(features, lengths))
        return torch.cat(
            (old * 0.35, shared[:, None, :].expand(-1, old.size(1), -1) * 0.65), -1
        ), mask

    def predicates(self, states):
        return torch.cat(
            (self.parent.predicates(states), F.normalize(self.source(states), dim=-1)),
            -1,
        )

    def scores(self, states, features, lengths):
        q, mask = self.questions(features, lengths)
        return (
            torch.einsum("id,jkd->ijk", self.predicates(states), q)
            .masked_fill(~mask, -torch.inf)
            .amax(-1)
        )

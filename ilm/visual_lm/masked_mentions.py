"""Read context around anonymous name occurrences, then copy the chosen source strip."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedMentions(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.ink = nn.Linear(256, width)
        self.marker = nn.Parameter(torch.zeros(width))
        self.conv = nn.ModuleList(
            nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in (1, 3, 7)
        )
        self.norm = nn.ModuleList(nn.LayerNorm(width) for _ in self.conv)
        self.topic = nn.Linear(width, width)
        self.refer = nn.Linear(width, width)
        self.local = nn.Linear(width, 1)
        self.adjust = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        self.reject = nn.Sequential(
            nn.Linear(width + 1, 24), nn.GELU(), nn.Linear(24, 1)
        )
        self.dropout = nn.Dropout(0.22)

    def forward(self, features, slot, query, lengths):
        pos = torch.arange(features.size(1), device=features.device)[None]
        visible = pos < lengths[:, None]
        named = slot >= 0
        h = torch.where(named[..., None], self.marker, self.ink(features))
        for conv, norm in zip(self.conv, self.norm):
            delta = conv((h * visible[..., None]).transpose(1, 2)).transpose(1, 2)
            h = norm(h + self.dropout(F.gelu(delta)))
        marked = (pos >= query[:, 0, None]) & (pos < query[:, 1, None])
        q = (h * marked[..., None]).sum(1) / marked.sum(1, keepdim=True).clamp_min(1)
        eligible = named & (pos < query[:, 0, None])
        distance = (query[:, 0, None] - pos).clamp_min(0).float()
        distance_features = torch.stack(
            (
                distance / 288,
                torch.log1p(distance) / 5,
                torch.exp(-distance / 8),
                torch.exp(-distance / 50),
            ),
            dim=-1,
        )
        score = (self.topic(h) * self.refer(q)[:, None]).sum(-1) / h.size(-1) ** 0.5
        score = (
            score
            + self.local(h).squeeze(-1)
            + self.adjust(distance_features).squeeze(-1)
            - 0.45 * torch.log1p(distance)
        )
        null = self.reject(
            torch.cat((q, torch.log1p(eligible.sum(1, keepdim=True).float())), 1)
        )
        return torch.cat((score.masked_fill(~eligible, -1e4), null), dim=1)

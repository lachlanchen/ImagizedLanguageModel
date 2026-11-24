"""A visual adjustment to the odds of the two previously introduced candidates."""

import torch
from torch import nn


class ReferenceOrder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = nn.Linear(256, 24)
        self.compare = nn.Sequential(
            nn.Flatten(), nn.Linear(14 * 24, 64), nn.GELU(), nn.Linear(64, 2)
        )
        nn.init.zeros_(self.compare[-1].weight)
        nn.init.zeros_(self.compare[-1].bias)

    def forward(self, patch):
        delta, present = self.compare(torch.tanh(self.cells(patch))).unbind(1)
        return torch.stack(
            (present - delta, present + delta, torch.zeros_like(delta)), 1
        )


def cue(features, query, slots, lengths, erase=False):
    visible = (
        torch.arange(features.size(1), device=query.device)[None] < lengths[:, None]
    )
    features = features.masked_fill((~visible | (slots >= 0))[..., None], 0.0)
    pos = torch.arange(8, device=query.device)[None] + query[:, 0, None]
    right = torch.arange(6, device=query.device)[None] + query[:, 1, None]
    index = torch.cat((pos, right), 1)
    keep = torch.cat(
        ((pos < query[:, 1, None]) & (not erase), right < features.shape[1]), 1
    )
    return (
        features.gather(
            1,
            index.clamp_max(features.shape[1] - 1)[..., None].expand(
                -1, -1, features.shape[-1]
            ),
        )
        * keep[:, :, None]
    )

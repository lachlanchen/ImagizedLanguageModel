"""Condition the physical writer on an ordered visual phrase around a gap."""
import torch
from torch import nn

from ilm.visual_lm.idiom_ink import PicturedIdiom


class NeighborhoodWriter(PicturedIdiom):
    def __init__(self, landmarks):
        super().__init__(landmarks)
        self.neighborhood = nn.Sequential(
            nn.LayerNorm(8 * 112), nn.Linear(8 * 112, 512), nn.GELU(), nn.Linear(512, 192),
        )
        nn.init.zeros_(self.neighborhood[-1].weight)
        nn.init.zeros_(self.neighborhood[-1].bias)

    def read(self, features, neighborhood=None):
        cells = features[:, 20:28] if neighborhood is None else neighborhood
        return super().read(features) + self.neighborhood(cells.flatten(1))

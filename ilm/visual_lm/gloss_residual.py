"""A small residual on the continuous visual state presented to the speller."""

from torch import nn


class GlossResidual(nn.Module):
    def __init__(self, width=160):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, context):
        return context + self.delta(context)

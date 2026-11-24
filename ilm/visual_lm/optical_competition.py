"""Compare question-conditioned pictured span candidates in a permutation-equivariant set."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class CandidateCompetition(nn.Module):
    modes = ("independent", "competing")

    def __init__(self, mode, width=64):
        super().__init__()
        if mode not in self.modes:
            raise ValueError(mode)
        self.mode = mode
        self.project = nn.Linear(128, width)
        self.role = nn.Parameter(torch.zeros(3, width))
        self.length = nn.Embedding(257, width)
        self.query = nn.Linear(3 * width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, 1, bias=False),
        )
        nn.init.zeros_(self.out[-1].weight)

    def forward(self, images, base, lengths, within=False):
        states = F.gelu(self.project(images.float()) + self.role)
        query = self.query(states.flatten(2)) + self.length(lengths)
        # All candidate states encode the pictured question as well as their source.
        keys = self.key(states).flatten(1, 2)
        values = self.value(states).flatten(1, 2)
        affinity = query @ keys.transpose(1, 2) / math.sqrt(query.size(-1))
        valid = base.isfinite()
        mask = valid.repeat_interleave(3, dim=1)[:, None, :]
        if self.mode == "independent" or within:
            positions = torch.arange(base.size(1), device=base.device)
            mask = mask & (
                positions[:, None] == positions.repeat_interleave(3)[None, :]
            )
        logits = affinity.masked_fill(~mask, -torch.inf)
        # Padded candidates have no own valid evidence, including on very short pages.
        attn = logits.masked_fill(~valid[:, :, None], 0).softmax(-1)
        context = attn @ values
        return base + self.out(
            torch.cat((query, context, query * context), -1)
        ).squeeze(-1)


def shortlist(reader, features, query, page):
    """The fixed directional reader's exact global list and float16 states."""
    scores, state = reader.lattice(features, query, page)
    flat = scores.flatten(1)
    ix = flat.argsort(descending=True, stable=True)[:, :32]
    base = flat.gather(1, ix)
    start = (ix // scores.size(-1)).masked_fill(~base.isfinite(), 0)
    size = (ix % scores.size(-1) + 1).masked_fill(~base.isfinite(), 1)
    end = start + size - 1
    batch = torch.arange(len(page), device=page.device)[:, None]
    total = F.pad(state.float().cumsum(1), (0, 0, 1, 0))
    inside = (total[batch, end + 1] - total[batch, start]) / size.unsqueeze(-1)
    return (
        torch.stack((state[batch, start], state[batch, end], inside), 2).half(),
        base,
        start,
        size,
    )

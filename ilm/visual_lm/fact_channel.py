"""Read a contiguous fact directly from a page of glyph bitmaps."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.region_writer import StripEncoder


def span_scores(start, end, cells, length_bias):
    """Score every legal span. Zero cells delimit regions and batch padding."""
    length = min(len(length_bias), cells.shape[1])
    tails = F.pad(end, (0, length - 1)).unfold(1, length, 1)
    boundary = F.pad(cells.eq(0).long(), (0, length - 1), value=1).cumsum(1)
    previous = F.pad(boundary, (1, 0))[:, : cells.shape[1]]
    valid = boundary.unfold(1, length, 1).eq(previous[:, :, None])
    return (start[:, :, None] + tails + length_bias[:length]).masked_fill(
        ~valid, -torch.inf
    )


class OpticalFactReader(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.ink = StripEncoder(width)
        self.qkey = nn.Linear(width, width)
        self.pkey = nn.Linear(width, width)
        self.pool = nn.Linear(width, 1)
        self.fuse = nn.Sequential(
            nn.Linear(4 * width, width), nn.LayerNorm(width), nn.GELU()
        )
        self.local = nn.Conv1d(width, width, 5, padding=6, dilation=3)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Linear(width, 3 * width)
        self.boundaries = nn.Linear(2 * width, 2)
        self.length_bias = nn.Parameter(
            -torch.arange(1, 257, dtype=torch.float32).log()
        )

    def forward(self, features, query, page):
        bank = self.ink.input(features)
        q = self.ink(bank, query)
        p = self.ink(bank, page)
        align = self.pkey(p) @ self.qkey(q).transpose(1, 2) / math.sqrt(q.shape[-1])
        attn = (
            align.float()
            .masked_fill(query[:, None, :].eq(0), -1e4)
            .softmax(-1)
            .to(q.dtype)
        )
        mean = (
            self.pool(q)
            .squeeze(-1)
            .float()
            .masked_fill(query.eq(0), -1e4)
            .softmax(-1)
            .to(q.dtype)[:, None, :]
            @ q
        )
        local = attn @ q
        h = (
            self.fuse(torch.cat([p, local, p * local, mean.expand_as(p)], -1))
            * page.ne(0)[:, :, None]
        )
        local = self.norm(h + F.gelu(self.local(h.transpose(1, 2)).transpose(1, 2)))
        heads = (
            self.attention(local)
            .reshape(len(page), page.shape[1], 3, 4, -1)
            .permute(2, 0, 3, 1, 4)
        )
        ctx = F.scaled_dot_product_attention(
            *heads.unbind(0), attn_mask=page[:, None, None, :].ne(0)
        )
        ctx = ctx.transpose(1, 2).reshape_as(h)
        start, end = self.boundaries(torch.cat([h, ctx], -1)).float().unbind(-1)
        return span_scores(start, end, page, self.length_bias)

    def loss(self, scores, aliases):
        flat = scores.flatten(1)
        correct = flat.gather(1, aliases.clamp_min(0)).masked_fill(
            aliases.lt(0), -torch.inf
        )
        return flat.logsumexp(1) - correct.logsumexp(1)

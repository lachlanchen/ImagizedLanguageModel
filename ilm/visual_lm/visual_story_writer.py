"""Causal attention over earlier glyph states with an open copy channel from page pixels."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.region_writer import RegionWriter


class AttentiveHistory(nn.Module):
    def __init__(self, recurrent, heads=4):
        super().__init__()
        self.recurrent = recurrent
        width = recurrent.hidden_size
        self.heads = heads
        self.norm = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, x, previous):
        cached = isinstance(previous, tuple)
        recurrent = previous[0] if cached else previous
        states, hidden = self.recurrent(x, recurrent)
        batch, length, width = states.shape
        q, k, v = (
            self.qkv(self.norm(states))
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        mask = None
        if cached:
            offset = previous[1].shape[2]
            k = torch.cat((previous[1], k), 2)
            v = torch.cat((previous[2], v), 2)
            if length > 1:
                mask = (
                    torch.arange(k.size(2), device=x.device)[None, :]
                    <= offset + torch.arange(length, device=x.device)[:, None]
                )
        ctx = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=not cached and length > 1
        )
        ctx = ctx.transpose(1, 2).reshape(batch, length, width)
        return states + self.project(ctx), (hidden, k, v)


class VisualStoryWriter(RegionWriter):
    def __init__(self):
        super().__init__()
        self.gru = AttentiveHistory(self.gru)

    def objective(self, ctx, hidden, y):
        bank, concepts, src, _, _ = ctx
        prev = torch.cat((self.bos.expand(len(y), 1, -1), bank[y[:, :-1]]), 1)
        states, _ = self.gru(prev, hidden)
        history = torch.stack([F.pad(y, (i, 0))[:, : y.size(1)] for i in (1, 2, 3)], -1)
        lp, copy, gate = self.distribution(ctx, states, history)
        known = (y > 0) & (y <= len(concepts) + 1)
        positions = (src[:, None, :] == y[:, :, None]) & (y[:, :, None] != 0)
        copied = (copy * positions).sum(-1)
        reachable = known | positions.any(-1)
        spelling = lp.gather(
            -1, (y - 1).clamp(0, lp.size(-1) - 1).unsqueeze(-1)
        ).squeeze(-1)
        # Give unscored padding/unsupported cells a finite dummy branch before logaddexp.
        a = (F.logsigmoid(gate) + spelling).masked_fill(~known & reachable, -torch.inf)
        b = (F.logsigmoid(-gate) + copied.clamp_min(1e-38).log()).masked_fill(
            ~positions.any(-1), -torch.inf
        )
        mass = torch.logaddexp(a, b).masked_fill(~reachable, 0)
        return -(mass.sum(1) / reachable.sum(1)), ((y != 0) & ~reachable).sum(1)

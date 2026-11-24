"""Independent writer with optical glyph keys, a recurrent prefix and passage attention."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class StripEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(256, width), nn.LayerNorm(width), nn.GELU()
        )
        self.conv = nn.ModuleList(
            nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in (1, 3)
        )
        self.norm = nn.ModuleList(nn.LayerNorm(width) for _ in self.conv)

    def forward(self, bank, ids):
        x = bank[ids]
        mask = ids.ne(0).unsqueeze(-1)
        for layer, norm in zip(self.conv, self.norm):
            v = layer((x * mask).transpose(1, 2)).transpose(1, 2)
            x = norm(x + F.gelu(v))
        return x * mask


class RegionSelector(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.ink = StripEncoder(width)
        self.query = nn.Linear(width, width)
        self.source = nn.Linear(width, width)
        self.qweight = nn.Linear(width, 1)
        self.bias = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, features, q, blocks, owners, offsets):
        bank = self.ink.input(features)
        h = self.ink(bank, q)
        s = self.ink(bank, blocks)
        qm = q.ne(0)
        sm = blocks.ne(0)
        alignment = F.normalize(self.query(h)[owners], dim=-1) @ F.normalize(
            self.source(s), dim=-1
        ).transpose(1, 2)
        local = alignment.masked_fill(~sm[:, None, :], -2).amax(-1)
        weights = self.qweight(h).squeeze(-1).float().masked_fill(~qm, -1e4).softmax(-1)
        meanq = (h * weights[:, :, None]).sum(1)[owners]
        means = s.sum(1) / sm.sum(1, keepdim=True)
        score = (local * weights[owners]).sum(-1) * 12 + self.bias(
            torch.cat([meanq, means, meanq * means], -1)
        ).flatten()
        out = score.new_full((len(q), offsets.shape[1]), -1e4)
        valid = offsets >= 0
        return torch.where(valid, score[offsets.clamp_min(0)], out)


class RegionWriter(nn.Module):
    def __init__(self, width=160):
        super().__init__()
        self.ink = StripEncoder(width)
        self.glyphs = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width))
        self.role = nn.Parameter(torch.randn(2, width) * 0.02)
        self.bos = nn.Parameter(torch.randn(width) * 0.02)
        self.stop = nn.Linear(width, 1)
        self.start = nn.Linear(width * 2, width * 2)
        self.gru = nn.GRU(width, width, num_layers=2, batch_first=True)
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.join = nn.Sequential(
            nn.Linear(2 * width, width), nn.LayerNorm(width), nn.GELU()
        )
        self.copy = nn.Linear(width, width, bias=False)
        self.route = nn.Linear(width, 1)
        self.suffix_bias = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def encode(self, features, n, q, p):
        b = self.ink.input(features)
        hh, ss = (self.ink(b, ids) + role for ids, role in zip((q, p), self.role))
        means = [
            (v * (i != 0)[:, :, None]).sum(1) / (i != 0).sum(1, keepdim=True)
            for v, i in ((hh, q), (ss, p))
        ]
        hidden = (
            self.start(torch.cat(means, -1))
            .tanh()
            .reshape(len(q), 2, -1)
            .transpose(0, 1)
            .contiguous()
        )
        # A masked boundary prevents accidental matching from the query into the passage.
        ids = F.pad(q, (0, 1), value=0)
        src = torch.cat([ids, p], 1)
        values = torch.cat([F.pad(hh, (0, 0, 0, 1)), ss], 1)
        return (b, self.glyphs(b[2:n]), src, values, self.key(values)), hidden

    def distribution(self, ctx, state, history):
        bank, concepts, ids, values, keys = ctx
        scale = math.sqrt(state.shape[-1])
        scores = (self.query(state) @ keys.transpose(1, 2)) / scale
        mask = ids[:, None, :] == 0
        attended = (
            scores.float().masked_fill(mask, -1e4).softmax(-1).to(values.dtype) @ values
        )
        h = self.join(torch.cat([attended, state], -1))
        copy = (self.copy(h) @ keys.transpose(1, 2)).float() / scale
        # history contains the previous three emitted cell identities at each step.
        suffix = torch.ones_like(copy, dtype=torch.bool)
        for i, w in enumerate(self.suffix_bias):
            before = F.pad(ids, (i + 1, 0))[:, : ids.size(1)]
            past = history[:, :, i, None]
            suffix = suffix & (past >= 2) & (past == before[:, None, :])
            copy = copy + w * suffix
        copy = copy.masked_fill(mask, -1e4).softmax(-1)
        logits = torch.cat([self.stop(h), (h @ concepts.T) / scale], -1).float()
        return logits.log_softmax(-1), copy, self.route(h).float().squeeze(-1)

    def loss(self, ctx, hidden, y):
        b, _, src, _, _ = ctx
        inputs = torch.cat([self.bos.expand(len(y), 1, -1), b[y[:, :-1]]], 1)
        h, _ = self.gru(inputs, hidden)
        history = torch.stack([F.pad(y, (i, 0))[:, : y.size(1)] for i in (1, 2, 3)], -1)
        lp, cp, gate = self.distribution(ctx, h, history)
        spell = lp.gather(-1, (y - 1).clamp(0, lp.size(-1) - 1)[:, :, None]).squeeze(-1)
        spell = spell.masked_fill(y > lp.size(-1), -1e4)
        copy = (cp * (src[:, None, :] == y[:, :, None])).sum(-1)
        joint = torch.logaddexp(
            F.logsigmoid(gate) + spell,
            F.logsigmoid(-gate) + copy.clamp_min(1e-30).log(),
        )
        return -(joint * (y != 0)).sum(1) / (y != 0).sum(1)

    @torch.inference_mode()
    def write(self, ctx, hidden, limit=256):
        b, concepts, src, _, _ = ctx
        bs = len(src)
        past = src.new_zeros((bs, 1, 3))
        x = self.bos.expand(bs, 1, -1)
        ended = torch.zeros(bs, dtype=torch.bool, device=x.device)
        out = []
        for _ in range(limit):
            state, hidden = self.gru(x, hidden)
            lp, cp, g = self.distribution(ctx, state, past)
            mass = cp.new_zeros((bs, len(b)))
            mass[:, 1 : len(concepts) + 2] = lp[:, 0].exp() * g.sigmoid()
            mass.scatter_add_(1, src, cp[:, 0] * (-g).sigmoid())
            v = mass.argmax(-1)
            v = torch.where(ended, torch.ones_like(v), v)
            out.append(v)
            ended |= v == 1
            if ended.all():
                break
            past = torch.cat([v[:, None, None], past[:, :, :2]], -1)
            x = b[v][:, None, :]
        result = []
        for row in torch.stack(out, 1).tolist():
            result.append((row[: row.index(1)], True) if 1 in row else (row, False))
        return result

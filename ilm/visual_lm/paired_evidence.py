"""Question-conditioned complete source intervals and open glyph generation."""

import torch
from torch import nn
from torch.nn import functional as F

LENGTH = 48


class LocalContext(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2 * d, dilation=d), nn.GELU()
            )
            for d in (1, 2, 4)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x, mask):
        h = x.transpose(1, 2)
        m = mask[:, None]
        for block in self.blocks:
            h = (h + block(h * m)) * m
        return self.norm(h.transpose(1, 2)) * mask[:, :, None]


class EvidenceWriter(nn.Module):
    def __init__(self, width=80):
        super().__init__()
        self.ink = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, width), nn.LayerNorm(width)
        )
        self.context = LocalContext(width)
        self.ask = nn.Linear(width, 1)
        self.fuse = nn.Linear(4 * width + 6, width)
        self.reason = LocalContext(width)
        self.evidence = nn.Linear(width, 1)
        self.where = nn.Linear(width, 2)
        self.summary = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh())
        self.length = nn.Linear(width, LENGTH)
        self.span = nn.Linear(width, 1)
        self.begin = nn.Parameter(torch.zeros(width))
        self.gru = nn.GRU(width, width, batch_first=True)
        self.readout = nn.Linear(width, width)
        self.free_mix = nn.Linear(width, 1)
        self.stop = nn.Linear(width, 1)

    def read(self, bank, q, s):
        qm = q >= 2
        sm = s >= 2
        u = self.context(bank[q.long()], qm)
        v = self.context(bank[s.long()], sm)
        qpool = (
            self.ask(u)
            .squeeze(-1)
            .float()
            .masked_fill(~qm, -1e9)
            .softmax(-1)[:, None]
            .to(u.dtype)
            @ u
        )
        a = (
            (v @ u.transpose(1, 2) * u.size(-1) ** -0.5)
            .float()
            .masked_fill(~qm[:, None, :], -1e9)
        )
        other = a.softmax(-1).to(u.dtype) @ u
        match = (s[:, :, None] == q[:, None, :]) & qm[:, None, :] & sm[:, :, None]
        two = match & F.pad(match[:, :-1, :-1], (1, 0, 1, 0))
        three = (
            two & F.pad(match[:, :-2, :-2], (2, 0, 2, 0))[:, : s.size(1), : q.size(1)]
        )
        denom = sm.sum(-1, keepdim=True).clamp_min(1)
        pos = torch.arange(s.size(1), device=s.device)[None, :] / denom
        cue = torch.stack(
            (
                match.any(-1),
                two.any(-1),
                three.any(-1),
                match.sum(-1).float().log1p(),
                a.amax(-1),
                pos,
            ),
            -1,
        )
        fused = self.fuse(
            torch.cat((v, other, v * other, qpool.expand_as(v), cue.to(v.dtype)), -1)
        )
        key = self.reason(fused, sm)
        ev = self.evidence(key).squeeze(-1).float().masked_fill(~sm, -1e9)
        summary = self.summary(
            torch.cat(
                (qpool[:, 0], (ev.softmax(-1)[:, None].to(key.dtype) @ key)[:, 0]), -1
            )
        )
        valid = F.pad(sm, (0, LENGTH - 1)).unfold(1, LENGTH, 1)
        scores = self.interval_scores(key, ev, summary)
        score = scores.masked_fill(~valid, -1e9)
        return (
            score.flatten(1).log_softmax(-1).view_as(score),
            ev,
            summary,
            self.span(summary).float().squeeze(-1),
        )

    def interval_scores(self, key, ev, summary):
        start, end = self.where(key).float().unbind(-1)
        stops = F.pad(end, (0, LENGTH - 1), value=-1e9).unfold(1, LENGTH, 1)
        return (
            start[:, :, None]
            + ev[:, :, None]
            + stops
            + self.length(summary).float()[:, None, :]
        )

    def glyph_scores(self, state, bank, n):
        logits = (self.readout(state) @ bank[2:n].T * bank.size(-1) ** -0.5).float()
        stop = self.stop(state).float()
        return torch.cat(
            (F.logsigmoid(stop), F.logsigmoid(-stop) + logits.log_softmax(-1)), -1
        )

    def loss(self, features, q, s, y, base_lp, n, left, right, weight):
        bank = self.ink(features)
        spans, ev, summary, gate = self.read(bank, q, s)
        hist = torch.cat((self.begin.expand(len(q), 1, -1), bank[y[:, :-1].long()]), 1)
        states, _ = self.gru(hist, summary[None].contiguous())
        pred = (
            self.glyph_scores(states, bank, n)
            .gather(-1, (y.long() - 1).clamp(0, n - 2)[:, :, None])
            .squeeze(-1)
        )
        pred = pred.masked_fill(y >= n, -1e9)
        mix = self.free_mix(states).float().squeeze(-1)
        free = torch.logaddexp(F.logsigmoid(mix) + pred, F.logsigmoid(-mix) + base_lp)
        free = (free * (y > 0)).sum(1)
        mask = compatible(s, y)
        copied = spans.masked_fill(~mask, -1e9).flatten(1).logsumexp(-1)
        total = torch.logaddexp(F.logsigmoid(gate) + copied, F.logsigmoid(-gate) + free)
        loss = -(total / (y > 0).sum(-1).clamp_min(1)).mean()
        pos = torch.arange(s.size(1), device=s.device)
        marked = (
            (pos[None] >= left[:, None])
            & (pos[None] < right[:, None])
            & (left[:, None] >= 0)
        )
        aux = -(ev.log_softmax(-1) * marked).sum(-1) / marked.sum(-1).clamp_min(1)
        return loss + weight * aux.sum() / marked.any(-1).sum().clamp_min(1)


def compatible(s, y):
    """Source intervals emitting this stopped string or this censored prefix."""
    cells = (y >= 2).sum(1)
    stopped = (y == 1).any(1)
    length = torch.arange(1, LENGTH + 1, device=s.device)
    windows = F.pad(s, (0, LENGTH - 1)).unfold(1, LENGTH, 1)
    target = F.pad(y, (0, max(0, LENGTH - y.size(1))))[:, :LENGTH]
    relevant = length[None, :] <= cells[:, None]
    equal = ((windows == target[:, None, :]) | ~relevant[:, None, :]).all(-1)
    size = torch.where(
        stopped[:, None],
        length[None, :] == cells[:, None],
        length[None, :] >= cells[:, None],
    )
    return (
        equal[:, :, None]
        & size[:, None, :]
        & (cells <= LENGTH)[:, None, None]
        & (windows >= 2)
    )

"""Globally normalized visual field plans with a latent position for each requested fact."""

import torch
from torch import nn
from torch.nn import functional as F


def plan_scores(z, ink, available, pair_bias):
    """Log-mean evidence over one anchor or over all strictly increasing anchor pairs."""
    z = z.float().masked_fill(~ink[:, :, None], -1e4)
    count = ink.sum(1).clamp_min(1).float()
    one = torch.logsumexp(z, 1) - count.log()[:, None]
    left = torch.logcumsumexp(z[:, :-1], 1)
    pairs = torch.logsumexp(left[:, :, :, None] + z[:, 1:, None, :], 1)
    pairs = (
        pairs
        - ((count * (count - 1) / 2).clamp_min(1)).log()[:, None, None]
        + pair_bias
    )
    ok = available[:, :, None] & available[:, None, :]
    ok = ok & ~torch.eye(z.size(2), device=z.device, dtype=torch.bool)[None]
    ok = ok & (count >= 2)[:, None, None]
    return torch.cat(
        (
            one.masked_fill(~available, -torch.inf),
            pairs.masked_fill(~ok, -torch.inf).flatten(1),
        ),
        1,
    )


class FieldAlignment(nn.Module):
    def __init__(self, fields, domains):
        super().__init__()
        self.fields = fields
        self.vision = nn.Linear(256, 64)
        self.context = nn.Sequential(
            nn.Conv1d(64, 80, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(80, 80, 5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(80, 64, 3, padding=1),
            nn.GELU(),
        )
        self.domain = nn.Embedding(domains, 64)
        self.detect = nn.Linear(64, fields)
        self.pair_bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, query, domains, available):
        features = F.gelu(self.vision(query))
        local = self.context(features.transpose(1, 2)).transpose(1, 2)
        z = self.detect(torch.tanh(local + self.domain(domains)[:, None, :]))
        return plan_scores(z, query.abs().sum(2) > 0, available, self.pair_bias)

    def choose(self, query, domains, available):
        best = self(query, domains, available).argmax(1)
        pair = (best - self.fields).clamp_min(0)
        a = torch.where(best < self.fields, best, pair // self.fields)
        b = torch.where(best < self.fields, self.fields, pair % self.fields)
        return torch.stack((a, b), 1)


def target_loss(scores, fields, ordered, nfields):
    a, b = fields.unbind(1)
    pair = b < nfields
    ids = torch.where(pair, nfields + a * nfields + b, a)
    opposite = nfields + b.clamp_max(nfields - 1) * nfields + a
    p = scores.log_softmax(1)
    idx = torch.arange(len(p), device=p.device)
    first = p[idx, ids]
    # Replace invalid entries before combining so a single-label sample never
    # differentiates through logaddexp(-inf,-inf).
    second = p[idx, opposite].masked_fill(~pair, -1e4)
    return -torch.where(pair & ~ordered, torch.logaddexp(first, second), first).mean()

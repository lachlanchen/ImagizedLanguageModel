"""A glyph transducer with latent copy runs and an autoregressive spelling path."""

import numpy as np
import torch
from numba import njit
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.paired_evidence import EvidenceWriter, compatible

RUNS = (1, 2, 4, 8)


@njit
def inside_outside(arcs, counts):
    """Sum segmentation paths and return edge posterior visits on a finite DAG."""
    b, t, k = arcs.shape
    costs = np.empty(b, np.float32)
    visits = np.zeros_like(arcs)
    for i in range(b):
        n = counts[i]
        alpha = np.full(n + 1, -np.inf)
        alpha[0] = 0.0
        beta = np.full(n + 1, -np.inf)
        beta[n] = 0.0
        for j in range(n):
            for a in range(k):
                end = min(n, j + (1 if a == 0 else 2 ** (a - 1)))
                alpha[end] = np.logaddexp(alpha[end], alpha[j] + arcs[i, j, a])
        for j in range(n - 1, -1, -1):
            for a in range(k):
                end = min(n, j + (1 if a == 0 else 2 ** (a - 1)))
                v = arcs[i, j, a] + beta[end]
                beta[j] = np.logaddexp(beta[j], v)
                visits[i, j, a] = np.exp(alpha[j] + v - alpha[n])
        costs[i] = alpha[n]
    return costs, visits


class PathMass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arcs, counts):
        mass, visits = inside_outside(
            arcs.detach().float().cpu().numpy(), counts.cpu().numpy()
        )
        ctx.save_for_backward(torch.from_numpy(visits).to(arcs.device))
        return torch.from_numpy(mass).to(arcs.device)

    @staticmethod
    def backward(ctx, grad):
        (visits,) = ctx.saved_tensors
        return visits * grad[:, None, None], None


def previous_marks(ids):
    return torch.stack(
        [F.pad(ids, (lag, 0))[:, : ids.size(1)] for lag in (1, 2, 3)], -1
    )


def run_matches(s, y):
    """Match observed prefixes at cell boundaries; pad after truncation is unobserved."""
    n, t = y.shape
    v = s.size(1)
    sy = F.pad(y, (0, 7))
    ss = F.pad(s, (0, 7))
    good = torch.ones((n, t, v), device=s.device, dtype=torch.bool)
    for k in range(8):
        target = sy[:, k : k + t, None]
        good = good & ((ss[:, None, k : k + v] == target) | (target == 0))
        if k + 1 in RUNS:
            yield good


class ComposingWriter(EvidenceWriter):
    def __init__(self, dynamic=True, width=80):
        super().__init__(width)
        self.dynamic = dynamic
        self.keys = nn.Linear(width, width)
        self.address = nn.Linear(width, width)
        self.cues = nn.Parameter(torch.ones(3))
        self.bridge = nn.Linear(width, width, bias=False)
        self.run_type = nn.Linear(width, 5)
        nn.init.zeros_(self.bridge.weight)

    def page(self, bank, q, s):
        spans, ev, summary, gate = self.read(bank, q, s)
        values = self.context(bank[s.long()], s >= 2)
        return spans, ev, summary, gate, values, self.keys(values), previous_marks(s)

    def actions(self, state, history, s, page):
        _, ev, summary, _, values, keys, prev = page
        query = state if self.dynamic else summary[:, None].expand_as(state)
        raw = (
            self.address(query) @ keys.transpose(1, 2) * keys.shape[-1] ** -0.5
        ).float() + ev[:, None]
        if self.dynamic:
            for j in range(3):
                seen = history[:, :, j, None]
                raw = raw + self.cues[j] * ((seen == prev[:, None, :, j]) & (seen >= 2))
        focus = raw.softmax(-1).to(values.dtype) @ values
        mixed = state + self.bridge(focus)
        policies = self.run_type(mixed).float()
        enough = (s >= 2).sum(-1)[:, None, None] >= torch.tensor(RUNS, device=s.device)
        policies = torch.cat(
            (policies[:, :, :1], policies[:, :, 1:].masked_fill(~enough, -1e9)), -1
        ).log_softmax(-1)
        copy = []
        for i, length in enumerate(RUNS):
            end = F.pad(s, (0, length - 1))[:, length - 1 :]
            weight = raw.masked_fill(end[:, None, :] < 2, -1e9).log_softmax(-1)
            copy.append(weight + policies[:, :, i + 1, None])
        return mixed, policies[:, :, 0], copy

    def observed(self, features, q, s, y, base_lp, n):
        bank = self.ink(features)
        page = self.page(bank, q, s)
        spans, ev, summary, gate = page[:4]
        hist = torch.cat((self.begin.expand(len(q), 1, -1), bank[y[:, :-1].long()]), 1)
        states, _ = self.gru(hist, summary[None].contiguous())
        state, spell, runs = self.actions(states, previous_marks(y), s, page)
        pred = (
            self.glyph_scores(state, bank, n)
            .gather(-1, (y.long() - 1).clamp(0, n - 2)[:, :, None])
            .squeeze(-1)
        )
        pred = pred.masked_fill(y >= n, -1e9)
        mix = self.free_mix(state).float().squeeze(-1)
        mark = torch.logaddexp(F.logsigmoid(mix) + pred, F.logsigmoid(-mix) + base_lp)
        arcs = [spell + mark]
        for weight, match in zip(runs, run_matches(s, y), strict=True):
            arcs.append(weight.masked_fill(~match, -1e9).logsumexp(-1))
        composed = PathMass.apply(torch.stack(arcs, -1), (y > 0).sum(-1))
        whole = spans.masked_fill(~compatible(s, y), -1e9).flatten(1).logsumexp(-1)
        return torch.logaddexp(
            F.logsigmoid(gate) + whole, F.logsigmoid(-gate) + composed
        ), ev

"""Score each physical glyph as skipped, the beginning of a fact, or its continuation."""

import numpy as np
import torch
from numba import njit
from torch import nn
from ilm.visual_lm.paired_evidence import EvidenceWriter

MAX_RUNS = 16


class RunReader(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        # Only deploy layers that this reader actually uses.
        base = base if base is not None else EvidenceWriter()
        for name in ("ink", "context", "ask", "fuse", "reason", "evidence", "summary"):
            setattr(self, name, getattr(base, name))
        self.roles = nn.Sequential(nn.Linear(160, 96), nn.GELU(), nn.Linear(96, 3))
        self.count = nn.Sequential(
            nn.Linear(162, 96), nn.GELU(), nn.Linear(96, MAX_RUNS + 1)
        )

    def forward(self, bank, questions, pages):
        key, _, summary, mask = EvidenceWriter.read_features(
            self, bank, questions, pages
        )
        roles = self.roles(
            torch.cat((key, summary[:, None].expand_as(key)), -1)
        ).float()
        pr = roles.softmax(-1) * mask[:, :, None]
        mass = torch.stack((pr[:, :, 1].sum(1), pr[:, :, 2].sum(1)), 1).log1p()
        counts = self.count(
            torch.cat(
                (
                    summary,
                    key.masked_fill(~mask[:, :, None], -1e4).amax(1),
                    mass.to(summary.dtype),
                ),
                -1,
            )
        ).float()
        return roles, counts


@njit(cache=True)
def best_path(scores, counts):
    """Viterbi for exact signed integer utilities. A new run can follow another run."""
    n = len(scores)
    kmax = len(counts) - 1
    neg = np.int64(-(1 << 57))
    prev = np.full((kmax + 1, 2), neg, np.int64)
    prev[0, 0] = 0
    back = np.zeros((n, kmax + 1, 2), np.int8)
    for t in range(n):
        cur = np.full_like(prev, neg)
        for k in range(min(t + 1, kmax) + 1):
            a = 0 if prev[k, 0] >= prev[k, 1] else 1
            if prev[k, a] != neg:
                cur[k, 0] = prev[k, a] + scores[t, 0]
                back[t, k, 0] = a
            if prev[k, 1] != neg:
                cur[k, 1] = prev[k, 1] + scores[t, 2]
                back[t, k, 1] = 2
            if k:
                a = 0 if prev[k - 1, 0] >= prev[k - 1, 1] else 1
                if prev[k - 1, a] != neg:
                    v = prev[k - 1, a] + scores[t, 1]
                    if v > cur[k, 1]:
                        cur[k, 1] = v
                        back[t, k, 1] = 3 + a
        prev = cur
    value = neg
    kbest = 0
    active = 0
    for k in range(min(n, kmax) + 1):
        for a in range(2):
            if prev[k, a] != neg and prev[k, a] + counts[k] > value:
                value = prev[k, a] + counts[k]
                kbest = k
                active = a
    tags = np.zeros(n, np.int8)
    for t in range(n - 1, -1, -1):
        choice = back[t, kbest, active]
        if active:
            if choice == 2:
                tags[t] = 2
            else:
                tags[t] = 1
                kbest -= 1
                active = choice - 3
        else:
            active = choice
    return tags, value


def intervals(role_logits, count_logits, mode="count"):
    if mode not in ("count", "free", "single"):
        raise ValueError("Unknown decoding mode")
    if not np.isfinite(role_logits).all() or not np.isfinite(count_logits).all():
        raise ValueError("Nonfinite optical scores")
    score = np.rint(np.asarray(role_logits, dtype=np.float64) * 1000).astype(np.int64)
    prior = np.rint(np.asarray(count_logits, dtype=np.float64) * 1000).astype(np.int64)
    if mode == "free":
        prior = np.zeros_like(prior)
    if mode == "single":
        prior = prior[:2]
    tags, _ = best_path(score, prior)
    out = []
    for i, tag in enumerate(tags):
        if tag == 1:
            out.append([i, i + 1])
        elif tag == 2:
            out[-1][1] = i + 1
    return out

"""A page-wide latent sign for binding a collection of visible facts."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.multiple_ink_runs import best_path
from ilm.visual_lm.paired_evidence import EvidenceWriter
from ilm.visual_lm.run_survival import utilities


class BindingHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 2))


@torch.no_grad()
def frozen_features(reader, bank, q, p):
    key, _, summary, mask = EvidenceWriter.read_features(reader, bank, q, p)
    features = torch.cat((key, summary[:, None].expand_as(key)), dim=-1)
    roles, counts = reader(bank, q, p)
    return features, utilities(roles, mask), counts


def pack_paths(paths):
    counts = [[len(p) for p in row] for row in paths]
    shape = (len(paths), len(paths[0]), max(max(row) for row in counts), 2)
    bounds = np.zeros(shape, dtype=np.int64)
    for i, row in enumerate(paths):
        for j, path in enumerate(row):
            bounds[i, j, : len(path)] = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    return bounds


def path_gains(gains, bounds):
    """Start gain plus the sum of interior gains relative to skipping."""
    first, end = bounds.unbind(-1)
    valid = end > first
    batch = torch.arange(len(first), device=first.device)[:, None, None]
    prefix = F.pad(gains[:, :, 1].cumsum(dim=1), (1, 0))
    score = gains[batch, first, 0] + prefix[batch, end] - prefix[batch, first + 1]
    return (score * valid).sum(-1)


def score_paths(role, count, head, bounds, mixture):
    raw = path_gains(head, bounds)
    k = (bounds[..., 1] > bounds[..., 0]).sum(-1)
    base = path_gains(role[..., 1:] - role[..., :1], bounds) + count.gather(1, k)
    return base + (raw.abs() if mixture else raw), raw


def fixed(x):
    return torch.round(x.double() * 1000).long()


def decode(role, count, head, mixture):
    """Find the best unrestricted path over already quantized int64 utilities."""
    best = None
    value = None
    for sign in (1, -1) if mixture else (1,):
        scores = role.copy()
        scores[:, 1:] += sign * head
        tags, utility = best_path(scores, count)
        if value is None or utility > value:
            best, value = tags, utility
    spans = []
    for i, tag in enumerate(best):
        if tag == 1:
            spans.append([i, i + 1])
        elif tag == 2:
            spans[-1][1] = i + 1
    return spans, int(value)

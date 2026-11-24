"""A null-or-contiguous-interval model over a pictured paragraph."""

import torch
from torch import nn
from torch.nn import functional as F


def endpoints(start, end, interior):
    # The interval [i,j] earns start[i] + end[j] + sum(interior[i:j+1]).
    prefix = F.pad(interior.float().cumsum(-1), (1, 0))
    return start.float() - prefix[..., :-1], end.float() + prefix[..., 1:]


def normalizer(left, right, null, lengths):
    mask = (
        torch.arange(left.size(-1), device=left.device)[None, None, :]
        < lengths[:, None, None]
    )
    total = torch.logsumexp(
        (torch.logcumsumexp(left, -1) + right).masked_fill(~mask, -torch.inf), -1
    )
    return torch.logaddexp(null.float(), total)


def interval_loss(outputs, lengths, bounds):
    left, right, null = outputs
    a, b = bounds.unbind(-1)

    def at(x, t):
        return x.gather(-1, t.clamp_min(0).unsqueeze(-1)).squeeze(-1)

    target = torch.where(a >= 0, at(left, a) + at(right, b - 1), null)
    return (normalizer(left, right, null, lengths) - target).mean()


def select(outputs, lengths):
    left, right, null = outputs
    running, ix = left.cummax(-1)
    mask = (
        torch.arange(left.size(-1), device=left.device)[None, None, :]
        < lengths[:, None, None]
    )
    score, last = (running + right).masked_fill(~mask, -torch.inf).max(-1)
    first = ix.gather(-1, last[..., None]).squeeze(-1)
    result = torch.stack((first, last + 1), -1)
    return result.masked_fill((score <= null)[..., None], -1)


class ProseFields(nn.Module):
    def __init__(self, fields, domains, entity=False):
        super().__init__()
        width = 80
        self.visual = nn.Linear(256, width)
        self.entity = nn.Linear(3, width, bias=False) if entity else None
        self.domain = nn.Embedding(domains, width)
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 1)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.convs)
        self.scores = nn.Linear(width, 3 * fields)
        self.null = nn.Linear(width, fields)

    def forward(self, cells, lengths, domains, anchor=None):
        mask = (
            torch.arange(cells.size(1), device=cells.device)[None, :] < lengths[:, None]
        )
        state = self.visual(cells) + self.domain(domains)[:, None]
        if self.entity is not None:
            state = state + self.entity(anchor)
        state = F.gelu(state)
        for conv, norm in zip(self.convs, self.norms):
            local = conv((state * mask[..., None]).transpose(1, 2)).transpose(1, 2)
            state = norm(state + F.gelu(local))
        a, b, inside = self.scores(state).transpose(1, 2).chunk(3, 1)
        left, right = endpoints(a, b, inside.tanh())
        return (
            left,
            right,
            self.null((state * mask[..., None]).sum(1) / lengths[:, None]),
        )


def name_position(length, start, size):
    """Three visual alignment features at each cell, or zeros for no unique match."""
    import numpy as np

    result = np.zeros((length, 3), np.float32)
    if start is None:
        return result
    steps = np.arange(length)
    delta = np.minimum(steps - start, 0) + np.maximum(steps - (start + size - 1), 0)
    result[:, 0] = delta == 0
    result[:, 1] = np.sign(delta)
    result[:, 2] = np.exp(-np.abs(delta) / 10)
    return result


def pictured_name(cells, name):
    import numpy as np

    if not len(name) or len(name) > len(cells):
        return None
    found = [
        int(i)
        for i in np.flatnonzero((cells == name[0]).all(1))
        if i + len(name) <= len(cells)
        and np.array_equal(cells[i : i + len(name)], name)
    ]
    return found[0] if len(found) == 1 else None

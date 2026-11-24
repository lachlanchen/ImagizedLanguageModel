"""Score alternative source spans of a frozen optical reader."""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def candidates(reader, features, query, page, count=32):
    """Expose the same question-conditioned states used by the original endpoints."""
    states = []
    hook = reader.boundaries.register_forward_pre_hook(
        lambda _, args: states.append(args[0])
    )
    try:
        scores = reader(features, query, page)
    finally:
        hook.remove()
    flat = scores.flatten(1)
    k = min(count, flat.shape[1])
    rank = flat.argsort(descending=True, stable=True)[:, :k]
    base = flat.gather(1, rank)
    valid = base.isfinite()
    start = torch.where(valid, rank // scores.shape[-1], 0)
    length = torch.where(valid, rank % scores.shape[-1] + 1, 1)
    batch = torch.arange(len(page), device=page.device)[:, None]
    h = states[0]
    end = start + length - 1
    sums = F.pad(h.float().cumsum(1), (0, 0, 1, 0))
    interior = (sums[batch, end + 1] - sums[batch, start]) / length.unsqueeze(-1)
    embedding = torch.stack((h[batch, start], h[batch, end], interior), dim=2).half()
    return embedding, base, start, length


class SpanRanker(nn.Module):
    def __init__(self, mode, width=128, hidden=96):
        super().__init__()
        if mode not in ("endpoints", "whole"):
            raise ValueError("Unknown span score")
        self.mode = mode
        self.left = nn.Linear(width, hidden)
        self.right = nn.Linear(width, hidden)
        self.extra = nn.Linear(width, hidden)
        self.length = nn.Embedding(257, width)
        self.out = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, embeddings, base, lengths):
        left, right, interior = embeddings.float().unbind(2)
        a, b = self.left(left), self.right(right)
        size = self.length(lengths)
        if self.mode == "endpoints":
            h = F.gelu(a) + F.gelu(b) + F.gelu(self.extra(size))
        else:
            h = F.gelu(a + b + self.extra(size + interior))
        return base + self.out(h).squeeze(-1)


def candidate_loss(scores, correct):
    """Conditional likelihood on supported lists. Unsupported rows have no gradient."""
    valid = correct.any(1)
    active = scores[valid]
    positive = active.masked_fill(~correct[valid], -torch.inf)
    return (active.logsumexp(1) - positive.logsumexp(1)).sum() / valid.sum().clamp_min(
        1
    )

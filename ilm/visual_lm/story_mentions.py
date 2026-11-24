"""A query-conditioned distribution over short, earlier strips of source ink."""

import torch
from torch import nn
from torch.nn import functional as F

MAX_SPAN = 24


class StoryMentions(nn.Module):
    def __init__(self, width=80):
        super().__init__()
        self.visual = nn.Linear(256, width)
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in (1, 3, 7)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.convs)
        self.attention = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.context = nn.LayerNorm(width)
        self.boundary = nn.Linear(width, 2)
        self.content = nn.Linear(width, 2 * width)
        self.question = nn.Linear(width, 2 * width)
        self.distance = nn.Sequential(nn.Linear(3, 24), nn.GELU(), nn.Linear(24, 2))
        self.length = nn.Linear(width, MAX_SPAN)
        self.null = nn.Linear(width, 1)

    def forward(self, cells, bounds, lengths):
        batch, size, _ = cells.shape
        pos = torch.arange(size, device=cells.device)[None]
        visible = pos < lengths[:, None]
        inside = (pos >= bounds[:, 0, None]) & (pos < bounds[:, 1, None])
        h = F.gelu(self.visual(cells))
        for conv, norm in zip(self.convs, self.norms):
            local = conv((h * visible[..., None]).transpose(1, 2)).transpose(1, 2)
            h = norm(h + F.gelu(local))
        q, k, v = (
            self.attention(h)
            .reshape(batch, size, 3, 4, -1)
            .permute(2, 0, 3, 1, 4)
            .unbind()
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=visible[:, None, None, :]
        )
        h = self.context(h + self.output(attended.transpose(1, 2).reshape_as(h)))
        query = (h * inside[..., None]).sum(1) / inside.sum(1, keepdim=True).clamp_min(
            1
        )
        edges = self.boundary(h)
        features = self.content(h).reshape(batch, size, 2, -1)
        question = self.question(query).reshape(batch, 1, 2, -1)
        role = (features * question).sum(-1) / features.size(-1) ** 0.5
        delta = (bounds[:, 0, None] - pos).float().clamp_min(0)
        distance = torch.stack(
            (delta / 288, torch.exp(-delta / 8), torch.exp(-delta / 55)), dim=-1
        )
        start, end = (edges + role + self.distance(distance)).unbind(-1)
        endings = F.pad(end, (0, MAX_SPAN - 1)).unfold(1, MAX_SPAN, 1)
        spans = start[:, :, None] + endings + self.length(query)[:, None]
        offsets = torch.arange(MAX_SPAN, device=cells.device)
        valid = pos[:, :, None] + offsets < bounds[:, None, 0:1]
        spans = spans.masked_fill(~valid, -1e4)
        return torch.cat((spans.flatten(1), self.null(query)), 1), edges


def losses(logits, edges, correct, candidates, bounds):
    """Marginalize over coreferent printed names, including the no-name outcome."""
    n, size, _ = edges.shape
    truth = torch.cat((correct.flatten(1), ~correct.flatten(1).any(1, keepdim=True)), 1)
    main = (logits.logsumexp(1) - logits.masked_fill(~truth, -1e4).logsumexp(1)).mean()
    linked = correct.any((1, 2))
    choices = torch.cat((candidates.flatten(1), torch.ones_like(linked[:, None])), 1)
    ranking = (
        logits.masked_fill(~choices, -1e4).logsumexp(1)
        - logits.masked_fill(~truth, -1e4).logsumexp(1)
    ).mean()
    starts = candidates.any(-1)
    ends = torch.zeros((n, size + MAX_SPAN), device=edges.device, dtype=torch.int32)
    at = torch.arange(size, device=edges.device)[:, None] + torch.arange(
        MAX_SPAN, device=edges.device
    )
    ends.scatter_add_(1, at.flatten()[None].expand(n, -1), candidates.flatten(1).int())
    targets = torch.stack((starts, ends[:, :size] > 0), dim=-1).float()
    earlier = torch.arange(size, device=edges.device)[None, :] < bounds[:, 0, None]
    aux = F.binary_cross_entropy_with_logits(
        edges, targets, pos_weight=edges.new_tensor(8.0), reduction="none"
    )
    return (
        main
        + 0.5 * ranking
        + 0.25 * (aux * earlier[..., None]).sum() / (2 * earlier.sum().clamp_min(1))
    )

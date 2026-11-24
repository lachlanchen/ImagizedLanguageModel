"""Causal four-glyph page model with a visual shape density and a source pointer."""

import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.book_prefix import CausalBlock


class DiscourseWriter(nn.Module):
    block = 4
    context = 256

    def __init__(self, width=192, depth=6, shape_dim=96):
        super().__init__()
        self.shape = nn.Sequential(
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Linear(192, shape_dim),
            nn.LayerNorm(shape_dim),
        )
        self.boundary = nn.Parameter(torch.randn(2, shape_dim) * 0.02)
        self.patch = nn.Linear(self.block * shape_dim, width)
        self.pos = nn.Parameter(torch.randn(self.context // self.block, width) * 0.02)
        self.body = nn.Sequential(
            *(CausalBlock(width) for _ in range(depth)), nn.LayerNorm(width)
        )
        self.phase = nn.Parameter(torch.randn(self.block, width) * 0.02)
        self.local = nn.Sequential(
            nn.Linear(3 * shape_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, shape_dim),
        )
        self.pointer_q = nn.Linear(width, 64)
        self.pointer_k = nn.Linear(shape_dim + width, 64)
        self.gate = nn.Linear(width, 1)

    def encode(self, features):
        return torch.cat(
            (self.boundary.to(features.dtype), self.shape(features[2:])), 0
        )

    def distributions(self, bank, ids, n_shapes, positions=None):
        """Prediction at j can read only input cells before j. Id 0 pads; 1 stops."""
        b, t = ids.shape
        if not 1 <= t <= self.context:
            raise ValueError("Page window outside the model context")
        x = bank[ids.long()]
        padded = F.pad(x, (0, 0, 0, (-t) % 4))
        patches = padded.reshape(b, -1, x.shape[-1] * 4)
        prior = F.pad(self.patch(patches)[:, :-1], (0, 0, 1, 0))
        h = self.body(prior + self.pos[: prior.size(1)])
        cells = padded.reshape(b, -1, 4, x.shape[-1])
        predecessors = []
        for lag in (3, 2, 1):
            predecessors.append(F.pad(cells[:, :, :-lag], (0, 0, lag, 0)))
        local = self.local(torch.cat(predecessors, -1))
        h = (h[:, :, None, :] + self.phase + local).reshape(b, -1, h.shape[-1])[:, :t]
        key = self.pointer_k(torch.cat((h, x), -1))
        absolute = torch.arange(t, device=x.device)
        if positions is None:
            where = absolute.expand(b, -1)
        else:
            where = positions
            h = h.gather(1, where[..., None].expand(-1, -1, h.size(-1)))
        logits = self.readout(h) @ bank[1:n_shapes].T / bank.shape[-1] ** 0.5
        attention = self.pointer_q(h) @ key.transpose(1, 2) / 8
        visible = (absolute[None, None, :] < where[:, :, None]) & (ids[:, None, :] >= 2)
        pointer = attention.float().masked_fill(~visible, -1e8).softmax(-1) * visible
        gate = self.gate(h).float().sigmoid() * visible.any(-1, keepdim=True)
        return logits.float(), pointer, gate.squeeze(-1)

    def log_prob(self, bank, ids, n_shapes, positions=None):
        logits, copy, gate = self.distributions(bank, ids, n_shapes, positions)
        observed = ids if positions is None else ids.gather(1, positions)
        index = (observed.long() - 1).clamp(0, logits.shape[-1] - 1)
        log_shape = logits.log_softmax(-1).gather(-1, index[..., None]).squeeze(-1)
        shape = log_shape.exp() * (observed > 0) * (observed < n_shapes)
        repeat = (copy * (observed[:, :, None] == ids[:, None, :])).sum(-1)
        mass = (1 - gate) * shape + gate * repeat
        return mass.clamp_min(1e-30).log()

    def next_mass(self, bank, ids, n_shapes):
        """ids ends with one unobserved placeholder. Copy can return novel ink."""
        p = torch.full(
            (ids.size(0), 1), ids.size(1) - 1, device=ids.device, dtype=torch.long
        )
        logits, copy, g = self.distributions(bank, ids, n_shapes, p)
        result = bank.new_zeros((len(ids), len(bank)), dtype=torch.float32)
        result[:, 1:n_shapes] = logits[:, 0].softmax(-1) * (1 - g)
        return result.scatter_add_(1, ids.long(), copy[:, 0] * g)

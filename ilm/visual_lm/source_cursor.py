"""A visual writer that marginalizes repeated source ink and a free emission path."""

import torch
from torch import nn
from torch.nn import functional as F


class SourceCursor(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.shape = nn.Sequential(
            nn.Linear(256, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.patch = nn.Linear(4 * width, width)
        self.position = nn.Parameter(torch.randn(192, width) * 0.025)
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width, 4, 3 * width, dropout=0, batch_first=True, norm_first=True
                )
                for _ in range(3)
            ]
        )
        self.cells = nn.Linear(width, 4 * width)
        self.local = nn.Conv1d(width, width, 5, padding=2)
        self.norm = nn.LayerNorm(width)
        self.pool = nn.Linear(width, 1)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.start = nn.Linear(width, width)
        self.advance = nn.Linear(width, width)
        self.endpoint = nn.Linear(width, 1)
        self.gates = nn.Linear(width, 3)
        self.free = nn.Linear(width, width)
        self.begin = nn.Parameter(torch.zeros(width))

    def page(self, bank, ids):
        observed = ids >= 2
        cells = bank[ids.long()] * observed[..., None]
        cells = F.pad(cells, (0, 0, 0, (-ids.size(1)) % 4))
        groups = (
            F.pad(observed, (0, (-ids.size(1)) % 4)).reshape(len(ids), -1, 4).any(-1)
        )
        patches = (
            self.patch(cells.reshape(len(ids), len(groups[0]), -1))
            + self.position[: groups.size(1)]
        )
        for layer in self.encoder:
            patches = layer(patches, src_key_padding_mask=~groups)
        local = self.local(cells.transpose(1, 2)).transpose(1, 2)
        keys = self.norm(cells + local + self.cells(patches).reshape_as(cells))[
            :, : ids.size(1)
        ]
        attention = (
            self.pool(keys).squeeze(-1).float().masked_fill(~observed, -1e9).softmax(-1)
        )
        return keys, torch.bmm(attention[:, None, :].to(keys.dtype), keys).transpose(
            0, 1
        ).contiguous()

    def emissions(self, states, keys, observed):
        scale = keys.size(-1) ** -0.5
        scores = self.start(states) @ keys.transpose(1, 2) * scale
        starts = scores.float().masked_fill(~observed[:, None, :], -1e9).softmax(-1)
        advance = (
            (
                self.advance(states) @ keys.transpose(1, 2) * scale
                + self.endpoint(keys).transpose(1, 2)
            )
            .float()
            .sigmoid()
        )
        beta, stop, custom = self.gates(states).float().sigmoid().unbind(-1)
        return starts, advance, beta, stop, custom

    @staticmethod
    def paths(previous, advance, starts, beta, observed, persistent):
        if persistent:
            carry = F.pad(previous[:, :-1] * advance[:, :-1], (1, 0)) * observed
        else:
            carry = torch.zeros_like(previous)
        reset = (1 - carry.sum(-1)).clamp(0, 1)
        return reset * (1 - beta), carry + (reset * beta)[:, None] * starts

    def likelihood(self, features, ids, targets, base_mass, n, persistent):
        bank = self.shape(features)
        keys, initial = self.page(bank, ids)
        past = torch.cat(
            (self.begin.expand(len(ids), 1, -1), bank[targets[:, :-1].long()]), 1
        )
        states, _ = self.recurrent(past, initial)
        starts, adv, beta, stop, custom = self.emissions(states, keys, ids >= 2)
        logits = (self.free(states) @ bank[2:n].T * keys.size(-1) ** -0.5).float()
        glyph = (
            logits.log_softmax(-1)
            .gather(-1, (targets.long() - 2).clamp(0, n - 3)[..., None])
            .squeeze(-1)
            .exp()
        )
        free = (1 - custom) * base_mass + custom * glyph * (
            (targets >= 2) & (targets < n)
        )
        matches = ids[:, None, :] == targets[:, :, None]
        posterior = torch.zeros_like(starts[:, 0])
        logs = []
        for t in range(targets.size(1)):
            w, s = self.paths(
                posterior, adv[:, t], starts[:, t], beta[:, t], ids >= 2, persistent
            )
            mass = s * matches[:, t] * (1 - stop[:, t, None])
            den = (
                stop[:, t] * (targets[:, t] == 1)
                + (1 - stop[:, t]) * w * free[:, t]
                + mass.sum(-1)
            ).clamp_min(1e-30)
            logs.append(den.log())
            posterior = mass / den[:, None]
        return torch.stack(logs, 1)

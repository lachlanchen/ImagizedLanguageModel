"""Question-conditioned extraction from a pair of cell-aligned page images."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.gloss_focus import cell_features


def word_runs(cells):
    """Half-open intervals of ink, split only at physical white cells."""
    if cells.ndim != 2 or cells.shape[1] != 128 or cells.dtype != np.uint8:
        raise ValueError("Expected packed 32x32 monochrome cells")
    ink = cells.any(axis=1)
    transitions = np.flatnonzero(np.diff(np.r_[False, ink, False]))
    return transitions.reshape(-1, 2)


class OpticalPassageQA(nn.Module):
    width = 112
    max_span = 16
    cuts = 24

    def __init__(self):
        super().__init__()
        # Fixed visual projections retain letter shape, boundary punctuation and length.
        rng = torch.Generator().manual_seed(41093)
        self.register_buffer("sketch", torch.randn(256, 16, generator=rng) / 8)
        self.project = nn.Sequential(
            nn.Linear(115, self.width), nn.LayerNorm(self.width), nn.GELU()
        )
        self.local = nn.Conv1d(self.width, self.width, 3, padding=1)
        self.query_pool = nn.Linear(self.width, 1)
        self.key = nn.Linear(self.width, 64, bias=False)
        self.query = nn.Linear(self.width, 64, bias=False)
        self.combine = nn.Sequential(
            nn.Linear(4 * self.width + 1, self.width),
            nn.LayerNorm(self.width),
            nn.GELU(),
        )
        self.context = nn.ModuleList(
            nn.Conv1d(self.width, self.width, 3, dilation=d, padding=d)
            for d in (1, 2, 4)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(self.width) for _ in self.context)
        self.endpoints = nn.Linear(self.width, 2)
        self.trim = nn.Linear(self.width, 2 * self.cuts)

    @torch.no_grad()
    def surface(self, words, lengths, optical):
        """Each padded word is an array of original packed glyph images."""
        f = cell_features(words)
        limit = optical.width
        canonical = F.pad(f[:, :limit], (0, 0, 0, max(0, limit - f.size(1))))
        meaning = optical(optical.cell(canonical))
        last = lengths - 1
        places = torch.stack(
            (torch.zeros_like(last), last.clamp(max=1), (last - 1).clamp(min=0), last),
            1,
        )
        edges = (
            f[torch.arange(len(words), device=f.device)[:, None], places] @ self.sketch
        )
        return torch.cat(
            (meaning, edges.flatten(1), lengths[:, None].float().log1p() / 3), 1
        )

    @staticmethod
    def mask(x, sizes):
        return torch.arange(x.size(1), device=x.device)[None] < sizes[:, None]

    def encode(self, x, mask):
        h = self.project(x) * mask[..., None]
        return (h + F.gelu(self.local(h.transpose(1, 2)).transpose(1, 2))) * mask[
            ..., None
        ]

    def forward(self, passage, psizes, question, qsizes):
        pm, qm = self.mask(passage, psizes), self.mask(question, qsizes)
        p, q = self.encode(passage, pm), self.encode(question, qm)
        pool = self.query_pool(q).squeeze(-1).masked_fill(~qm, -1e4).softmax(1)
        global_q = (pool[:, :, None] * q).sum(1, keepdim=True).expand_as(p)
        lexical = passage[:, :, :50] @ question[:, :, :50].transpose(1, 2)
        sim = self.key(p) @ self.query(q).transpose(1, 2) / math.sqrt(64) + 3 * lexical
        aligned = sim.masked_fill(~qm[:, None], -1e4).softmax(-1) @ q
        overlap = lexical.masked_fill(~qm[:, None], -1).amax(2, keepdim=True)
        h = self.combine(torch.cat((p, aligned, p * aligned, global_q, overlap), -1))
        for conv, norm in zip(self.context, self.norms):
            z = (h * pm[:, :, None]).transpose(1, 2)
            h = norm(h + F.gelu(conv(z).transpose(1, 2)))
        return self.endpoints(h).masked_fill(~pm[:, :, None], -1e4), self.trim(
            h
        ).unflatten(-1, (2, self.cuts))

    def select(self, endpoints, trim, sizes, widths):
        n, w, _ = endpoints.shape
        pos = torch.arange(w, device=sizes.device)
        end = pos[:, None] + torch.arange(self.max_span, device=sizes.device)
        valid = end[None] < sizes[:, None, None]
        pair = endpoints[:, :, 0, None] + endpoints[:, end.clamp(max=w - 1), 1]
        ix = pair.masked_fill(~valid, -torch.inf).flatten(1).argmax(-1)
        b = torch.arange(n, device=sizes.device)
        left, right = ix // self.max_span, ix // self.max_span + ix % self.max_span
        offsets = torch.arange(self.cuts, device=sizes.device)
        lscore, rscore = trim[b, left, 0], trim[b, right, 1]
        legal = (offsets[None, :, None] < widths[b, left, None, None]) & (
            offsets[None, None, :] < widths[b, right, None, None]
        )
        legal &= (left != right)[:, None, None] | (
            offsets[:, None] + offsets[None, :] < widths[b, left, None, None]
        )
        best = (
            (lscore[:, :, None] + rscore[:, None, :])
            .masked_fill(~legal, -torch.inf)
            .flatten(1)
            .argmax(1)
        )
        return torch.stack((left, right, best // self.cuts, best % self.cuts), 1)

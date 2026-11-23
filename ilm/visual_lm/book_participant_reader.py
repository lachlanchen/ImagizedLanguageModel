"""Read one pictured question and assign probability to each distinct source strip."""

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 8
CELL = 128


class InkBank:
    """An input cache of raw bitmaps. The network has no glyph-indexed parameters."""

    def __init__(self):
        self.cells = [bytes(CELL), bytes(CELL)]
        self.lookup = {bytes(CELL): 1}

    def encode(self, raw):
        if not raw or len(raw) % CELL:
            raise ValueError("Expected a nonempty row of packed 32-pixel cells")
        ids = []
        for start in range(0, len(raw), CELL):
            cell = raw[start : start + CELL]
            if cell not in self.lookup:
                self.lookup[cell] = len(self.cells)
                self.cells.append(cell)
            ids.append(self.lookup[cell])
        return ids

    def raster(self, device):
        packed = np.frombuffer(b"".join(self.cells), dtype=np.uint8).reshape(-1, CELL)
        pixels = torch.from_numpy(np.unpackbits(packed, axis=1)).float().to(device)
        return F.avg_pool2d(pixels.view(-1, 1, 32, 32), 2).flatten(1)


def positions(page, candidate):
    n = len(candidate)
    return (
        [
            i * WIDTH + n - 1
            for i in range(len(page) - n + 1)
            if page[i : i + n] == candidate
        ]
        if 0 < n <= WIDTH
        else []
    )


def padded(rows, device):
    pad = torch.nn.utils.rnn.pad_sequence
    return pad([torch.tensor(s, dtype=torch.long) for s in rows], batch_first=True).to(
        device
    )


def literal_evidence(page, question):
    """Give more credit to rare matched glyphs and consecutive matches."""
    same = page[:, :, None].eq(question[:, None, :]) & page[:, :, None].gt(1)
    count = same.sum(dim=1, keepdim=True).clamp_min(1)
    matches = same / count
    consecutive = F.pad(same[:, :-1, :-1] & same[:, 1:, 1:], (0, 1, 0, 1))
    adjacent = consecutive.float() + F.pad(consecutive[:, :-1, :-1], (1, 0, 1, 0))
    return (matches * (1 + adjacent)).sum(-1)


def neighborhood(evidence, width):
    return F.avg_pool1d(
        evidence.unsqueeze(1), 2 * width + 1, stride=1, padding=width
    ).squeeze(1)


def end_values(x):
    return F.pad(x, (0, WIDTH - 1)).unfold(-1, WIDTH, 1)


def valid_spans(page):
    return end_values(page).ne(0)


def literal_spans(p, q):
    s = literal_evidence(p, q)
    local = 2 * neighborhood(s, 4) + neighborhood(s, 12)
    return ((local.unsqueeze(-1) + end_values(local)) * 0.5).masked_fill(
        ~valid_spans(p), -1e4
    )


class VisualContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv1d(dim, dim, 3, dilation=k, padding=k) for k in (1, 2, 4)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in self.layers)

    def forward(self, x, mask):
        for conv, norm in zip(self.layers, self.norms):
            x = x * mask.unsqueeze(-1)
            x = norm(x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2)))
        return x * mask.unsqueeze(-1)


class BookParticipantReader(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim), nn.GELU())
        self.context = VisualContext(dim)
        self.question_weight = nn.Linear(dim, 1)
        self.cross = nn.Linear(4 * dim + 3, dim)
        self.read = VisualContext(dim)
        self.start = nn.Linear(dim, WIDTH)
        self.end = nn.Linear(dim, 1)

    def forward(self, pixels, page, question):
        bank = self.ink(pixels)
        p = self.context(bank[page], page != 0)
        q = self.context(bank[question], question != 0)
        affinity = torch.bmm(
            F.normalize(p, dim=-1), F.normalize(q, dim=-1).transpose(1, 2)
        )
        attn = (5 * affinity).masked_fill(question[:, None, :] == 0, -1e4).softmax(-1)
        local_q = attn @ q
        salience = self.question_weight(q).squeeze(-1).masked_fill(question == 0, -1e4)
        global_q = (salience.softmax(-1).unsqueeze(1) @ q).expand_as(p)
        s = literal_evidence(page, question)
        fields = torch.stack((s, neighborhood(s, 4), neighborhood(s, 12)), dim=-1)
        x = self.cross(
            torch.cat((p, local_q, p * local_q, p * global_q, fields), dim=-1)
        )
        u = self.read(x, page != 0)
        logits = self.start(u) + end_values(self.end(u).squeeze(-1))
        return logits.masked_fill(~valid_spans(page), -1e4)


def group_spans(page):
    """Map distinct emitted bitmaps to every valid source occurrence."""
    groups = defaultdict(list)
    for i in range(len(page)):
        for n in range(1, min(WIDTH, len(page) - i) + 1):
            groups[tuple(page[i : i + n])].append(i * WIDTH + n - 1)
    return groups


def best_strip(scores, groups):
    flat = scores.flatten()
    best, value = None, -np.inf
    # Occurrence masses share one normalizer; sum in shifted probability units.
    mass = np.exp(flat.astype(np.float64) - flat.max())
    for name, indices in groups.items():
        total = sum(mass[j] for j in indices)
        if total > value:
            best, value = name, total
    return best

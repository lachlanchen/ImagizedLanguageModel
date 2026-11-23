"""Score pictured paragraphs from glyph appearance, local context and exact runs."""

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def tiles(strip):
    if not strip or len(strip) % 128:
        raise ValueError("Expected a nonempty strip of packed 32 by 32 glyphs")
    return [strip[i : i + 128] for i in range(0, len(strip), 128)]


def lexical_score(question, paragraph):
    match = SequenceMatcher(None, tiles(question), tiles(paragraph), autojunk=False)
    # Break longest-run ties by the total size of the nonoverlapping matches.
    runs = [part.size for part in match.get_matching_blocks()]
    return max(runs), sum(runs)


class PhysicalBank:
    """Integer addresses coalesce identical pixels within this batch or corpus."""

    def __init__(self):
        self.ids = {}
        self.sequences = {}

    def index(self, raw):
        if raw not in self.sequences:
            self.sequences[raw] = torch.tensor(
                [self.ids.setdefault(cell, len(self.ids) + 1) for cell in tiles(raw)],
                dtype=torch.long,
            )
        return self.sequences[raw]

    def pooled_ink(self, device):
        packed = np.frombuffer(bytes(128) + b"".join(self.ids), np.uint8).reshape(
            -1, 128
        )
        pixels = np.unpackbits(packed, axis=1).reshape(-1, 16, 2, 16, 2)
        return torch.from_numpy(
            pixels.mean((2, 4), dtype=np.float32).reshape(-1, 256)
        ).to(device)


def pad(rows, device):
    return pad_sequence(rows, batch_first=True, padding_value=0).to(device)


class ParagraphRanker(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256, width), nn.LayerNorm(width), nn.GELU())
        self.context = nn.Conv1d(width, width, 3, padding=1)
        self.query = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.query.weight)
        self.weight = nn.Linear(width, 1)
        self.matches = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.matches.weight, 1.0)

    def encode(self, bank, ids):
        x = self.ink(bank[ids]) * (ids != 0).unsqueeze(-1)
        return x + F.gelu(self.context(x.transpose(1, 2)).transpose(1, 2))

    def forward(self, bank, q, p):
        qh, ph = self.encode(bank, q), self.encode(bank, p)
        similarity = torch.bmm(
            F.normalize(self.query(qh), dim=-1), F.normalize(ph, dim=-1).transpose(1, 2)
        )
        best = similarity.masked_fill((p == 0)[:, None, :], -2.0).amax(-1)
        valid = q != 0
        importance = self.weight(qh).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        semantic = (importance * best).sum(-1)
        same = (
            (q[:, :, None] == p[:, None, :]) & valid[:, :, None] & (p[:, None, :] != 0)
        )
        run = same
        lengths = []
        for k in range(4):
            if k:
                run = F.pad(run[:, 1:, 1:], (0, 1, 0, 1)) & same
            lengths.append(run.any(-1).float().sum(-1) / valid.sum(-1))
        return 20 * semantic + self.matches(torch.stack(lengths, -1)).squeeze(-1)

    @classmethod
    def restore(cls, path, device="cpu"):
        model = cls()
        model.load_state_dict(load_file(str(Path(path))))
        return model.to(device).eval()

    @torch.inference_mode()
    def scores(self, question, paragraphs):
        if not paragraphs:
            raise ValueError("At least one paragraph is required")
        device = self.query.weight.device
        bank = PhysicalBank()
        q = bank.index(question)
        sources = [bank.index(p) for p in paragraphs]
        ink = bank.pooled_ink(device)
        result = []
        for start in range(0, len(sources), 8):
            take = sources[start : start + 8]
            result.extend(
                self(ink, pad([q] * len(take), device), pad(take, device))
                .cpu()
                .tolist()
            )
        return result

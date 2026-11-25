"""Sentence-level optical alignment during paired Mandarin/English writer training."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def read_sentence(writer, bank, cells):
    # Keep the stride-one writer's actual source encoder and its padding convention.
    features = bank[cells].masked_fill((cells == 0).unsqueeze(-1), 0)
    window = F.pad(features, (0, 0, 0, 3)).unfold(1, 4, 1).transpose(2, 3).flatten(2)
    hidden = writer.window(window) + writer.pos[: window.size(1)]
    valid = (cells != 0)[:, None, None, :]
    for layer in writer.enc:
        hidden, _ = layer(hidden, valid)
    context = [layer.cross.keys(hidden) for layer in writer.dec]
    mean = hidden.masked_fill((cells == 0).unsqueeze(-1), 0).sum(1) / cells.ne(0).sum(
        1, keepdim=True
    )
    return context, valid, mean


class BilingualTraining(nn.Module):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        width = writer.begin.size(-1)
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

    def forward(self, features, source, targets, inventories, local, identities, align):
        writer = self.writer
        bank = writer.optical(features)
        enc = [read_sentence(writer, bank, c) for c in source]
        sums = []
        counts = []
        for t, y in enumerate(targets):
            ctx, mask, _ = enc[1 - t]
            x = torch.cat((writer.begin[t].expand(len(y), 1, -1), bank[y[:, :-1]]), 1)
            h, _ = writer.decode(x, ctx, mask)
            logits = (
                h @ writer.prototypes(bank, inventories[t]).T / math.sqrt(h.size(-1))
            )
            labels = local[t][y]
            sums.append(
                F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
            )
            counts.append((labels != -100).sum())
        chars = sum(counts)
        ce = sum(sums) / chars
        if align:
            a, b = [F.normalize(self.project(v[2]).float(), dim=-1) for v in enc]
            scores = a @ b.T / 0.07
            same = (identities[:, 0, None] == identities[:, 0]) | (
                identities[:, 1, None] == identities[:, 1]
            )
            positives = scores.masked_fill(~same, -torch.inf)
            match = (
                sum(
                    (scores.logsumexp(i) - positives.logsumexp(i)).mean()
                    for i in (0, 1)
                )
                * 0.5
            )
        else:
            match = ce.new_zeros(())
        return ce, match, chars

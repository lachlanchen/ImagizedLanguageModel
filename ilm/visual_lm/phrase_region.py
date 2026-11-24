"""Emit learned runs of physical ink, while updating the language state at every glyph."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.region_writer import RegionWriter

LENGTHS = (1, 2, 4, 8, 16)


def source_runs(source):
    return [F.pad(source, (0, length - 1)).unfold(1, length, 1) for length in LENGTHS]


def extend_matches(matches, offset):
    """Double the tested run while respecting the end of either dimension."""
    t, s = matches.shape[1:]
    after = F.pad(matches, (0, offset, 0, offset))[
        :, offset : offset + t, offset : offset + s
    ]
    return matches & after


class PhraseRegionWriter(RegionWriter):
    def __init__(self, width=160):
        super().__init__(width)
        self.run_length = nn.Linear(width, len(LENGTHS))
        self.singleton_only = False

    def length_logprobs(self, state, valid):
        scores = self.run_length(state).float()
        available = torch.stack([v.any(-1) for v in valid], -1)
        return scores.masked_fill(~available[:, None, :], -torch.inf).log_softmax(-1)

    def loss(self, ctx, hidden, y):
        bank, _, src, _, _ = ctx
        inputs = torch.cat((self.bos.expand(len(y), 1, -1), bank[y[:, :-1]]), 1)
        state, _ = self.gru(inputs, hidden)
        history = torch.stack([F.pad(y, (i, 0))[:, : y.size(1)] for i in (1, 2, 3)], -1)
        spell, pointer, gate = self.distribution(ctx, state, history)
        valid = [r.ge(2).all(-1) for r in source_runs(src)]
        lengths = self.length_logprobs(state, valid)
        target = (y - 1).clamp(0, spell.shape[-1] - 1)
        spelling = spell.gather(-1, target.unsqueeze(-1)).squeeze(-1) + F.logsigmoid(
            gate
        )
        spelling = spelling.masked_fill(y > spell.shape[-1], -1e4)
        matches = y[:, :, None].eq(src[:, None, :]) & src[:, None, :].ge(2)
        for k, length in enumerate(LENGTHS):
            if k:
                matches = extend_matches(matches, length // 2)
            total = (pointer * valid[k][:, None, :]).sum(-1)
            aligned = (pointer * matches).sum(-1)
            # Finite logs keep gradients on masked/unavailable branches well-defined.
            copy = (
                F.logsigmoid(-gate)
                + lengths[:, :, k].clamp_min(-1e4)
                + aligned.clamp_min(1e-30).log()
                - total.clamp_min(1e-30).log()
            )
            copy = copy.masked_fill(~matches.any(-1), -1e4)
            if k == 0:
                correct = torch.logaddexp(spelling, copy)
            else:
                correct = torch.where(matches.any(-1), copy, correct)
        return -(correct * y.ne(0)).sum(1) / y.ne(0).sum(1)

    @torch.inference_mode()
    def write(self, ctx, hidden, limit=256):
        bank, concepts, src, _, _ = ctx
        batch = len(src)
        runs = source_runs(src)
        valid = [r.ge(2).all(-1) for r in runs]
        # Equal bitmap sequences are one output action even at different positions.
        device = src.device
        glyphs = torch.arange(len(bank), device=device)[:, None]
        options = [F.pad(glyphs, (0, LENGTHS[-1] - 1))]
        aliases = [src]
        if not self.singleton_only:
            for k in range(1, len(LENGTHS)):
                length = LENGTHS[k]
                words, inv = torch.unique(
                    runs[k].reshape(-1, length), dim=0, return_inverse=True
                )
                aliases.append(inv.reshape_as(src))
                options.append(F.pad(words, (0, LENGTHS[-1] - length)))
        sizes = [len(x) for x in options]
        units = torch.cat(options)
        pending = torch.zeros(batch, LENGTHS[-1], device=device, dtype=torch.long)
        past = torch.zeros(batch, 1, 3, device=device, dtype=torch.long)
        x = self.bos.expand(batch, 1, -1)
        ended = torch.zeros(batch, device=device, dtype=torch.bool)
        out = []
        for _ in range(limit):
            state, hidden = self.gru(x, hidden)
            lp, ptr, gate = self.distribution(ctx, state, past)
            pointer = ptr[:, 0]
            chance = self.length_logprobs(state, valid)[:, 0].exp()
            if self.singleton_only:
                chance = torch.ones_like(chance[:, :1])
            copy = (-gate).sigmoid()
            scores = []
            for k, alias in enumerate(aliases):
                locations = pointer * valid[k]
                locations = locations / locations.sum(1, keepdim=True).clamp_min(1e-30)
                mass = pointer.new_zeros(batch, sizes[k])
                mass.scatter_add_(1, alias, locations * chance[:, k : k + 1] * copy)
                if k == 0:
                    mass[:, 1 : len(concepts) + 2] += lp[:, 0].exp() * gate.sigmoid()
                scores.append(mass)
            proposal = units[torch.cat(scores, 1).argmax(1)]
            pending = torch.where(pending[:, :1].ne(0), pending, proposal)
            v = torch.where(ended, 1, pending[:, 0])
            pending = F.pad(pending[:, 1:], (0, 1))
            ended |= v == 1
            out.append(v)
            if ended.all():
                break
            past = torch.cat((v[:, None, None], past[:, :, :2]), -1)
            x = bank[v][:, None, :]
        return [
            (r[: r.index(1)], True) if 1 in r else (r, False)
            for r in torch.stack(out, 1).tolist()
        ]

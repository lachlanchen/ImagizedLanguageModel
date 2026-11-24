"""Supervise full pictured answers and vary how the writer learns their context."""

import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.region_writer import RegionWriter


def scoring_positions(target, start, length):
    """Return compressed history and equally many original positions, including STOP."""
    if start < 0 or length <= 0 or start + length >= len(target) or target[-1] != 1:
        raise ValueError("Fact must be inside the answer before STOP")
    compact = np.concatenate((target[:start], [0], target[start + length :]))
    count = len(compact) - 1
    if count < 2:
        raise ValueError("No surrounding ink")
    positions = np.arange(count, dtype=np.int64) * (len(target) - 1) // (count - 1)
    return compact, positions


class ContextWriter(RegionWriter):
    def position_nll(self, ctx, hidden, target):
        bank, _, src, _, _ = ctx
        inputs = torch.cat(
            [self.bos.expand(len(target), 1, -1), bank[target[:, :-1]]], 1
        )
        state, _ = self.gru(inputs, hidden)
        history = torch.stack(
            [F.pad(target, (n, 0))[:, : target.shape[1]] for n in (1, 2, 3)], -1
        )
        lp, cp, gate = self.distribution(ctx, state, history)
        spelling = lp.gather(
            -1, (target - 1).clamp(0, lp.size(-1) - 1).unsqueeze(-1)
        ).squeeze(-1)
        spelling = spelling.masked_fill(target > lp.size(-1), -1e4)
        copy = (cp * (src[:, None, :] == target[:, :, None])).sum(-1)
        return -torch.logaddexp(
            F.logsigmoid(gate) + spelling,
            F.logsigmoid(-gate) + copy.clamp_min(1e-30).log(),
        )

    def objective(self, mode, ctx, hidden, targets, aux_rows, auxiliary, control_mask):
        nll = self.position_nll(ctx, hidden, targets)
        mask = targets.ne(0)
        ordinary = ((nll * mask).sum(1) / mask.sum(1)).mean()
        if not len(aux_rows):
            return ordinary
        if mode == "direct":
            extra = (nll[aux_rows] * control_mask).sum(1)
        elif mode == "surround":
            bank, concept, src, value, key = ctx
            sub = (bank, concept, src[aux_rows], value[aux_rows], key[aux_rows])
            extra = (
                self.position_nll(sub, hidden[:, aux_rows].contiguous(), auxiliary)
                * auxiliary.ne(0)
            ).sum(1)
        else:
            raise ValueError(mode)
        return ordinary + 0.5 * (extra / control_mask.sum(1)).sum() / len(targets)

"""Marginalize evidence locations under complete, stopped physical-answer supervision."""

import torch
from torch.nn import functional as F


def stopped_log_probability(writer, ctx, hidden, target):
    """Exact spell/copy mixture on observed glyphs; target contains stop and optional zero padding."""
    bank, _, source, _, _ = ctx
    prev = torch.cat((writer.bos.expand(len(target), 1, -1), bank[target[:, :-1]]), 1)
    state, _ = writer.gru(prev, hidden)
    history = torch.stack(
        [F.pad(target, (i, 0))[:, : target.size(1)] for i in (1, 2, 3)], -1
    )
    spell, copy, gate = writer.distribution(ctx, state, history)
    selected = spell.gather(
        -1, (target - 1).clamp(0, spell.size(-1) - 1)[:, :, None]
    ).squeeze(-1)
    selected = selected.masked_fill(
        (target < 1) | (target > spell.size(-1)), -torch.inf
    )
    copied = (copy.double() * (source[:, None, :] == target[:, :, None])).sum(-1)
    logmass = torch.logaddexp(
        F.logsigmoid(gate.double()) + selected.double(),
        F.logsigmoid(-gate.double()) + copied.log(),
    )
    return logmass.masked_fill(target == 0, 0).sum(1)


def answer_marginal(logits, log_likelihood):
    """Log probability of each distinct accepted wording summed in likelihood beforehand."""
    logprior = logits.double().log_softmax(-1)
    joint = logprior + log_likelihood.double()
    return joint.logsumexp(-1)

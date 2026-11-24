"""Differentiate the likelihood of a complete stopped image over all optical segmentations."""

import math
import numba
import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.phrase_region import (
    LENGTHS,
    PhraseRegionWriter,
    extend_matches,
    source_runs,
)


@numba.njit(cache=True, nogil=True)
def forward_backward(logmass, lengths):
    batch, time, choices = logmass.shape
    logprob = np.zeros(batch, np.float64)
    occupancy = np.zeros_like(logmass)
    for b in range(batch):
        size = lengths[b]
        alpha = np.full(size + 1, -np.inf)
        beta = np.full(size + 1, -np.inf)
        alpha[0] = beta[size] = 0.0
        for i in range(size):
            for k in range(choices):
                j = i + (1 << k)
                if j <= size:
                    alpha[j] = np.logaddexp(alpha[j], alpha[i] + logmass[b, i, k])
        for i in range(size - 1, -1, -1):
            for k in range(choices):
                j = i + (1 << k)
                if j <= size:
                    beta[i] = np.logaddexp(beta[i], logmass[b, i, k] + beta[j])
        logprob[b] = alpha[size]
        if not np.isfinite(alpha[size]) or abs(alpha[size] - beta[0]) > 1e-7:
            raise ValueError(
                "Unsupported reference or inconsistent forward and backward probabilities"
            )
        for i in range(size):
            for k in range(choices):
                j = i + (1 << k)
                if j <= size:
                    occupancy[b, i, k] = math.exp(
                        alpha[i] + logmass[b, i, k] + beta[j] - alpha[size]
                    )
    return logprob, occupancy


class SegmentationLikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, lengths):
        values, weights = forward_backward(
            scores.detach().cpu().double().numpy(), lengths.cpu().numpy()
        )
        posterior = torch.as_tensor(weights, device=scores.device, dtype=scores.dtype)
        ctx.save_for_backward(posterior)
        return torch.as_tensor(values, device=scores.device, dtype=scores.dtype)

    @staticmethod
    def backward(ctx, grad):
        (posterior,) = ctx.saved_tensors
        return posterior * grad[:, None, None], None


class MarginalPhraseWriter(PhraseRegionWriter):
    def loss(self, ctx, hidden, y):
        bank, _, source, _, _ = ctx
        inputs = torch.cat((self.bos.expand(len(y), 1, -1), bank[y[:, :-1]]), 1)
        states, _ = self.gru(inputs, hidden)
        history = torch.stack([F.pad(y, (i, 0))[:, : y.size(1)] for i in (1, 2, 3)], -1)
        spell, ptr, gate = self.distribution(ctx, states, history)
        valid = [r.ge(2).all(-1) for r in source_runs(source)]
        lengthlog = self.length_logprobs(states, valid)
        ids = (y - 1).clamp(0, spell.size(-1) - 1)
        spelled = spell.gather(-1, ids.unsqueeze(-1)).squeeze(-1) + F.logsigmoid(gate)
        allowed = (y > 0) & (y <= spell.size(-1))
        matches = (y[:, :, None] == source[:, None, :]) & source[:, None, :].ge(2)
        edges = []
        for k, length in enumerate(LENGTHS):
            if k:
                matches = extend_matches(matches, length // 2)
            aligned = (ptr * matches).sum(-1)
            total = (ptr * valid[k][:, None, :]).sum(-1)
            copied = (
                F.logsigmoid(-gate)
                + lengthlog[:, :, k].clamp_min(-1e4)
                + aligned.clamp_min(1e-30).log()
                - total.clamp_min(1e-30).log()
            )
            possible = matches.any(-1)
            if k == 0:
                # A finite masked operand keeps logaddexp's gradient defined on padding.
                copied = torch.logaddexp(
                    copied.masked_fill(~possible, -1e4),
                    spelled.masked_fill(~allowed, -1e4),
                )
                possible = possible | allowed
            edges.append(copied.masked_fill(~possible, -torch.inf))
        lengths = y.ne(0).sum(1)
        return -SegmentationLikelihood.apply(torch.stack(edges, -1), lengths) / lengths

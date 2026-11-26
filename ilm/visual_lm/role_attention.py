"""Measure attention assigned to grammatical roles within overlapping visual windows."""

import math

import torch
from torch.nn import functional as F


def window_roles(source, roles, stride):
    """Rows give class fractions in each physical window; class zero is unlabelled."""
    if source.shape != roles.shape or stride not in (1, 4):
        raise ValueError("Role and source grids must agree and use a supported stride")
    valid = source.ne(0)
    tags = F.one_hot(roles.long(), 4).float() * valid.unsqueeze(-1)
    counts = F.pad(tags, (0, 0, 0, 3)).unfold(1, 4, stride).sum(-1)
    return counts / counts.sum(-1, keepdim=True).clamp_min(1)


def source_role_mass(attention_module, query, keys, valid, fractions):
    """Average the four physical attention distributions before scoring roles."""
    q = attention_module.heads(attention_module.q(query)).float()
    scores = q @ keys.float().transpose(-1, -2) / math.sqrt(q.size(-1))
    weights = scores.masked_fill(~valid, -torch.inf).softmax(-1).mean(1)
    return weights @ fractions


def compare_assignments(mass, source_tags, target_tags):
    """Exact expectation when permuting the nonzero source tags among tagged cells."""
    counts = F.one_hot(source_tags.long(), 4).sum(1)
    proportions = counts.float() / counts[:, 1:].sum(-1, keepdim=True).clamp_min(1)
    expected = mass[:, :, 1:].sum(-1) * proportions.gather(1, target_tags.long())
    observed = mass.gather(2, target_tags.long().unsqueeze(-1)).squeeze(-1)
    valid = target_tags.ne(0) & counts.gather(1, target_tags.long()).gt(0)
    return observed, expected, valid

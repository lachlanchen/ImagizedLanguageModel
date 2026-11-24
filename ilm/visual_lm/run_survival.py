"""Conditional probabilities for opening, continuing, or closing a physical fact."""

import torch
from torch.nn import functional as F


def supervised(roles, tags, keep):
    opened = roles[:, :, 1] - roles[:, :, 0]
    continued = roles[:, :, 2] - roles[:, :, 0]
    previously_on = F.pad(tags[:, :-1] > 0, (1, 0))
    # Only score the opening decision after a skip or a decision to close.
    opening = F.binary_cross_entropy_with_logits(
        opened, (tags == 1).float(), reduction="none"
    )
    survival = F.binary_cross_entropy_with_logits(
        continued, (tags == 2).float(), reduction="none"
    )
    nll = opening * (tags != 2) + survival * previously_on
    return ((nll * keep).sum(1) / keep.sum(1)).mean()


def utilities(roles, keep):
    """Move each exit cost to the preceding selected cell; the page end is forced."""
    opened = roles[:, :, 1] - roles[:, :, 0]
    continued = roles[:, :, 2] - roles[:, :, 0]
    close = F.logsigmoid(-continued).masked_fill(~keep, 0)
    following = F.pad(close[:, 1:], (0, 1))
    return torch.stack(
        (
            F.logsigmoid(-opened),
            F.logsigmoid(opened) + following,
            F.logsigmoid(continued) - close + following,
        ),
        dim=-1,
    )

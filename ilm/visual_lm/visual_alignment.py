"""Exact likelihood of a pictured target with path-independent edit probabilities."""
import torch
from torch.nn import functional as F


def alignment_log_likelihood(insert, drop, diagonal, stop):
    """Marginalize all insert, drop and diagonal paths to the bottom-right corner.

    A source has n glyphs and a target has m glyphs. The log edge probabilities
    have final dimensions (n+1,m), (n,m+1), and (n,m). Stop is the log
    probability of ending at (n,m). Equal-sized examples may share batch axes.
    Insertion and deletion log probabilities must be finite.
    Distinct copy and replace actions with the same endpoint belong together in
    diagonal via logaddexp. The neural state at (i,j) must be path-independent.
    """
    n, m = diagonal.shape[-2:]
    batch = diagonal.shape[:-2]
    if (insert.shape != (*batch, n+1, m) or drop.shape != (*batch, n, m+1)
            or stop.shape != batch):
        raise ValueError('Incompatible source or target grid dimensions')
    # Prefix sums permit a vectorized scan over every insertion within a row.
    # Accumulate in double precision because long runs subtract similar log scores.
    prefix = F.pad(insert.double().cumsum(-1), (1, 0))
    mass = prefix[..., 0, :]
    for i in range(n):
        deleted = mass + drop[..., i, :]
        advanced = mass[..., :-1] + diagonal[..., i, :]
        entry = torch.cat((deleted[..., :1], torch.logaddexp(deleted[..., 1:], advanced)), -1)
        cost = prefix[..., i+1, :]
        mass = cost + torch.logcumsumexp(entry - cost, -1)
    return mass[..., -1] + stop

"""Exact bitmap correspondences shared by every maximum ordered copy alignment."""

from bisect import bisect_left
from collections import Counter, defaultdict


def chain_lengths(columns):
    tails = []
    lengths = []
    for j in columns:
        k = bisect_left(tails, j)
        if k == len(tails):
            tails.append(j)
        else:
            tails[k] = j
        lengths.append(k + 1)
    return lengths


def forced_matches(source, target):
    """Return longest copy length and pairs mandatory in every such alignment.

    Each row's columns descend, so an increasing column chain uses distinct
    source positions. A pair lies on an optimal chain exactly when its forward
    and reverse ranks sum to the optimum plus one. Every optimal chain visits
    each rank once. A rank with one admissible pair therefore forces that pair.
    """
    locations = defaultdict(list)
    for j, cell in enumerate(target):
        locations[cell].append(j)
    matches = [
        (i, j) for i, cell in enumerate(source) for j in reversed(locations[cell])
    ]
    if not matches:
        return 0, []
    forward = chain_lengths(j for _, j in matches)
    backward = chain_lengths(-j for _, j in reversed(matches))[::-1]
    optimum = max(forward)
    eligible = [
        (rank, pair)
        for pair, rank, tail in zip(matches, forward, backward)
        if rank + tail == optimum + 1
    ]
    alternatives = Counter(rank for rank, _ in eligible)
    return optimum, [pair for rank, pair in eligible if alternatives[rank] == 1]

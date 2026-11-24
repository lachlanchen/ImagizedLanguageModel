"""Locate and collapse candidate spellings by physical glyph equality."""

import numpy as np
from ilm.visual_lm.story_mentions import MAX_SPAN


def symbols(cells):
    _, ids = np.unique(cells, axis=0, return_inverse=True)
    return "".join(chr(int(i) + 1) for i in ids)


def proposals(scores, before):
    # Greedy nonoverlap limits how much context can be consumed by speculative names.
    limit = max(4, min(14, before // 12))
    chosen = []
    taken = np.zeros(before, bool)
    spans = scores[:-1].reshape(-1, MAX_SPAN)[:before].flatten()
    for ix in np.argsort(-spans, kind="stable"):
        a, k = divmod(int(ix), MAX_SPAN)
        b = a + k + 1
        if b <= before and not taken[a:b].any():
            chosen.append((a, b))
            taken[a:b] = True
            if len(chosen) == limit:
                break
    return chosen


def collapse(string, query, candidates):
    """Return source positions, shared-name slots, original intervals and marked bounds."""
    qa, qb = query
    spellings = sorted(
        {string[a:b] for a, b in candidates if 0 <= a < b <= qa},
        key=lambda s: (-len(s), s),
    )
    starts = {}
    for slot, name in enumerate(spellings):
        at = string.find(name)
        while at >= 0:
            end = at + len(name)
            if (end <= qa or at >= qb) and at not in starts:
                starts[at] = (end, slot)
            at = string.find(name, at + 1)
    indices = []
    slots = []
    intervals = []
    new_query = []
    i = 0
    while i < len(string):
        if i in (qa, qb):
            new_query.append(len(indices))
        end, slot = starts.get(i, (i + 1, -1))
        indices.append(i)
        slots.append(slot)
        intervals.append((i, end))
        i = end
    if qb == len(string):
        new_query.append(len(indices))
    if len(new_query) != 2:
        raise ValueError("Unable to locate the visible query")
    return indices, slots, intervals, new_query


def select(logits, slots, intervals, q):
    weights = np.exp(logits.astype(np.float64) - np.max(logits))
    mass = {-1: weights[-1]}
    witness = {-1: None}
    for i, (sid, span) in enumerate(zip(slots[:q], intervals[:q])):
        if sid < 0:
            continue
        mass[sid] = mass.get(sid, 0.0) + weights[i]
        if sid not in witness or weights[i] > weights[witness[sid]]:
            witness[sid] = i
    i = witness[max(mass, key=mass.get)]
    return None if i is None else tuple(intervals[i])

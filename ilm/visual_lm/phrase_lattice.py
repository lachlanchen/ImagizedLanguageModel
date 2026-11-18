"""Exact inference over the phrase composer's monotone image graph."""

from collections import defaultdict
from functools import lru_cache


def graph(model, source):
    return [
        [
            (end, segment, logprob)
            for end in range(i + 1, min(i + 6, len(source)) + 1)
            for segment, logprob in model.options(source[i:end])
        ]
        for i in range(len(source))
    ]


def best_streams(model, edges, k=6):
    """K distinct highest-scoring image streams, using their best segmentation."""
    if k < 1:
        raise ValueError("Choose a positive number of alternatives")
    # Future scores depend only on the source cursor and the last output cell.
    paths = [defaultdict(dict) for _ in range(len(edges) + 1)]
    paths[0][-1][()] = 0.0

    @lru_cache(maxsize=None)
    def join(last, first):
        return model.weight * max(-3.0, min(3.0, model.transition(last, first)))

    for i, options in enumerate(edges):
        for last, candidates in paths[i].items():
            best = sorted(candidates.items(), key=lambda p: (-p[1], p[0]))[:k]
            for end, segment, logprob in options:
                cost = logprob - 0.65
                if model.weight and segment:
                    cost += join(last, segment[0])
                layer = paths[end][segment[-1] if segment else last]
                for prefix, score in best:
                    output = prefix + segment
                    if score + cost > layer.get(output, float("-inf")):
                        layer[output] = score + cost
        paths[i].clear()
    results = [(v, s) for bucket in paths[-1].values() for v, s in bucket.items()]
    return sorted(results, key=lambda p: (-p[1], p[0]))[:k]


def nearest_stream(edges, target):
    """Minimum edit distance over all complete paths, with an actual attaining path.

    For each phrase edge, the incoming row contains the best cost to each
    reference prefix. Apply one Levenshtein update for every emitted cell.
    Columns on arrival can independently choose the incoming edge and column.
    """
    n, m = len(edges), len(target)
    infinity = float("inf")
    distance = [[infinity] * (m + 1) for _ in range(n + 1)]
    parent = [[None] * (m + 1) for _ in range(n + 1)]
    distance[0] = list(range(m + 1))
    for i, options in enumerate(edges):
        for end, segment, _ in options:
            row = distance[i]
            origin = list(range(m + 1))
            for cell in segment:
                nxt = [row[0] + 1]
                starts = [origin[0]]
                for j, expected in enumerate(target, 1):
                    diagonal = row[j - 1] + (cell != expected)
                    drop = row[j] + 1
                    insert = nxt[-1] + 1
                    best = min(diagonal, drop, insert)
                    nxt.append(best)
                    starts.append(
                        origin[j - 1]
                        if best == diagonal
                        else origin[j]
                        if best == drop
                        else starts[-1]
                    )
                row, origin = nxt, starts
            for j, score in enumerate(row):
                if score < distance[end][j]:
                    distance[end][j] = score
                    parent[end][j] = i, origin[j], segment
    path = []
    i, j = n, m
    while i:
        i, j, part = parent[i][j]
        path.append(part)
    return distance[-1][-1], tuple(v for part in reversed(path) for v in part)


def admits(edges, target):
    """Independent exact reachability, without scores or edit-distance arithmetic."""
    columns = [{0}] + [set() for _ in edges]
    for i, options in enumerate(edges):
        for end, part, _ in options:
            for j in columns[i]:
                if target[j : j + len(part)] == part:
                    columns[end].add(j + len(part))
    return len(target) in columns[-1]

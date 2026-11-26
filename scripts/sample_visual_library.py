"""Make a repetition control with both language length distributions exactly matched."""

from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scripts.prepare_visual_library import OUTPUT, RECEIPT, ROOT, read, sha

RUN = ROOT / "artifacts/library_diversity_v1"
EVIDENCE = ROOT / "references/evidence/library_diversity_v1"
MODELS = ROOT / "models/library_diversity"


def library(split):
    p = OUTPUT / (split + ".jsonl")
    if sha(p) != read(RECEIPT)["files"][p.name]:
        raise ValueError(f"Changed library shard {p}")
    return [json.loads(line) for line in p.read_text().splitlines()]


def path_edges(forest, source, dest):
    queue = deque([source])
    parent = {source: None}
    while queue:
        node = queue.popleft()
        if node == dest:
            edges = []
            while node != source:
                node, edge = parent[node]
                edges.append(edge)
            return edges[::-1]
        for other, edge in forest[node].items():
            if other not in parent:
                parent[other] = node, edge
                queue.append(other)
    return None


def cycle_sample(rows, rng, cap=8):
    weights = np.ones(len(rows), dtype=np.int64)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["work"]].append(i)
    # Positive English and negative Chinese lengths are disjoint vertices.
    ends = [(len(r["cells"][0]), -len(r["cells"][1])) for r in rows]
    moves = 0
    for indices in groups.values():
        forest = defaultdict(dict)
        for i in rng.permutation(indices):
            left, right = ends[i]
            path = path_edges(forest, left, right)
            if path is not None:
                plus = [i, *path[1::2]]
                minus = path[::2]
                up = min(cap - weights[plus].max(), weights[minus].min())
                down = min(weights[plus].min(), cap - weights[minus].max())
                delta = up if rng.integers(up + down) < down else -down
                weights[plus] += delta
                weights[minus] -= delta
                for j in path:
                    if weights[j] in (0, cap):
                        a, b = ends[j]
                        del forest[a][b], forest[b][a]
                moves += 1
            if 0 < weights[i] < cap:
                # At least one old forest edge on this path was saturated.
                forest[left][right] = forest[right][left] = i
    if weights.min() < 0 or weights.max() > cap:
        raise ValueError("Invalid reweighted count")
    return weights, moves


def schedule(rows):
    rng = np.random.default_rng(712953)
    weights, moves = cycle_sample(rows, rng)
    replacements = []
    for t in (0, 1):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r["work"], len(r["cells"][t])].append(i)
        narrow = np.empty(len(rows), dtype=np.int64)
        for indices in groups.values():
            repeated = np.repeat(indices, weights[indices])
            if len(repeated) != len(indices):
                raise ValueError("Target-length marginal changed")
            narrow[indices] = rng.permutation(repeated)
        replacements.append(narrow)
    kept = weights[weights > 0]
    p = kept / kept.sum()
    histogram = Counter(map(int, weights))
    return replacements, dict(
        pairs=len(rows), distinct=len(kept), cycles=moves,
        repetition_counts=dict(sorted(histogram.items())),
        entropy_nats=float(-(p * np.log(p)).sum()), full_entropy_nats=float(np.log(len(rows))),
        per_work_length_marginals_matched=True,
        schedule=sha256(np.asarray(replacements, dtype="<i4").tobytes()).hexdigest(),
    )

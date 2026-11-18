"""Measure ambiguity of identical four-cell visual contexts on the fresh roster."""
import json
import math
from collections import Counter, defaultdict

import numpy as np
from safetensors import safe_open

from ilm.visual_lm.local_ink import signatures
from scripts.evaluate_stream_ink import RUN, load, require, roster, sha
from scripts.infer_visual_phrase import MODELS
from scripts.report_visual_phrases import sites_independent
from scripts.train_local_ink import fields


def audit():
    rows, bank = roster('fresh')
    coords = list(sites_independent(rows))
    with safe_open(MODELS / 'editor.safetensors', framework='np') as weights:
        landmarks = weights.get_tensor('landmarks')
    contexts = fields(rows, signatures(np.unpackbits(bank, axis=1)[:, landmarks]), coords)
    sites_by_context = defaultdict(list)
    targets = []
    for n, ((i, _, start, end), context) in enumerate(zip(coords, contexts, strict=True)):
        targets.append(bank[rows[i]['y'][start:end]].tobytes())
        sites_by_context[tuple(context)].append(n)

    ambiguous = []
    ceiling = 0
    entropy_sum = 0.0
    for indices in sites_by_context.values():
        counts = Counter(targets[i] for i in indices)
        if len(counts) > 1:
            ambiguous.extend(indices)
        ceiling += max(counts.values())
        entropy_sum += sum(v * math.log2(len(indices) / v) for v in counts.values())

    split = RUN / 'stream/fresh'
    report = load(split / 'report.json')
    exact = {}
    for name in ('source', 'prefix', 'local'):
        path = split / (name + '.npz')
        require(sha(path) == report['arms'][name]['pixels_sha256'], 'Changed output stream')
        with np.load(path) as archive:
            strips = np.split(archive['pixels'], archive['offsets'][1:-1])
            hit = [bool(stop and strip.tobytes() == target)
                   for strip, stop, target in zip(strips, archive['stopped'], targets, strict=True)]
        require(sum(hit) == report['arms'][name]['exact'], 'Changed phrase score')
        exact[name] = sum(hit[i] for i in ambiguous)

    return dict(sites=len(coords), distinct_local_contexts=len(sites_by_context),
                singleton_contexts=sum(len(v) == 1 for v in sites_by_context.values()),
                ambiguous_sites=len(ambiguous), empirical_conditional_bits=entropy_sum / len(coords),
                local_context_oracle_exact=ceiling, exact_at_ambiguous_sites=exact,
                neural_forward_passes=0)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

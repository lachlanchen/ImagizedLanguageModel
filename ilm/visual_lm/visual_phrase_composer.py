"""Compose observed bitmap phrases in reading order; copy unfamiliar source ink."""

import gzip
import math
from collections import Counter, defaultdict

import msgpack
import numpy as np


class VisualPhraseComposer:
    def __init__(self, cells, phrases, starts, bigrams, totals, weight):
        self.cells = cells
        self.indices = {bytes(cell): i for i, cell in enumerate(cells)}
        self.phrases = phrases
        self.starts = starts
        self.bigrams = bigrams
        self.totals = totals
        self.weight = weight
        self.mass = sum(starts.values())

    @classmethod
    def fit(cls, cells, pairs, align, weight=0):
        # Count only recurrent source spans before retaining their target sequences.
        frequency = Counter()
        for source, _ in pairs:
            for i in range(len(source)):
                for size in range(1, min(6, len(source) - i) + 1):
                    frequency[source[i : i + size]] += 1
        counts = defaultdict(Counter)
        starts, edges, totals = Counter(), Counter(), Counter()
        for source, target in pairs:
            parts = align(list(source), list(target))
            starts.update(target)
            for a, b in zip((-1,) + target, target):
                edges[a, b] += 1
                totals[a] += 1
            for i in range(len(source)):
                value = ()
                for j in range(i, min(len(source), i + 6)):
                    key = source[i : j + 1]
                    value += tuple(parts[j])
                    if frequency[key] >= 2:
                        counts[key][value] += 1
        phrases = {}
        for key, counter in counts.items():
            # Unit prior on faithful copying; shrink rare, one-off paraphrases.
            total = counter.total() + 1
            counter[key] += 1
            choices = [
                (v, math.log(n / total)) for v, n in counter.most_common(4) if n >= 2
            ]
            if choices:
                phrases[key] = choices
        return cls(cells, phrases, starts, edges, totals, weight)

    def transition(self, last, first):
        probability = (self.starts.get(first, 0) + 1) / (self.mass + len(self.starts))
        return math.log(
            (self.bigrams.get((last, first), 0) + 12 * probability)
            / ((self.totals.get(last, 0) + 12) * probability)
        )

    def options(self, key):
        options = self.phrases.get(key, ())
        if len(key) == 1 and not any(value == key for value, _ in options):
            return [*options, (key, -1.6 if options else 0.0)]
        return options

    def decode(self, source):
        paths = [{-1: (0.0, ())}] + [{} for _ in source]
        for i in range(len(source)):
            prior = sorted(paths[i].items(), key=lambda p: -p[1][0])[:4]
            for size in range(1, min(6, len(source) - i) + 1):
                key = source[i : i + size]
                for segment, logprob in self.options(key):
                    for last, (score, emitted) in prior:
                        next_score = score + logprob - 0.65
                        if segment and self.weight:
                            # Only the cross-boundary transition was absent from the phrase itself.
                            next_score += self.weight * max(
                                -3.0, min(3.0, self.transition(last, segment[0]))
                            )
                        end = segment[-1] if segment else last
                        layer = paths[i + size]
                        if end not in layer or next_score > layer[end][0]:
                            layer[end] = (next_score, emitted + segment)
        return max(paths[-1].values(), key=lambda entry: entry[0])[1]

    def generate(self, packed):
        if (
            packed.dtype != np.uint8
            or packed.ndim != 2
            or packed.shape[1] != 128
            or not len(packed)
        ):
            raise ValueError("Expected at least one packed 32-by-32 source image")
        source = tuple(self.indices.get(bytes(v), -2 - i) for i, v in enumerate(packed))
        result = self.decode(source)
        return np.array(
            [self.cells[i] if i >= 0 else packed[-2 - i] for i in result],
            dtype=np.uint8,
        ).reshape(-1, 128)

    def save(self, path):
        record = dict(
            version=1,
            cells=self.cells.tobytes(),
            weight=self.weight,
            phrases=list(self.phrases.items()),
            starts=list(self.starts.items()),
            bigrams=list(self.bigrams.items()),
            totals=list(self.totals.items()),
        )
        with gzip.open(path, "xb") as f:
            msgpack.pack(record, f)

    @classmethod
    def restore(cls, path):
        with gzip.open(path, "rb") as f:
            record = msgpack.unpack(f, raw=False, use_list=False)
        if record["version"] != 1:
            raise ValueError("Unrecognized visual composer")
        cells = np.frombuffer(record["cells"], np.uint8).reshape(-1, 128)
        return cls(
            cells,
            dict(record["phrases"]),
            dict(record["starts"]),
            dict(record["bigrams"]),
            dict(record["totals"]),
            record["weight"],
        )

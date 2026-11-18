"""Source-span and target-image realization conditioned on a learned visual sense."""

from collections import Counter, defaultdict
import gzip
import math

import msgpack
import numpy as np
from rapidfuzz.distance import Levenshtein


def sides(sequence, index):
    left = bytes(sequence[index - 1]) if index else None
    right = bytes(sequence[index + 1]) if index + 1 < len(sequence) else None
    return left, right


class VisualPhraseBoundary:
    def __init__(self, senses):
        self.senses = senses
        self.tables = defaultdict(Counter)

    @staticmethod
    def contexts(left, right):
        # The three levels remain distinct when a sentence starts or ends.
        return [(0, b"", b""), (1, left, b""), (2, b"", right), (3, left, right)]

    def observe(self, source, index, sense, phrase, target, target_slice, candidates):
        """Marginalize plausible source boundaries for this observed output phrase."""
        left, right = sides(source, index)
        a, b = target_slice
        prefix = tuple(map(bytes, target[:a]))
        suffix = tuple(map(bytes, target[b:]))
        groups = self.contexts(left, right)
        # Cache the two possible prefixes and suffixes within this occurrence.
        costs = {}
        def distance(segment, start, end, reference):
            key = segment, start, end
            if key not in costs:
                output = self.senses.composer.generate(source[start:end]) if start < end else []
                costs[key] = Levenshtein.distance(tuple(map(bytes, output)), reference)
            return costs[key]
        masses = Counter()
        for option in candidates:
            l, r = option["left"], option["right"]
            cost = distance("prefix", 0, index - l, prefix)
            cost += distance("suffix", index + 1 + r, len(source), suffix)
            # Record the identities of consumed cells. A learned deletion applies
            # only if its adjacent source image is physically present at inference.
            masses[(left if l else None, right if r else None, phrase)] = (
                -cost - 0.5 * option["outer_edit_cost"] - 0.7 * (l + r)
            )
        best = max(masses.values())
        weights = {k: math.exp(v - best) for k, v in masses.items()}
        total = sum(weights.values())
        for context in groups:
            table = self.tables[(bytes(source[index]), sense, *context)]
            for action, weight in weights.items():
                table[action] += weight / total

    def action(self, cell, sense, left, right, allow_left, allow_right):
        score = Counter()
        for context, weight in zip(self.contexts(left, right), (1, 16, 16, 64)):
            for (l, r, phrase), mass in self.tables.get((cell, sense, *context), {}).items():
                if l is not None and (not allow_left or l != left):
                    continue
                if r is not None and (not allow_right or r != right):
                    continue
                score[(l is not None, r is not None, phrase)] += weight * mass
        return max(score, key=score.get) if score else (False, False, sense)

    def generate(self, packed):
        centres = [i for i, v in enumerate(packed) if bytes(v) in self.senses.heads]
        if not centres:
            return self.senses.composer.generate(packed)
        parts = []
        begin = 0
        for index, sense in zip(centres, self.senses.choices(packed, centres)):
            if sense is None:
                continue
            left, right = sides(packed, index)
            l, r, phrase = self.action(bytes(packed[index]), sense, left, right,
                begin < index and left not in self.senses.heads, right not in self.senses.heads)
            end = index - int(l)
            if end > begin:
                parts.append(self.senses.composer.generate(packed[begin:end]))
            parts.append(np.frombuffer(phrase, np.uint8).reshape(-1, 128))
            begin = index + 1 + int(r)
        if begin < len(packed):
            parts.append(self.senses.composer.generate(packed[begin:]))
        return np.concatenate(parts)

    def save(self, path):
        with gzip.open(path, "xb") as f:
            msgpack.pack((1, [(k, list(v.items())) for k, v in self.tables.items()]), f)

    @classmethod
    def restore(cls, path, senses):
        with gzip.open(path, "rb") as f:
            version, rows = msgpack.unpack(f, raw=False, use_list=False)
        if version != 1:
            raise ValueError("Unknown source-boundary format")
        model = cls(senses)
        model.tables.update({k: Counter(dict(v)) for k, v in rows})
        return model

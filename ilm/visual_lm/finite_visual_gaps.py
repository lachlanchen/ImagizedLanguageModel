"""Empirical image sequences conditioned on the source gap and adjacent preserved ink."""

from collections import Counter, defaultdict
import gzip
import msgpack
import numpy as np


class FiniteVisualGaps:
    def __init__(self, cells, rules):
        self.cells = cells
        self.indices = {bytes(v): i for i, v in enumerate(cells)}
        self.rules = rules

    @staticmethod
    def contexts(span, left, right):
        return (
            (left, span, right),
            (left, span, -2),
            (-2, span, right),
            (-2, span, -2),
        )

    @classmethod
    def fit(cls, archive):
        with np.load(archive) as z:
            cells, pairs, offsets, x, y, xs, ys = (
                z[k]
                for k in (
                    "cells",
                    "pairs",
                    "offsets",
                    "source",
                    "target",
                    "x_offsets",
                    "y_offsets",
                )
            )
        counts = defaultdict(Counter)
        for row, (a, b) in enumerate(zip(xs[:-1], ys[:-1])):
            src = x[a : xs[row + 1]].tolist()
            tgt = y[b : ys[row + 1]].tolist()
            left = previous = -1
            for right, j in [
                *pairs[offsets[row] : offsets[row + 1]].tolist(),
                (len(src), len(tgt)),
            ]:
                if right - left - 1 <= 6:
                    span = tuple(src[left + 1 : right])
                    output = (
                        tuple(tgt[previous + 1 : j]) if j - previous - 1 <= 12 else None
                    )
                    for key in cls.contexts(
                        span,
                        src[left] if left >= 0 else -1,
                        src[right] if right < len(src) else -1,
                    ):
                        counts[key][output] += 1
                left, previous = right, j
        rules = {}
        for key, counter in counts.items():
            value, count = counter.most_common(1)[0]
            total = counter.total()
            if value is not None and count >= 2 and 2 * count > total:
                rules[key] = (value, count, total)
        return cls(cells, rules)

    def generate(self, pictures, keep):
        if (
            pictures.dtype != np.uint8
            or pictures.shape != (len(keep), 128)
            or np.asarray(keep).dtype != bool
        ):
            raise ValueError(
                "Expected packed source glyphs and one Boolean decision per glyph"
            )
        src = tuple(self.indices.get(bytes(v), -3) for v in pictures)
        left = -1
        segments = []
        for right in [*np.flatnonzero(keep), len(src)]:
            interval = pictures[left + 1 : right]
            if len(interval) <= 6:
                keys = self.contexts(
                    src[left + 1 : right],
                    src[left] if left >= 0 else -1,
                    src[right] if right < len(src) else -1,
                )
                for tier in (keys[:1], keys[1:3], keys[3:]):
                    matches = [self.rules[key] for key in tier if key in self.rules]
                    if matches:
                        output, _, _ = max(matches, key=lambda v: (v[1] / v[2], v[1]))
                        interval = self.cells[list(output)]
                        break
            segments.append(interval)
            if right < len(src):
                segments.append(pictures[right : right + 1])
            left = right
        return np.concatenate(segments)

    def save(self, path):
        with gzip.open(path, "xb") as f:
            msgpack.pack((1, self.cells.tobytes(), tuple(self.rules.items())), f)

    @classmethod
    def restore(cls, path):
        with gzip.open(path, "rb") as f:
            version, bank, pairs = msgpack.unpack(f, raw=False, use_list=False)
        if version != 1:
            raise ValueError("Unsupported finite gap model")
        return cls(np.frombuffer(bank, np.uint8).reshape(-1, 128), dict(pairs))

"""Fit contextual participant probabilities and bind unfamiliar ink reversibly."""

from collections import Counter, defaultdict
import gzip

import msgpack
import numpy as np
import torch


class InkPermutation:
    """Disjoint physical transpositions are their own inverse."""

    def __init__(self, pairs, device):
        pairs = list(pairs)
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Each physical transposition requires exactly two glyphs")
        flat = [cell for pair in pairs for cell in pair]
        if len(flat) != len(set(flat)) or any(len(cell) != 128 for cell in flat):
            raise ValueError("Expected disjoint pairs of packed 32-by-32 images")
        a = np.frombuffer(b"".join(flat), np.uint8).reshape(-1, 128)
        self.cells = torch.as_tensor(np.unpackbits(a, axis=1), device=device).bool()

    def __call__(self, image):
        if not len(self.cells):
            return image
        # Compare to the original input for every pair, including reverse matches.
        out = image.clone()
        for j, cell in enumerate(self.cells):
            mask = (image == cell).all(-1)
            out[mask] = self.cells[j ^ 1]
        return out


class ContextualInkBinding:
    OFFSETS = ((-2, -1, 1, 2), (-1, 1), (-2, -1), (1, 2), (-1,), (1,))
    START = -1
    END = -2
    UNKNOWN = -3
    OTHER = -4

    def __init__(self, bank, counts, prior):
        self.bank = tuple(bank)
        self.indices = {c: i for i, c in enumerate(bank)}
        self.counts = counts
        self.prior = prior

    @classmethod
    def contexts(cls, ids, i):
        def cell(j):
            if j < 0:
                return cls.START
            if j >= len(ids):
                return cls.END
            return ids[j]

        for idx, offsets in enumerate(cls.OFFSETS):
            values = tuple(cell(i + d) for d in offsets)
            if cls.UNKNOWN not in values:
                yield (idx, *values)

    @classmethod
    def fit(cls, known, aligned):
        # Each training pair supplies the original pixels and the two positions
        # changed by participant exchange, without textual identities.
        vocab = set(known)
        for seq, _ in aligned:
            vocab.update(map(bytes, seq))
        bank = sorted(vocab)
        index = {c: i for i, c in enumerate(bank)}
        counts = defaultdict(Counter)
        prior = Counter()
        for seq, locations in aligned:
            ids = [index[bytes(c)] for c in seq]
            if len(locations) != 2 or len(set(locations)) != 2:
                raise ValueError("Expected two participant positions")
            for i, c in enumerate(ids):
                label = c if i in locations else cls.OTHER
                prior[label] += 1
                for key in cls.contexts(ids, i):
                    counts[key][label] += 1
        return cls(bank, dict(counts), prior)

    def choose(self, packed, *, frequency_only=False):
        values = tuple(map(bytes, packed))
        ids = [self.indices.get(c, self.UNKNOWN) for c in values]
        reserved = set(ids)
        candidates = defaultdict(Counter)
        for i, c in enumerate(ids):
            if c != self.UNKNOWN:
                continue
            matches = [
                key
                for key in self.contexts(ids, i)
                if key in self.counts and self.counts[key].total() >= 2
            ]
            if not matches:
                continue
            # Use the longest observed neighborhoods. No held-out threshold tuning.
            longest = max(map(len, matches))
            tally = Counter()
            for key in matches:
                if len(key) == longest:
                    tally.update(self.counts[key])
            candidates[values[i]].update(tally)
        chosen = []
        for unknown, tally in candidates.items():
            positive = tally.total() - tally[self.OTHER]
            if positive <= tally[self.OTHER]:
                continue
            choices = self.prior if frequency_only else tally
            eligible = [n for n in choices if n >= 0 and n not in reserved]
            if not eligible:
                eligible = [n for n in self.prior if n >= 0 and n not in reserved]
            if not eligible:
                continue
            alias = max(eligible, key=lambda n: (choices[n], self.prior[n], -n))
            chosen.append((unknown, self.bank[alias]))
            reserved.add(alias)
        return chosen

    def save(self, path):
        with gzip.open(path, "xb") as f:
            msgpack.pack(
                (
                    1,
                    self.bank,
                    list((k, list(v.items())) for k, v in self.counts.items()),
                    list(self.prior.items()),
                ),
                f,
                use_bin_type=True,
            )

    @classmethod
    def restore(cls, path):
        with gzip.open(path, "rb") as f:
            version, bank, counts, prior = msgpack.unpack(f, raw=False, use_list=False)
        if version != 1:
            raise ValueError("Unknown contextual binding format")
        return cls(bank, {k: Counter(dict(v)) for k, v in counts}, Counter(dict(prior)))

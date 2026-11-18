"""Supervised answer boundaries in physically aligned 32-by-32 glyph streams."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def packed_tiles(cells):
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != 128 or cells.dtype != np.uint8 or not len(cells):
        raise ValueError("Expected a nonempty stream of packed 32 by 32 binary images")
    return np.ascontiguousarray(cells)


def pooled_ink(cells):
    pixels = np.unpackbits(cells, axis=1).reshape(-1, 4, 8, 4, 8)
    return pixels.mean(axis=(2, 4), dtype=np.float32).reshape(-1, 16)


def position_features(passage, question):
    passage, question = map(packed_tiles, (passage, question))
    n, m = len(passage), len(question)
    # Integer handles exist only inside this pair and encode exact bitmap equality.
    _, local = np.unique(np.concatenate((passage, question)).view('V128').ravel(),
                         return_inverse=True)
    matches = local[:n, None] == local[None, n:]
    active = matches.copy()
    channels = []
    for length in range(1, min(m, n, 4) + 1):
        if length > 1:
            active = active[:-1, :-1] & matches[length-1:, length-1:]
        coverage = active.any(axis=1)
        channels.append(np.pad(coverage, (0, length-1)))
        channels.append(np.pad(coverage, (length-1, 0)))
    channels.extend(np.zeros(n) for _ in range(8 - len(channels)))
    overlap = np.column_stack(channels)
    idx = np.arange(n)
    # Distinguish overlapping text on the left and right of each candidate.
    prefix = np.vstack((np.zeros((1, 8)), overlap.cumsum(0)))
    features = [overlap, (idx / max(n-1, 1))[:, None],
                matches.sum(1)[:, None], matches.argmax(1)[:, None] / m,
                np.full((n, 1), np.log1p(m))]
    for width in (2, 8, 24, 64):
        features.append((prefix[idx] - prefix[np.maximum(idx-width, 0)]) / width)
        features.append((prefix[np.minimum(idx+width+1, n)] - prefix[idx+1]) / width)
    pooled = pooled_ink(passage)
    padded = np.vstack((np.zeros((1, 16)), pooled, np.zeros((1, 16))))
    features.extend(padded[idx+offset] for offset in (0, 1, 2))
    qpool = pooled_ink(question)
    for summary in (qpool.mean(0), qpool[0], qpool[max(0, m-2)]):
        features.append(np.broadcast_to(summary, (n, 16)))
    return np.asarray(np.column_stack(features), dtype=np.float32)


@dataclass
class VisualExtractiveReader:
    start: HistGradientBoostingClassifier
    end: HistGradientBoostingClassifier
    lengths: np.ndarray

    @classmethod
    def fit(cls, features, starts, ends, lengths):
        options = dict(max_iter=64, max_leaf_nodes=15, learning_rate=.1,
                       l2_regularization=1., max_bins=64, early_stopping=False, random_state=202611210)
        return cls(HistGradientBoostingClassifier(**options).fit(features, starts),
                   HistGradientBoostingClassifier(**options).fit(features, ends),
                   np.bincount(lengths))

    def bounds(self, passage, question):
        features = position_features(passage, question)
        start = self.start.decision_function(features)
        end = self.end.decision_function(features)
        prior = np.zeros(len(passage)+1)
        limit = min(len(prior), len(self.lengths))
        prior[:limit] = np.log1p(self.lengths[:limit])
        # Scan over lengths in linear working memory. Keep the complete range.
        best_score, best = -np.inf, (0, 1)
        for length in range(1, len(passage)+1):
            size = len(passage)-length+1
            scores = start[:size] + end[length-1:] + prior[length]
            i = int(scores.argmax())
            if scores[i] > best_score:
                best_score, best = float(scores[i]), (i, i+length)
        return best

    def read(self, passage, question):
        a, b = self.bounds(passage, question)
        return passage[a:b].copy()

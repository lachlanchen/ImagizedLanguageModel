"""Selective fixed-font content verification, ONLY for training/evaluation.

This finite clean-glyph inventory is a recognizer. It is forbidden in the
student's answer-generation path and is not a general semantic model. A unique
decoding ball is a geometric statement under bounded bit corruption, not a
proof that arbitrary images in that ball are human-readable language.
"""
import numpy as np


def packed_images(ink):
    value = np.asarray(ink)
    if value.ndim != 4 or value.shape[1:] != (1, 32, 32):
        raise ValueError('Complete [N,1,32,32] image cells required')
    if not np.all((value == 0) | (value == 1)):
        raise ValueError('Binary physical ink required')
    return np.packbits(value.reshape(len(value), 1024).astype(np.uint8), axis=1, bitorder='big')


def ordered_edit_distance(actual, expected):
    """Every position is retained, including duplicates, spaces and abstentions."""
    previous = list(range(len(expected) + 1))
    for i, a in enumerate(actual):
        current = [i + 1]
        for j, b in enumerate(expected):
            current.append(min(previous[j + 1] + 1, current[-1] + 1,
                               previous[j] + int(a != b)))
        previous = current
    return previous[-1]


class SelectiveGlyphVerifier:
    """Nearest clean image inside its disjoint, maximal unique decoding ball.

    Prototype IDs denote physical bitmaps, not Unicode characters. The finite
    image bank is nevertheless a training-only catalogue and must be disclosed.
    No question, reference answer, frequency prior or language context enters
    decode(). In particular, references cannot bias a damaged cell's identity.
    """
    def __init__(self, prototypes, *, guard=lambda: None):
        value = np.asarray(prototypes)
        if value.ndim != 2 or value.shape[1] != 128 or value.dtype != np.uint8 or len(value) < 2:
            raise ValueError('At least two packed uint8 physical prototypes required')
        if len(np.unique(value, axis=0)) != len(value):
            raise ValueError('Deduplicate equal physical images before construction')
        self.prototypes = value.copy()
        self.bits = np.unpackbits(value, axis=1, bitorder='big').astype(np.float32)
        self.mass = self.bits.sum(1)
        self.identities = {row.tobytes(): i for i, row in enumerate(value)}
        self.minimum_distance = np.empty(len(value), dtype=np.int64)
        for first in range(0, len(value), 128):
            guard()
            last = min(first + 128, len(value))
            d = self.mass[first:last, None] + self.mass[None] - 2 * (self.bits[first:last] @ self.bits.T)
            # The binary dot products and their sums are exact float32 integers.
            if not np.all(d == np.rint(d)) or np.any(d < 0):
                raise ValueError('Physical Hamming distances lost integer exactness')
            d[np.arange(last - first), np.arange(first, last)] = 2048
            self.minimum_distance[first:last] = d.min(1).astype(np.int64)
        if np.any(self.minimum_distance < 1):
            raise ValueError('Physical image identities must be distinct')

    def decode(self, ink, *, guard=lambda: None):
        packed = packed_images(ink)
        bits = np.unpackbits(packed, axis=1, bitorder='big').astype(np.float32)
        identities, distances = [], []
        for first in range(0, len(bits), 128):
            guard()
            block = bits[first:first + 128]
            d = block.sum(1)[:, None] + self.mass[None] - 2 * (block @ self.bits.T)
            chosen = d.argmin(1)
            identities.extend(chosen.tolist())
            distances.extend(d[np.arange(len(block)), chosen].astype(np.int64).tolist())
        identities = np.asarray(identities, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.int64)
        accepted = 2 * distances < self.minimum_distance[identities]
        return dict(identities=np.where(accepted, identities, -1),
                    accepted=accepted, distances=distances,
                    nearest_prototype_ids=identities)

    def reference_ids(self, reference):
        # Exact clean membership only. Never add held-out reference prototypes.
        return np.asarray([self.identities.get(row.tobytes(), -1)
                           for row in packed_images(reference)], dtype=np.int64)

    def score(self, ink, reference, *, stopped, guard=lambda: None):
        packed_images(ink); packed_images(reference)
        if type(stopped) is not bool or not len(reference) or not np.any(reference):
            raise ValueError('A visible complete reference and explicit stop event required')
        expected = self.reference_ids(reference)
        covered = bool(np.all(expected >= 0))
        if not covered:
            return dict(reward=0., reference_covered=False,
                        missing_reference_cells=int(np.sum(expected < 0)),
                        complete_content=False, accepted_cells=0, abstained_cells=len(ink))
        if not stopped or not len(ink) or not np.any(ink):
            return dict(reward=0., reference_covered=True, missing_reference_cells=0,
                        complete_content=False, accepted_cells=0, abstained_cells=len(ink))
        decoded = self.decode(ink, guard=guard)
        actual = decoded['identities']
        complete = bool(len(actual) == len(expected) and np.array_equal(actual, expected))
        distance = ordered_edit_distance(actual.tolist(), expected.tolist())
        dense = max(0., 1 - distance / max(len(actual), len(expected)))
        return dict(reward=.5 * (dense + float(complete)), reference_covered=True,
                    missing_reference_cells=0, complete_content=complete,
                    accepted_cells=int(decoded['accepted'].sum()),
                    abstained_cells=int((~decoded['accepted']).sum()))

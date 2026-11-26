"""Correct small bitmap perturbations inside disjoint Hamming balls."""

from functools import lru_cache

import numpy as np

_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    1, dtype=np.uint8
)


class InkTransport:
    def __init__(self, bitmaps):
        bitmaps = np.asarray(bitmaps)
        if bitmaps.dtype != np.uint8 or bitmaps.ndim != 2 or bitmaps.shape[1] != 128:
            raise ValueError("Expected packed 32 by 32 bitmaps")
        self.bank = bitmaps.copy()
        self.ids = {bytes(cell): i for i, cell in enumerate(self.bank)}
        if len(self.ids) != len(self.bank) or len(self.bank) < 2:
            raise ValueError("At least two distinct reference shapes are required")
        self.resolve = lru_cache(maxsize=8192)(self._resolve)
        self.separation = lru_cache(maxsize=len(self.bank))(self._separation)

    def distances(self, cell):
        return _POPCOUNT[self.bank ^ cell].sum(axis=1, dtype=np.uint16)

    def _separation(self, i):
        distances = self.distances(self.bank[i])
        distances[i] = 1025
        return int(distances.min())

    def _resolve(self, packed):
        if len(packed) != 128:
            raise ValueError("Expected a packed 32 by 32 bitmap")
        if packed in self.ids:
            return packed
        distances = self.distances(np.frombuffer(packed, np.uint8))
        i = int(distances.argmin())
        if 2 * int(distances[i]) < self.separation(i):
            return bytes(self.bank[i])
        return packed

    def apply(self, cells):
        return np.frombuffer(
            b"".join(self.resolve(bytes(cell)) for cell in cells), np.uint8
        ).reshape(-1, 128)

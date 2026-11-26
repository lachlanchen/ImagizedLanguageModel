"""Measure entropy of actual pictured pairs and check physical overlap across splits."""

from collections import Counter
import numpy as np
from scripts.prepare_book_breadth import BASE, WORK, PUB, ROOT, read, dump, sha
from scripts.train_book_breadth import matched_pool


def measure():
    corpus = read(PUB / "corpus.json")
    for name, h in corpus["files"].items():
        if sha(WORK / name) != h:
            raise ValueError(name)
    cells = np.load(WORK / "cells.npy", mmap_mode="r")
    domains = np.load(WORK / "domains.npy")
    lengths = np.count_nonzero(cells, axis=2)
    mapping = matched_pool(lengths, domains)
    palette = np.load(BASE / "palette.npy")
    assert len(set(map(bytes, palette[2:]))) == len(palette) - 2
    # Identifiers >=2 are a one-to-one naming of physical bitmap contents.
    assert np.all((cells == 0) | (cells >= 2))
    assert np.array_equal(cells != 0, np.arange(96) < lengths[:, :, None])
    report = {}
    for tag, indices in (("broad", np.arange(len(cells))), ("narrow", mapping)):
        whole = (
            np.ascontiguousarray(cells[indices]).reshape(len(cells), -1)
            .view(np.dtype((np.void, 384)))
            .reshape(-1)
        )
        _, c = np.unique(whole, return_counts=True)
        p = c / c.sum()
        report[tag] = dict(
            distinct_physical_pairs=len(c),
            entropy_bits=float(-p @ np.log2(p)),
            positions=int(lengths[indices].sum() + 2 * len(indices)),
        )
    split_arrays = {s: read(WORK / (s + ".json")) for s in ("dev", "test")}
    overlap = {}
    for t in (0, 1):
        observed = set()
        lookup = {tuple(r["cells"][t]) for rows in split_arrays.values() for r in rows}
        for part in np.array_split(cells[:, t], 128):
            for side in part:
                k = tuple(side[: np.count_nonzero(side)].tolist())
                if k in lookup:
                    observed.add(k)
        for s, rows in split_arrays.items():
            overlap[s + str(t)] = sum(tuple(r["cells"][t]) in observed for r in rows)
    counts = Counter(int(c) for c in np.bincount(mapping) if c)
    total = sum(n * c for c, n in counts.items())
    assert total == len(cells)
    lower = sum(n * c * (c.bit_length() - 1) for c, n in counts.items())
    entropy = report["broad"]["entropy_bits"] - report["narrow"]["entropy_bits"]
    if report["broad"]["distinct_physical_pairs"] == len(cells):
        assert entropy + 1e-10 >= lower / total
    result = dict(
        pairs=report,
        bit_difference=entropy,
        lower_bits_per_presentation=lower / total,
        lower_bits_per_pass=lower,
        selection_overlap_by_side=overlap,
        formal=sha(ROOT / "formal/span_list/BookExposure.lean"),
        fibres=counts,
    )
    dump(PUB / "exposure.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    measure()

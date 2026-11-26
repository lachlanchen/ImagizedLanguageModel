"""Replace training pairs by disjoint active/passive events of identical physical lengths."""

from collections import Counter, defaultdict
import numpy as np
from scripts.prepare_book_breadth import (
    BASE,
    ROOT,
    WORK as BOOK,
    PUB as BOOK_PUB,
    Ink,
    read,
    dump,
    sha,
)
from scripts.prepare_translation_paraphrases import (
    alternatives,
    WORK as SOURCE,
    PUB as SOURCE_PUB,
)

WORK = ROOT / "artifacts/book_rehearsal_v1"
PUB = ROOT / "references/evidence/book_rehearsal_v1"


def padded(pair):
    a = np.zeros((2, 96), dtype=np.uint16)
    for t, side in enumerate(pair):
        a[t, : len(side)] = side
    return a


def main():
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    corp = read(BOOK_PUB / "corpus.json")
    for k, h in corp["files"].items():
        if sha(BOOK / k) != h:
            raise ValueError(k)
    if (
        sha(SOURCE / "examples.json")
        != read(SOURCE_PUB / "corpus.json")["files"]["examples"]
    ):
        raise ValueError("Historical rehearsal pool")
    cells = np.load(BOOK / "cells.npy", mmap_mode="r")
    lens = np.count_nonzero(cells, axis=2)
    slots = defaultdict(list)
    for i, p in enumerate(lens.tolist()):
        slots[tuple(p)].append(i)
    old_pairs = {row.tobytes() for row in cells}
    originals = read(SOURCE / "examples.json")
    history = {
        (tuple(src), tuple(r["target"][1]))
        for r in originals
        for src in (r["target"][0], *r["variants"][0])
    }
    ink = Ink(np.load(BASE / "palette.npy"))
    forbidden = [set(), set()]
    graded = read(BOOK / "generated.json")
    for r in graded["rows"]:
        for t, side in ((r["t"], r["target"]), (1 - r["t"], r["input"])):
            forbidden[t].add(tuple(side))
    for r in read(BOOK / "dev.json"):
        for t, side in enumerate(r["cells"]):
            forbidden[t].add(tuple(side))
    candidates = defaultdict(dict)
    for kind, canon, variants in alternatives():
        if kind != "event":
            continue
        target = tuple(ink.encode(canon[1]))
        for voice, text in (("active", canon[0]), ("passive", variants[0][0])):
            src = tuple(ink.encode(text))
            pair = src, target
            if pair not in history or src in forbidden[0] or target in forbidden[1]:
                continue
            if padded(pair).tobytes() not in old_pairs:
                candidates[len(src), len(target)][pair] = voice
    rng = np.random.default_rng(519443)
    indices = []
    replacements = []
    counts = Counter()
    for shape, options in sorted(candidates.items()):
        eligible = slots[shape]
        take = min(len(eligible) // 4, len(options))
        pairs = list(options)
        for j, k in zip(
            rng.choice(eligible, take, replace=False),
            rng.choice(len(pairs), take, replace=False),
            strict=True,
        ):
            p = pairs[k]
            indices.append(j)
            replacements.append(padded(p))
            counts[options[p]] += 1
    idx = np.asarray(indices, np.int32)
    data = np.stack(replacements)
    if not np.array_equal(lens[idx], np.count_nonzero(data, axis=2)) or len(
        np.unique(idx)
    ) != len(idx):
        raise ValueError("Geometry")
    np.savez(WORK / "replacement.npz", indices=idx, cells=data)
    rc = dict(
        replaced_pairs=len(idx),
        types=counts,
        rehearsed_positions_per_pass=int(lens[idx].sum() + 2 * len(idx)),
        training_pairs=len(cells),
        total_positions_per_pass=int(lens.sum() + 2 * len(cells)),
        pool=sha(SOURCE / "examples.json"),
        control=sha(BOOK_PUB / "corpus.json"),
        replacement=sha(WORK / "replacement.npz"),
        excluded_complete_held_sides=[len(k) for k in forbidden],
    )
    dump(PUB / "design.json", rc)
    print(rc, flush=True)


if __name__ == "__main__":
    main()

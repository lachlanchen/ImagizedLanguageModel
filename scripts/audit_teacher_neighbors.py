"""Test whether fresh replies can be reproduced by copying a nearest corpus translation."""

from collections import Counter
import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scripts.prepare_teacher_translation import (
    ROOT,
    BASE,
    BOOK,
    PREVIOUS,
    OLD,
    WORK,
    PUB,
    read,
    dump,
    sha,
)
from scripts.prepare_translation_paraphrases import FIRST, LARGE


def word(cells):
    # Inventory handles are unique physical bitmaps. Dense strings accelerate edit distance.
    return np.asarray(cells, dtype="<u2").tobytes().decode("utf-16-le").rstrip("\0")


def pool(t):
    seen = {}

    def add(x, y, group):
        k = word(x), word(y)
        if k not in seen:
            seen[k] = group

    banks = [
        (np.load(BOOK / "cells.npy", mmap_mode="r"), "book-talk"),
        (np.load(PREVIOUS / "replacement.npz")["cells"], "role-rehearsal"),
        (np.load(LARGE / "images.npy", mmap_mode="r"), "previous-parallel"),
    ]
    for rows, group in banks:
        for row in rows:
            add(row[1 - t], row[t], group)
    for split in ("train", "news-train"):
        for r in read(OLD / (split + ".json")):
            pair = r["cells"]
            add(pair[1 - t], pair[t], split)
    for r in read(ROOT / "artifacts/translation_paraphrases_v1/examples.json"):
        for v in [r["target"][1 - t], *r["variants"][1 - t]]:
            add(v, r["target"][t], "paraphrases")
    for r in read(FIRST / "contrasts.json"):
        for pair in r["cells"]:
            add(pair[1 - t], pair[t], "contrast")
    for r in read(WORK / "selected.json"):
        if r["t"] == t:
            add(r["input"], r["teacher_cells"], "current-teacher")
    return seen


def main():
    gen = read(WORK / "generated.json")
    records = {}
    sizes = {}
    credited = {
        r["i"]
        for r in read(PUB / "fresh-source-review.json")["records"]
        if any(r[m] for m in ("parent", "distilled", "corpus"))
    }
    for t in (0, 1):
        choices = pool(t)
        pairs = list(choices)
        src = [x[0] for x in pairs]
        targets = {y for x, y in pairs}
        sizes[t] = len(pairs)
        for i, (r, out) in enumerate(
            zip(gen["rows"][:96], gen["answers"][:96], strict=True)
        ):
            if r["t"] != t:
                continue
            x = word(r["input"])
            _, s, j = process.extractOne(
                x, src, scorer=Levenshtein.normalized_similarity
            )
            p = pairs[j]
            row = dict(
                source_similarity=s,
                nearest_pool=choices[p],
                nearest_target_is_reference=p[1] == word(r["target"]),
            )
            for m in ("parent", "corpus", "distilled"):
                emitted = word(out[m]["cells"])
                row[m] = dict(
                    matches_nearest=emitted == p[1], exists_in_pool=emitted in targets
                )
                if i in credited and emitted in targets:
                    ix = [j for j, v in enumerate(pairs) if v[1] == emitted]
                    row[m]["closest_source_with_same_reply"] = max(
                        Levenshtein.normalized_similarity(x, src[j]) for j in ix
                    )
            records[i] = row
    dump(
        PUB / "neighbors.json",
        dict(
            pool_sizes=sizes,
            actual_requests=96,
            examples=records,
            emitter=sha(WORK / "generated.json"),
            exact_source_repeats=sum(
                r["source_similarity"] == 1 for r in records.values()
            ),
            nearest_reference_exact=sum(
                r["nearest_target_is_reference"] for r in records.values()
            ),
            scope="Eligible physical pairs across identified training pools; includes some candidates not sampled by the historical stochastic fits. One nearest neighbor, first tie by pool order.",
        ),
    )


if __name__ == "__main__":
    main()

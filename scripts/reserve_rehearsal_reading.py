"""Reserve new whole literary works and wiki article URLs for blind image translation."""

from collections import defaultdict
from functools import lru_cache
import numpy as np
from scripts.prepare_book_rehearsal import (
    BASE,
    ROOT,
    BOOK,
    WORK,
    PUB,
    Ink,
    read,
    dump,
    sha,
)
from scripts.prepare_book_breadth import flores, paired_windows, rank
from scripts.prepare_paired_ink import FIRST, OLD, key, opus_source, published


def reserve():
    works = ("great-expectations", "gullivers-travels", "robinson-crusoe", "walden")
    prior_docs = {
        r["doc"] for split in ("dev", "test") for r in read(BOOK / (split + ".json"))
    }
    ink = Ink(np.load(BASE / "palette.npy"))
    inventories = [set(x) for x in read(BASE / "inventories.json")]
    normalize = lru_cache(65536)(key)
    candidates = defaultdict(list)

    def add(group, doc, pair):
        if doc in prior_docs or not (
            4 <= len(pair[0]) <= 96 and 2 <= len(pair[1]) <= 96
        ):
            return
        if not any(s.isascii() and s.isalpha() for s in pair[0]) or not any(
            "\u3400" <= s <= "\u9fff" for s in pair[1]
        ):
            return
        cells = [ink.encode(s) for s in pair]
        if any(
            ink.bad.intersection(s) or not set(x) <= v
            for s, x, v in zip(pair, cells, inventories, strict=True)
        ):
            return
        candidates[group, doc].append((pair, cells))

    for work in works:
        files = sorted(
            (ROOT.parent / "ZhJpBook/books" / work).glob(
                "work/trilingual/interlinear/chunks/*.json"
            )
        )
        if not files:
            raise FileNotFoundError(work)
        for f in files:
            for paragraph in read(f)["paragraphs"]:
                units = [
                    (
                        u.get("source_en") or "".join(p["t"] for p in u.get("en", [])),
                        "".join(p["t"] for p in u.get("zh", [])),
                    )
                    for u in paragraph["units"]
                ]
                for pair in paired_windows([(a.strip(), b.strip()) for a, b in units]):
                    add("new-book", work, pair)
    for split, _, doc, pair in flores():
        if split == "test":
            add("new-flores", doc, pair)
    lookup = read(BASE / "readback.json")
    wanted = [
        {normalize(pair[t]) for rows in candidates.values() for pair, _ in rows}
        for t in (0, 1)
    ]
    observed = [set(), set()]

    def screen(s, t):
        if len(s) <= 96 and (k := normalize(s)) in wanted[t]:
            observed[t].add(k)

    for tag in ("train", "dev", "test", "news-train", "news-held"):
        for r in read(OLD / (tag + ".json")):
            for t, s in enumerate(r["text"]):
                screen(s, t)
    for pair in opus_source("train"):
        for t, s in enumerate(pair):
            screen(s, t)

    def from_cells(seq, t):
        screen("".join(lookup[str(int(i))] for i in seq if i), t)

    for chunk in np.array_split(np.load(BOOK / "cells.npy", mmap_mode="r"), 128):
        for pair in chunk:
            for t, s in enumerate(pair):
                from_cells(s[: np.count_nonzero(s)], t)
    for r in read(FIRST / "contrasts.json"):
        for pair in r["cells"]:
            for t, s in enumerate(pair):
                from_cells(s, t)
    for r in read(ROOT / "artifacts/translation_paraphrases_v1/examples.json"):
        for t in (0, 1):
            for seq in [r["target"][t], *r["variants"][t]]:
                from_cells(seq, t)
    for t, lang in enumerate(("en", "zh")):
        for r in published(lang, "train").values():
            for k in ("sentence1", "sentence2"):
                screen(r[k], t)
    # Pixel collisions across transliterations or typographic aliases can survive text normalization.
    physical = [set() for _ in range(2)]
    for arr in [
        np.load(BOOK / "cells.npy", mmap_mode="r"),
        np.load(WORK / "replacement.npz")["cells"],
    ]:
        for t in (0, 1):
            physical[t].update(map(bytes, np.ascontiguousarray(arr[:, t])))
    kept = []
    for group in ("new-book", "new-flores"):
        ordered = {
            doc: sorted(
                rows,
                key=lambda z: rank("reading-rehearsal/" + doc + "".join(z[0])),
                reverse=True,
            )
            for (g, doc), rows in candidates.items()
            if g == group
        }
        picked = 0
        for _ in range(6 if group == "new-book" else 1):
            for doc in sorted(ordered, key=lambda d: rank("reading-rehearsal/" + d)):
                while ordered[doc] and picked < 24:
                    pair, cells = ordered[doc].pop()
                    keys = [normalize(s) for s in pair]
                    bits = [
                        np.pad(np.asarray(v, np.uint16), (0, 96 - len(v))).tobytes()
                        for v in cells
                    ]
                    if any(
                        not k or k in observed[t] or bits[t] in physical[t]
                        for t, k in enumerate(keys)
                    ):
                        continue
                    for t, k in enumerate(keys):
                        observed[t].add(k)
                        physical[t].add(bits[t])
                    kept.append(dict(domain=group, doc=doc, text=pair, cells=cells))
                    picked += 1
                    break
        if picked != 24:
            raise ValueError((group, picked))
    dump(WORK / "fresh.json", kept)
    dump(
        PUB / "fresh.json",
        dict(
            works=works,
            passages=len(kept),
            articles=len({r["doc"] for r in kept if r["domain"] == "new-flores"}),
            sha256=sha(WORK / "fresh.json"),
        ),
    )


if __name__ == "__main__":
    reserve()

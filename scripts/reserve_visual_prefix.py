"""Reserve new document families and screen whole physical or normalized training sides."""

from collections import defaultdict
from functools import lru_cache
import numpy as np
from scripts.prepare_book_rehearsal import (
    ROOT,
    BASE,
    BOOK,
    WORK as PREVIOUS,
    Ink,
    read,
    dump,
    sha,
)
from scripts.prepare_book_breadth import paired_windows, rank, flores, talks
from scripts.prepare_paired_ink import FIRST, OLD, key, opus_source, published

WORK = ROOT / "artifacts/prefix_recovery_v1"
PUB = ROOT / "references/evidence/prefix_recovery_v1"
NAMES = ("emma", "hamlet", "bel-ami", "dead-souls")


def reserve():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    old = [
        r for split in ("test", "dev") for r in read(BOOK / (split + ".json"))
    ] + read(PREVIOUS / "fresh.json")
    used_docs = {r["doc"] for r in old}
    used_docs.update(
        read(ROOT / "references/evidence/book_breadth_v1/corpus.json")[
            "local_book_sources"
        ]
    )
    if used_docs.intersection(NAMES):
        raise ValueError("Work previously used")
    pool = defaultdict(list)
    ink = Ink(np.load(BASE / "palette.npy"))
    vocab = [set(v) for v in read(BASE / "inventories.json")]

    # Cap candidates per document in reproducible hash order before expensive screening.
    def collect(group, doc, pairs):
        opts = sorted(
            set(
                (a.strip(), b.strip())
                for a, b in pairs
                if 4 <= len(a) <= 96 and 2 <= len(b) <= 96
            ),
            key=lambda v: rank("recover/" + doc + repr(v)),
        )
        for pair in opts:
            if not any(c.isascii() and c.isalpha() for c in pair[0]) or not any(
                "\u3400" <= c <= "\u9fff" for c in pair[1]
            ):
                continue
            cells = [ink.encode(v) for v in pair]
            if all(
                set(ids) <= allowed and not ink.bad.intersection(s)
                for ids, allowed, s in zip(cells, vocab, pair, strict=True)
            ):
                pool[group, doc].append((pair, cells))
                if len(pool[group, doc]) >= 24:
                    break

    for doc in NAMES:
        pairs = []
        for f in sorted(
            (ROOT.parent / "ZhJpBook/books" / doc).glob(
                "work/trilingual/interlinear/chunks/*.json"
            )
        ):
            for para in read(f)["paragraphs"]:
                units = [
                    (
                        u.get("source_en") or "".join(a["t"] for a in u["en"]),
                        "".join(a["t"] for a in u["zh"]),
                    )
                    for u in para["units"]
                ]
                pairs.extend(paired_windows(units))
        collect("book", doc, pairs)
    for stream in (flores, talks):
        docs = defaultdict(list)
        for split, domain, doc, pair in stream():
            if split == "test" and doc not in used_docs:
                docs[doc].append(pair)
        for doc in sorted(docs, key=lambda v: rank("new-prefix-doc/" + v)):
            collect(domain, doc, docs[doc])
            if sum(g == domain for g, d in pool) >= 55:
                break
    normalize = lru_cache(65536)(key)
    wanted = [
        {normalize(p[t]) for vals in pool.values() for p, _ in vals} for t in (0, 1)
    ]
    forbidden = [set() for _ in range(2)]

    def screen(s, t):
        if len(s) <= 96 and (v := normalize(s)) in wanted[t]:
            forbidden[t].add(v)

    for tag in ("train", "dev", "test", "news-train", "news-held"):
        for r in read(OLD / (tag + ".json")):
            for t, s in enumerate(r["text"]):
                screen(s, t)
    for pair in opus_source("train"):
        for t, s in enumerate(pair):
            screen(s, t)
    for r in old:
        for t, s in enumerate(r["text"]):
            screen(s, t)
    for t, lang in enumerate(("en", "zh")):
        for r in published(lang, "train").values():
            for k in ("sentence1", "sentence2"):
                screen(r[k], t)
    lookup = read(BASE / "readback.json")

    def raster_screen(ids, t):
        screen("".join(lookup[str(int(n))] for n in ids if n), t)

    for r in read(FIRST / "contrasts.json"):
        for pair in r["cells"]:
            for t, ids in enumerate(pair):
                raster_screen(ids, t)
    for r in read(ROOT / "artifacts/translation_paraphrases_v1/examples.json"):
        for t in (0, 1):
            for ids in [r["target"][t], *r["variants"][t]]:
                raster_screen(ids, t)
    physical = [set(), set()]
    for a in (
        np.load(BOOK / "cells.npy", mmap_mode="r"),
        np.load(PREVIOUS / "replacement.npz")["cells"],
    ):
        for t in (0, 1):
            physical[t].update(map(bytes, np.ascontiguousarray(a[:, t])))
            for ids in a[:, t]:
                raster_screen(ids[: np.count_nonzero(ids)], t)
    cohort = []
    for g in ("book", "talk", "flores"):
        used = 0
        for _ in range(6 if g == "book" else 1):
            for (group, doc), options in pool.items():
                if group != g or used >= 24:
                    continue
                while options:
                    pair, cells = options.pop(0)
                    keys = [normalize(s) for s in pair]
                    packed = [
                        np.pad(np.asarray(s, np.uint16), (0, 96 - len(s))).tobytes()
                        for s in cells
                    ]
                    if any(
                        not k or k in forbidden[t] or packed[t] in physical[t]
                        for t, k in enumerate(keys)
                    ):
                        continue
                    for t, k in enumerate(keys):
                        forbidden[t].add(k)
                        physical[t].add(packed[t])
                    cohort.append(dict(domain=g, doc=doc, text=pair, cells=cells))
                    used += 1
                    break
        if used != 24:
            raise ValueError((g, used))
    dump(WORK / "test.json", cohort)
    dump(
        PUB / "reservation.json",
        dict(
            works=NAMES,
            documents={
                g: len({v["doc"] for v in cohort if v["domain"] == g})
                for g in ("book", "talk", "flores")
            },
            passages=len(cohort),
            sha256=sha(WORK / "test.json"),
        ),
    )


if __name__ == "__main__":
    reserve()

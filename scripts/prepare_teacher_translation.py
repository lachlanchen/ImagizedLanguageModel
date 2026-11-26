"""Reserve unseen books and talks and sample bilingual training sources by domain/direction."""

from collections import defaultdict
from functools import lru_cache
import numpy as np
from scripts.reserve_visual_prefix import (
    ROOT,
    BOOK,
    BASE,
    PREVIOUS,
    WORK as PREFIX,
    FIRST,
    OLD,
    Ink,
    read,
    dump,
    sha,
    key,
    opus_source,
    paired_windows,
    published,
    rank,
    talks,
)
from scripts.reserve_wide_translation import WORK as WIDTH
from scripts.prepare_visual_rollins import training_rows

WORK = ROOT / "artifacts/teacher_translation_v1"
PUB = ROOT / "references/evidence/teacher_translation_v1"
MODEL = ROOT / "models/teacher_translation"
WORKS = ("crime-and-punishment", "madame-bovary", "red-and-black", "fathers-and-sons")


def previous():
    return (
        read(BOOK / "test.json")
        + read(BOOK / "dev.json")
        + read(PREVIOUS / "fresh.json")
        + read(PREFIX / "test.json")
        + read(WIDTH / "fresh.json")
    )


def novel(doc):
    folder = ROOT.parent / "ZhJpBook/books" / doc / "work/trilingual/interlinear/chunks"
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise ValueError(f"Missing book {doc}")
    for p in paths:
        for para in read(p)["paragraphs"]:
            yield from paired_windows(
                [
                    (
                        u.get("source_en") or "".join(a["t"] for a in u["en"]),
                        "".join(a["t"] for a in u["zh"]),
                    )
                    for u in para["units"]
                ]
            )


def held_cases(old, bank, lookup):
    ink = Ink(np.load(BASE / "palette.npy"))
    allowed = [set(v) for v in read(BASE / "inventories.json")]
    used = {r["doc"] for r in old}
    used.update(
        read(ROOT / "references/evidence/book_breadth_v1/corpus.json")[
            "local_book_sources"
        ]
    )
    if used.intersection(WORKS):
        raise ValueError("Selected a previous document")
    pool = defaultdict(list)

    def add(group, doc, pairs):
        opts = {
            tuple(s.strip() for s in pair)
            for pair in pairs
            if all(3 <= len(s.strip()) <= 96 for s in pair)
        }
        for pair in sorted(opts, key=lambda p: rank("distill/" + doc + repr(p))):
            if not any(c.isascii() and c.isalpha() for c in pair[0]) or not any(
                "\u3400" <= c <= "\u9fff" for c in pair[1]
            ):
                continue
            cells = [ink.encode(s) for s in pair]
            if all(
                set(c) <= v and not ink.bad.intersection(s)
                for c, s, v in zip(cells, pair, allowed, strict=True)
            ):
                pool[(group, doc)].append(
                    dict(domain=group, doc=doc, text=pair, cells=cells)
                )
            if len(pool[group, doc]) >= 32:
                break

    for doc in WORKS:
        add("book", doc, novel(doc))
    collection = defaultdict(list)
    for split, _, doc, pair in talks():
        if split == "test" and doc not in used:
            collection[doc].append(pair)
    for doc in sorted(collection, key=lambda d: rank("distill-talk/" + d))[:70]:
        add("talk", doc, collection[doc])
    norm = lru_cache(65536)(key)
    wanted = [
        {norm(r["text"][t]) for rows in pool.values() for r in rows} for t in (0, 1)
    ]
    seen = [set(), set()]

    def scan(t, s):
        if len(s) <= 96 and (k := norm(s)) in wanted[t]:
            seen[t].add(k)

    for rows in [old] + [
        read(OLD / (name + ".json"))
        for name in ("train", "dev", "test", "news-train", "news-held")
    ]:
        for r in rows:
            for t, s in enumerate(r["text"]):
                scan(t, s)
    for pair in opus_source("train"):
        for t, s in enumerate(pair):
            scan(t, s)
    for t, lang in enumerate(("en", "zh")):
        for r in published(lang, "train").values():
            for k in ("sentence1", "sentence2"):
                scan(t, r[k])

    def decoded(t, seq):
        scan(t, "".join(lookup[str(int(k))] for k in seq if k))

    for r in read(FIRST / "contrasts.json"):
        for pair in r["cells"]:
            for t, s in enumerate(pair):
                decoded(t, s)
    for r in read(ROOT / "artifacts/translation_paraphrases_v1/examples.json"):
        for t in (0, 1):
            for seq in [r["target"][t], *r["variants"][t]]:
                decoded(t, seq)
    # Screen source and target images in both the original corpus and the replay replacements.
    for a in (
        bank,
        np.load(BOOK / "cells.npy", mmap_mode="r")[
            np.load(PREVIOUS / "replacement.npz")["indices"]
        ],
    ):
        for t in (0, 1):
            for side in a[:, t]:
                decoded(t, side[: np.count_nonzero(side)])
    rows = []
    for group, per_doc in (("book", 6), ("talk", 1)):
        subset = []
        for (g, doc), options in pool.items():
            if group != g or len(subset) >= 24:
                continue
            kept = 0
            for r in options:
                keys = [norm(s) for s in r["text"]]
                if any(not k or k in seen[t] for t, k in enumerate(keys)):
                    continue
                subset.append(r)
                kept += 1
                for t, k in enumerate(keys):
                    seen[t].add(k)
                if kept == per_doc:
                    break
        if len(subset) != 24:
            raise ValueError(("Incomplete reserved group", group, len(subset)))
        rows.extend(subset)
    return rows


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    frozen = PUB / "design.json"
    if frozen.exists():
        manifest = read(frozen)
        if any(sha(WORK / p) != d for p, d in manifest["files"].items()):
            raise ValueError("Reserved inputs changed")
        return
    old = previous()
    bank = training_rows()
    look = read(BASE / "readback.json")
    test = held_cases(old, bank, look)
    audit = read(ROOT / "references/evidence/prefix_recovery_v1/alignment-meaning.json")
    banned = {r["index"] for r in audit["records"] if r["group"] in ("book", "talk")}
    banned.update(
        r["training_index"]
        for r in read(
            ROOT / "references/evidence/wide_translation_v1/teacher-targets.json"
        )["records"]
    )
    banned.update(map(int, np.load(PREVIOUS / "replacement.npz")["indices"]))
    used = {key(s) for r in old + test for s in r["text"]}
    dom = np.load(BOOK / "domains.npy")
    rng = np.random.default_rng(413897)
    training = []
    for di, d in enumerate(("book", "talk")):
        order = rng.permutation(np.flatnonzero(dom == di))
        counts = [0, 0]
        for i in order:
            if i in banned:
                continue
            pair = [s[: np.count_nonzero(s)].tolist() for s in bank[i]]
            strings = [
                "".join(look[str(n)] for n in side).replace("\u3000", " ")
                for side in pair
            ]
            ks = list(map(key, strings))
            if not all(ks) or any(k in used for k in ks):
                continue
            t = sum(counts) % 2
            training.append(
                dict(
                    id=int(i),
                    domain=d,
                    t=t,
                    source=strings[1 - t],
                    ref=strings[t],
                    input=pair[1 - t],
                    target=pair[t],
                )
            )
            used.update(ks)
            counts[t] += 1
            if min(counts) >= 1024:
                break
        if counts != [1024, 1024]:
            raise ValueError((d, counts))
    rng.shuffle(training)
    dump(WORK / "held.json", test)
    dump(WORK / "sources.json", training)
    dump(
        frozen,
        dict(
            works=WORKS,
            held_pairs=len(test),
            held_documents=len({r["doc"] for r in test}),
            training_requests=len(training),
            files={p: sha(WORK / p) for p in ("held.json", "sources.json")},
            program=sha(ROOT / "scripts/prepare_teacher_translation.py"),
        ),
    )


if __name__ == "__main__":
    prepare()

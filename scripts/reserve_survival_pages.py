"""Screen a new official validation subset against both full source training pools."""

from itertools import chain
import numpy as np
from scripts.prepare_multiple_ink_runs import (
    WORK,
    PUB,
    english,
    chinese,
    key,
    normalize,
    fragments,
    read,
    dump,
    sha,
    multi,
    CorpusFace,
    prepared_files,
)


def main(work=WORK, public=PUB):
    names = {"confirm.json": ("sha256",), "confirm-tiles.npy": ("tiles",)}
    prepared_files(
        work,
        public / "confirmation-split.json",
        names,
        lambda stage: reserve(work, stage),
    )


def reserve(
    work, stage, prior=None, groups=None, seed="survival-confirm-v1 ", pool_limit=200
):
    path = stage / "confirm.json"
    if prior is None:
        prior = read(work / "held.json")
    if groups is None:
        groups = (("en", True, 96), ("zh", True, 96))
    banq = {normalize(r["q"]) for r in prior}
    banp = {normalize(r["p"]) for r in prior}
    old = set().union(*(fragments(r["p"], 24) for r in prior))
    pool = []
    owners = {}
    for lang, plural, _ in groups:
        rows = english("valid") if lang == "en" else chinese("valid")
        candidates = sorted(
            (
                r
                for r in rows
                if (multi(r) if plural else all(len(a) == 1 for a in r["refs"]))
            ),
            key=lambda r: key(seed + r["id"] + r["p"]),
        )
        n = 0
        for r in candidates:
            q, p = normalize(r["q"]), normalize(r["p"])
            if not p or not q or q in banq or p in banp:
                continue
            f = fragments(r["p"], 1)
            if f & old:
                continue
            index = len(pool)
            for v in f:
                owners.setdefault(v, []).append(index)
            banq.add(q)
            banp.add(p)
            pool.append(r)
            n += 1
            if n == pool_limit:
                break
    rejected = set()
    candidate_q = {normalize(r["q"]): i for i, r in enumerate(pool)}
    for r in chain(english("train"), chinese("train")):
        q = normalize(r["q"])
        if q in candidate_q:
            rejected.add(candidate_q[q])
        for v in fragments(r["p"], 32):
            if v in owners:
                rejected.update(owners[v])
    with np.load(work / "training.npz") as z:
        base = z["tiles"]
    lookup = {a.tobytes(): i for i, a in enumerate(base) if i >= 2}
    new = []
    face = CorpusFace()

    def encode(text):
        seq = []
        for c in text:
            try:
                b = face.tile(c)
            except ValueError:
                b = face.tile("�")
            if b not in lookup:
                lookup[b] = len(base) + len(new)
                new.append(b)
            seq.append(lookup[b])
        return seq

    chosen = []
    for lang, plural, quota in groups:
        rows = [
            r
            for i, r in enumerate(pool)
            if i not in rejected and r["lang"] == lang and multi(r) == plural
        ][:quota]
        assert len(rows) == quota, (lang, plural, len(rows), quota)
        for r in rows:
            r["pi"] = encode(r["p"])
            r["qi"] = encode(r["q"])
            r["visible_parts"] = [all(w in r["p"] for w in a) for a in r["refs"]]
        chosen.extend(rows)
    dump(path, chosen)
    extra = stage / "confirm-tiles.npy"
    np.save(extra, np.frombuffer(b"".join(new), np.uint8).reshape(-1, 128))
    return dict(
        sha256=sha(path),
        tiles=sha(extra),
        candidates=len(pool),
        overlap_rejected=len(rejected),
        ids=[r["id"] for r in chosen],
    )


if __name__ == "__main__":
    main()

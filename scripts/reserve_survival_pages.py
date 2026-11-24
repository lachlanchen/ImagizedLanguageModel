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
)


def main():
    path = WORK / "confirm.json"
    receipt = PUB / "confirmation-split.json"
    if receipt.exists():
        assert sha(path) == read(receipt)["sha256"]
        return
    prior = read(WORK / "held.json")
    banq = {normalize(r["q"]) for r in prior}
    banp = {normalize(r["p"]) for r in prior}
    old = set().union(*(fragments(r["p"], 24) for r in prior))
    pool = []
    owners = {}
    for lang, rows in (("en", english("valid")), ("zh", chinese("valid"))):
        candidates = sorted(
            (r for r in rows if multi(r)),
            key=lambda r: key("survival-confirm-v1 " + r["id"] + r["p"]),
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
            if n == 200:
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
    with np.load(WORK / "training.npz") as z:
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
    for lang in ("en", "zh"):
        rows = [
            r for i, r in enumerate(pool) if i not in rejected and r["lang"] == lang
        ][:96]
        assert len(rows) == 96
        for r in rows:
            r["pi"] = encode(r["p"])
            r["qi"] = encode(r["q"])
            r["visible_parts"] = [all(w in r["p"] for w in a) for a in r["refs"]]
        chosen.extend(rows)
    dump(path, chosen)
    extra = WORK / "confirm-tiles.npy"
    np.save(extra, np.frombuffer(b"".join(new), np.uint8).reshape(-1, 128))
    dump(
        receipt,
        dict(
            sha256=sha(path),
            tiles=sha(extra),
            candidates=len(pool),
            overlap_rejected=len(rejected),
            ids=[r["id"] for r in chosen],
        ),
    )


if __name__ == "__main__":
    main()

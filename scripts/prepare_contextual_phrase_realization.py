"""Select independent chapter-held-out contexts and extra training-only lexical alignments."""

from collections import Counter
from pathlib import Path
import tarfile

from scripts.prepare_predicate_likelihood import (
    OUT as INITIAL,
    EVENT,
    ROOT,
    BASE,
    PREDICATES,
    clauses,
    read_rows,
    OLD,
    Rasterizer,
    order,
    sha,
    window,
    load,
    write_new,
)
from scripts.prepare_visual_edit_translation import pairs

OUT = ROOT / "artifacts/contextual_phrase_realization_v1"
ALIASES = {
    "离开": ("离开", "离去", "走开"),
    "距离": ("距离", "相距", "相隔"),
    "砍伐": ("砍伐", "砍下", "砍掉", "砍倒", "砍去", "砍了", "采伐"),
    "攻打": ("攻打", "讨伐", "进攻", "征讨"),
}


def records(x, y):
    for p, gs in PREDICATES.items():
        if x.count(p) != 1:
            continue
        a, b = clauses(x), clauses(y)
        if len(a) != len(b):
            continue
        for s, t in zip(a, b):
            if p not in s or not (3 <= len(s) <= 22 and 2 <= len(t) <= 40):
                continue
            if any(w in s for g in gs for w in ALIASES[g]):
                continue
            found = [(g, w) for g in gs for w in ALIASES[g] if w in t]
            if len(found) == 1:
                yield dict(
                    source=s,
                    target=t,
                    predicate=p,
                    gloss=found[0][0],
                    expression=found[0][1],
                )


def prepare():
    OUT.mkdir()
    core = load(INITIAL / "train.json")
    train = [dict(r, expression=r["gloss"]) for r in core]
    parent = read_rows("train")
    held = read_rows("heldout")
    train_ch = {r["group"] for r in parent}
    blocked = train_ch | {r["group"] for r in held}
    forbidden = {
        c
        for group in (parent, held)
        for r in group
        for k in ("source", "new", "old")
        for c in clauses(r[k])
    }
    for name, roster, _ in OLD:
        for r in load(ROOT / "artifacts" / name / (roster + ".json")):
            blocked.add(r["chapter"])
            for k in ("source", "reference", "target"):
                if isinstance(r.get(k), str):
                    forbidden.update(clauses(r[k]))
    blocked.update(
        r["chapter"]
        for r in load(
            ROOT
            / "references/evidence/block_translation_20261118/calibration_mapping.json"
        )
    )
    for d, names in [
        (EVENT, ("train", "roles", "broad")),
        (INITIAL, ("proposed", "train")),
    ]:
        for n in names:
            for r in load(d / (n + ".json")):
                blocked.add(r["chapter"])
                for k in ("source", "target"):
                    forbidden.update(clauses(r[k]))
                    if "a" in r:
                        forbidden.update(
                            clauses(
                                r[k].translate(
                                    str.maketrans({r["a"]: r["b"], r["b"]: r["a"]})
                                )
                            )
                        )
    archived = BASE / "archive.tar.gz"
    assert sha(archived) == load(INITIAL / "preparation.json")["archive"]
    extra = []
    pool = []
    renderer = Rasterizer()

    def pixels(text):
        return b"".join(renderer.tile(c) for c in text)

    with tarfile.open(archived, "r|gz") as tar:
        for item in tar:
            path = Path(item.name)
            if (
                len(path.parts) < 4
                or path.parts[1] != "双语数据"
                or path.name != "bitext.txt"
            ):
                continue
            ch = "/".join(path.parts[2:-1])
            if ch not in train_ch and ch in blocked:
                continue
            for i, (x, y) in enumerate(pairs(tar.extractfile(item).read())):
                for r in records(x, y):
                    if r["source"] in forbidden or r["target"] in forbidden:
                        continue
                    try:
                        pixels(r["source"])
                        pixels(r["target"])
                    except ValueError:
                        continue
                    (extra if ch in train_ch else pool).append(
                        dict(r, chapter=ch, ordinal=i)
                    )

    def key(r):
        return order(f"context-phrases/{r['chapter']}/{r['ordinal']}/{r['source']}")

    used = {pixels(r[k]) for r in train for k in ("source", "target")}
    for r in sorted(extra, key=key):
        ks = {pixels(r[k]) for k in ("source", "target")}
        if ks & used:
            continue
        used.update(ks)
        train.append(r)
    # Screen physical aliases even where the transcribed source strings differ.
    # Hash relevant forbidden strings only at lengths appearing in the test pool.
    import hashlib

    # Distinct, renderable Han and punctuation images can share a physical cell.
    relevant = {len(r[k]) for r in pool for k in ("source", "target")}

    def dig(s):
        return hashlib.sha256(pixels(s)).digest()

    old_ink = set()
    for s in forbidden:
        if len(s) in relevant:
            try:
                old_ink.add(dig(s))
            except ValueError:
                pass
    window_train = {window(r["source"], r["predicate"]) for r in train}
    pool = [
        r for r in pool if all(dig(r[k]) not in old_ink for k in ("source", "target"))
    ]
    pool.sort(
        key=lambda r: (window(r["source"], r["predicate"]) in window_train, key(r))
    )
    selected = []
    used_ch = set()
    for p, (g, h) in PREDICATES.items():
        count = 0
        for a in (r for r in pool if r["gloss"] == g):
            ak = {pixels(a[k]) for k in ("source", "target")}
            if ak & used or a["chapter"] in used_ch:
                continue
            for b in (
                r
                for r in pool
                if r["gloss"] == h and len(r["source"]) == len(a["source"])
            ):
                bk = {pixels(b[k]) for k in ("source", "target")}
                if b["chapter"] in used_ch | {a["chapter"]} or bk & (ak | used):
                    continue
                selected.extend(
                    dict(r, new_window=window(r["source"], p) not in window_train)
                    for r in (a, b)
                )
                used.update(ak | bk)
                used_ch.update((a["chapter"], b["chapter"]))
                count += 1
                break
            if count == 8:
                break
    write_new(OUT / "train.json", train)
    write_new(OUT / "proposed.json", selected)
    write_new(
        OUT / "preparation.json",
        dict(
            extra=len(train) - len(core),
            training_counts=dict(Counter(r["gloss"] for r in train)),
            prior_chapters=len(blocked),
            old_physical_clauses=len(old_ink),
            eligible=dict(Counter(r["gloss"] for r in pool)),
            train_sha=sha(OUT / "train.json"),
            proposed_sha=sha(OUT / "proposed.json"),
        ),
    )
    for i, r in enumerate(selected):
        print(i, r["source"], r["target"], r["new_window"])


if __name__ == "__main__":
    prepare()

"""Partition paired optical translations and screen held families against prior exposure."""

from collections import Counter
from functools import lru_cache

import numpy as np
from scripts.prepare_translation_paraphrases import (
    BASE,
    ROOT,
    OLD,
    FIRST,
    WORK as PRIOR,
    Ink,
    key,
    sha,
    read,
    dump,
    published,
    opus_source,
)
from scripts.prepare_proposition_translation import hashed

WORK = ROOT / "artifacts/paired_ink_v1"
PUB = ROOT / "references/evidence/paired_ink_v1"
MODELS = ROOT / "models/paired_ink"


def main():
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    ink = Ink(np.load(BASE / "palette.npy"))
    glyph = lru_cache(None)(lambda c: ink.encode(c)[0])
    inv = [set(s) for s in read(BASE / "inventories.json")]
    norm = lru_cache(60000)(key)
    counts = Counter()

    def candidates(split):
        en, zh = (published(t, split) for t in ("en", "zh"))
        if en.keys() != zh.keys():
            raise ValueError("PAWS-X alignment")
        for i, a in en.items():
            b = zh[i]
            if a["label"] != 0 or b["label"] != 0:
                continue
            forms = [[m[k] for m in (a, b)] for k in ("sentence1", "sentence2")]
            if any(not 0 < len(s) <= 96 for pair in forms for s in pair):
                continue
            if any(norm(forms[0][t]) == norm(forms[1][t]) for t in (0, 1)):
                continue
            cells = [[[glyph(c) for c in s] for s in pair] for pair in forms]
            if any(
                not set(z[t]) <= inv[t] or ink.bad.intersection(forms[j][t])
                for j, z in enumerate(cells)
                for t in (0, 1)
            ):
                continue
            yield dict(id=i, forms=forms, cells=cells)

    choices = sorted(
        candidates("test"), key=lambda r: hashed("ink-pair-20261126/" + str(r["id"]))
    )
    wanted = [{norm(z[t]) for r in choices for z in r["forms"]} for t in (0, 1)]
    seen = [set(), set()]

    def exclude(s, t):
        if len(s) <= 96 and (k := norm(s)) in wanted[t]:
            seen[t].add(k)

    for name in ("train", "news-train"):
        for r in read(OLD / (name + ".json")):
            for t, s in enumerate(r["text"]):
                exclude(s, t)
    for z in opus_source("train"):
        for t, s in enumerate(z):
            exclude(s, t)
    lookup = read(BASE / "readback.json")

    def exclude_cells(z, t):
        exclude("".join(lookup[str(c)] for c in z), t)

    for r in read(FIRST / "contrasts.json"):
        for z in r["cells"]:
            for t in (0, 1):
                exclude_cells(z[t], t)
    for r in read(PRIOR / "examples.json"):
        for t in (0, 1):
            for z in (r["target"][t], *r["variants"][t]):
                exclude_cells(z, t)
    for r in read(PRIOR / "graded.json")["rows"]:
        exclude(r["source"], 1 - r["t"])
        exclude(r["ref"], r["t"])
        exclude(r["foil"], r["t"])
    # Check every PAWS-X training sentence, including those rejected from fitting.
    for t, lang in enumerate(("en", "zh")):
        for r in published(lang, "train").values():
            for f in ("sentence1", "sentence2"):
                exclude(r[f], t)
    held = []
    for r in choices:
        if any(norm(z[t]) in seen[t] for z in r["forms"] for t in (0, 1)):
            continue
        held.append(r)
        for z in r["forms"]:
            for t in (0, 1):
                seen[t].add(norm(z[t]))
        if len(held) == 48:
            break
    if len(held) != 48:
        raise ValueError("Incomplete reservation")
    reserved = [{norm(z[t]) for r in held for z in r["forms"]} for t in (0, 1)]
    seen_train = set()
    train = []
    for r in candidates("train"):
        fingerprint = tuple(sorted(tuple(norm(s) for s in z) for z in r["forms"]))
        if fingerprint in seen_train:
            continue
        if any(norm(z[t]) in reserved[t] for z in r["forms"] for t in (0, 1)):
            raise ValueError("Overlap")
        seen_train.add(fingerprint)
        train.append(dict(id=r["id"], cells=r["cells"]))
    counts.update(train=len(train), held=len(held), eligible_test=len(choices))
    for name, rows in (("train", train), ("held", held)):
        dump(WORK / (name + ".json"), rows)
    dump(
        PUB / "split.json",
        dict(
            counts=counts,
            held_ids=[r["id"] for r in held],
            files={s: sha(WORK / (s + ".json")) for s in ("train", "held")},
        ),
    )
    print(counts, flush=True)


if __name__ == "__main__":
    main()

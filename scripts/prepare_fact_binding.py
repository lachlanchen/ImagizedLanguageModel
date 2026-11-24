"""Align disjoint competing whole answers and reserve unseen visual binding questions."""

import json
from itertools import chain
import numpy as np
from scripts.prepare_complete_fact_sets import WORK as SETS, PUB as PREV
from scripts.prepare_multiple_ink_runs import (
    ROOT,
    WORK as OLD,
    chinese,
    english,
    normalize,
    fragments,
    key,
    read,
    dump,
    sha,
    CorpusFace,
    prepared_files,
)
from scripts.evaluate_visual_discourse import unpack

WORK = ROOT / "artifacts/fact_binding_v1"
PUB = ROOT / "references/evidence/fact_binding_v1"
MODEL = ROOT / "models/fact_binding"


def build(stage):
    cases = {s: {} for s in ("train", "valid")}
    for w in unpack(PREV / "binding-witnesses.json.gz"):
        cases[w["split"]][w["id"]] = w
    prior = (
        read(OLD / "held.json")
        + read(OLD / "confirm.json")
        + read(SETS / "confirm.json")
    )
    questions = {normalize(r["q"]) for r in prior}
    old = set().union(*(fragments(r["p"], 24) for r in prior))
    candidates = []
    owners = {}
    valid = sorted(
        (r for r in chinese("valid") if r["id"] in cases["valid"]),
        key=lambda r: key("fact-binding-v1 " + r["id"]),
    )
    for r in valid:
        q = normalize(r["q"])
        f = fragments(r["p"], 1)
        if q in questions or old & f:
            continue
        i = len(candidates)
        for word in f:
            owners.setdefault(word, []).append(i)
        candidates.append(r)
        questions.add(q)
        old.update(fragments(r["p"], 24))
    cq = {normalize(r["q"]): i for i, r in enumerate(candidates)}
    rejected = set()
    train = []
    for r in chain(chinese("train"), english("train")):
        q = normalize(r["q"])
        if q in cq:
            rejected.add(cq[q])
        for word in fragments(r["p"], 32):
            if word in owners:
                rejected.update(owners[word])
        if (
            r["lang"] == "zh"
            and r["id"] in cases["train"]
            and q not in questions
            and not (fragments(r["p"], 1) & old)
        ):
            train.append(r)
    held = [r for i, r in enumerate(candidates) if i not in rejected][:64]
    if not held or not train:
        raise ValueError("Empty source-disjoint split")
    with np.load(OLD / "training.npz") as f:
        values = list(map(bytes, f["tiles"]))
    lookup = {b: i for i, b in enumerate(values) if i >= 2}
    face = CorpusFace()

    def enc(text):
        out = []
        for c in text:
            try:
                bits = face.tile(c)
            except ValueError:
                bits = face.tile("�")
            if bits not in lookup:
                lookup[bits] = len(values)
                values.append(bits)
            out.append(lookup[bits])
        return out

    for split, records in (("train", train), ("valid", held)):
        for r in records:
            r["pi"] = enc(r["p"])
            r["qi"] = enc(r["q"])
            w = cases[split][r["id"]]
            a, b = w["left"], w["right"]
            r["paths"] = [
                a,
                b,
                [p for p in a if p != w["x"]] + [w["y"]],
                [p for p in b if p != w["y"]] + [w["x"]],
            ]
        dump(stage / (split + ".json"), records)
    np.save(
        stage / "tiles.npy", np.frombuffer(b"".join(values), np.uint8).reshape(-1, 128)
    )
    return dict(
        train=len(train),
        held=len(held),
        candidates=len(candidates),
        training_overlap=len(rejected),
        ids=[r["id"] for r in held],
        files={f: sha(stage / f) for f in ("train.json", "valid.json", "tiles.npy")},
    )


if __name__ == "__main__":
    prepared_files(
        WORK,
        PUB / "corpus.json",
        {f: ("files", f) for f in ("train.json", "valid.json", "tiles.npy")},
        build,
    )

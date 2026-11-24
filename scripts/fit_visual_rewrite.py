"""Fit a visual edit dictionary and measure it on an earlier development cohort."""

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json
import numpy as np
from ilm.visual_lm.visual_rewrite import fit, predict
from scripts.prepare_sentence_context import (
    ROOT,
    PREVIOUS,
    PREVIOUS_PUB,
    FOURTH,
    Ink,
    read,
    dump,
    sha,
)
from scripts.evaluate_visual_discourse import unpack

DEST = ROOT / "models/visual_rewrite"
PUB = ROOT / "references/evidence/visual_rewrite_v1"


def restore(path):
    result = {}
    for left, right, num, denom, n, rule in read(path):
        a, b, u, v = rule
        result[tuple(left), tuple(right)] = (
            Fraction(num, denom),
            n,
            (a, b, tuple(u), tuple(v)),
        )
    return result


def main():
    spec = read(PREVIOUS_PUB / "corpus.json")
    for name in ("tiles.npy", "strings.npz", "writer.json"):
        if sha(PREVIOUS / name) != spec["files"][name]:
            raise ValueError(name)
    ink = Ink(np.load(PREVIOUS / "tiles.npy"))
    with np.load(PREVIOUS / "strings.npz") as z:
        ids, ptr = z["ids"], z["ptr"]
    strings = [tuple(ids[a:b].tolist()) for a, b in zip(ptr[:-1], ptr[1:])]
    rows = [
        (strings[q], strings[f], [strings[y] for y in ys], family)
        for q, _, f, ys, _, family in read(PREVIOUS / "writer.json")
    ]
    index, rc = fit(rows)
    DEST.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    path = DEST / "boundaries.json"
    dump(
        path,
        [
            [a, b, p.numerator, p.denominator, n, rule]
            for (a, b), (p, n, rule) in sorted(index.items())
        ],
    )
    rebuilt = restore(path)
    if rebuilt != index:
        raise ValueError("Released rule table differs")
    held = read(FOURTH / "confirmation.json")
    full = unpack(FOURTH / "whole.json.gz")["original"]
    stats = defaultdict(Counter)
    samples = []
    for r, base in zip(held, full, strict=True):
        q = tuple(ink.encode(r["q"]))
        ys = [tuple(ink.encode(y)) for y in r["answers"]]

        def decode(name):
            raw = bytes.fromhex(base[name])
            return tuple(ink.byink[raw[i : i + 128]] for i in range(0, len(raw), 128))

        # Canonical order is tied to the original writer's tile table at training.
        for c in (base["fact_text"], base["text"]):
            ink.encode(c)
        fact = decode("fact_hex")
        before = decode("ink_hex")
        out = predict(rebuilt, q, fact)
        correct = base["stop"] and before in ys
        after = correct if out is None else out in ys
        c = stats[r["task"]]
        c.update(
            n=1,
            applied=out is not None,
            original=correct,
            edited=after,
            recovered=after and not correct,
            lost=correct and not after,
        )
        if out is not None:
            true = tuple(ink.encode(r["facts"]["answer"]))
            c["right_with_supplied_fact_on_applicable"] += (
                predict(rebuilt, q, true) in ys
            )
        if (after != correct) and len(samples) < 12:
            names = {i: c for c, i in ink.chars.items()}
            samples.append(
                dict(
                    question=r["q"],
                    fact=base["fact_text"],
                    original=base["text"],
                    edit="".join(names.get(i, "�") for i in out),
                    human=r["answers"],
                    gain=bool(after),
                )
            )
    dump(
        PUB / "development.json",
        dict(
            fit=rc,
            measures=stats,
            changes=samples,
            model_sha256=sha(path),
            corpus=sha(PREVIOUS_PUB / "corpus.json"),
            cohort=sha(FOURTH / "confirmation.json"),
            registered_protocol=sha(
                ROOT / "references/visual_rewrite_development_protocol.md"
            ),
        ),
    )
    print(rc, stats, flush=True)


if __name__ == "__main__":
    main()

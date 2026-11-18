"""Fit typed-by-pixels phrase spans and pass-through contexts from real paired clauses."""

from collections import Counter, defaultdict
import time
import torch
from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer, fit_head
from scripts.prepare_contextual_phrase_realization import (
    OUT,
    PREDICATES,
    load,
    sha,
    write_new,
    clauses,
    order,
)
from scripts.train_causal_glyph_translator import read_rows
from scripts.train_pictured_revision import composer
from scripts.train_paired_student import restore
from scripts.train_visual_event_roles import PARENT, Rasterizer, raster


def run():
    torch.set_num_threads(2)
    train = load(OUT / "train.json")
    seal = load(OUT / "admission.json")
    assert seal["train_sha"] == sha(OUT / "train.json") and seal["test_sha"] == sha(
        OUT / "test.json"
    )
    start = time.monotonic()
    renderer = Rasterizer()
    engine = ContextualPhraseRealizer(restore(PARENT), composer(), {})
    samples = defaultdict(list)
    counts = defaultdict(Counter)
    used = {r["source"] for r in train}

    def add(p, s, label):
        d, v = engine.features(raster(s + "。", renderer), [s.index(p)])[0]
        samples[renderer.tile(p)].append(
            (d, v, raster(label, renderer).tobytes() if label else None)
        )
        counts[p][label or "pass"] += 1

    for r in train:
        add(r["predicate"], r["source"], r["gloss"])
    pool = defaultdict(set)
    for r in read_rows("train"):
        x, y = clauses(r["source"]), clauses(r["new"])
        if len(x) != len(y):
            continue
        for s, t in zip(x, y):
            for p, gs in PREDICATES.items():
                if (
                    s in used
                    or not 2 <= len(s) <= 32
                    or s.count(p) != 1
                    or t.count(p) != 1
                    or any(g in t for g in gs)
                ):
                    continue
                i = s.index(p)
                bigrams = [s[i - 1 : i + 1] if i else "", s[i : i + 2]]
                if any(len(w) == 2 and w in t for w in bigrams):
                    pool[p].add(s)
    for p in PREDICATES:
        for s in sorted(pool[p], key=order)[:96]:
            add(p, s, None)
    for cell, examples in samples.items():
        engine.heads[cell] = fit_head(examples)
    path = OUT / "realizer.msgpack.gz"
    engine.save(path)
    write_new(
        OUT / "fit.json",
        dict(
            counts={p: dict(v) for p, v in counts.items()},
            seconds=time.monotonic() - start,
            bytes=path.stat().st_size,
            head_sha=sha(path),
            encoder_sha=sha(PARENT),
            train=seal["train_sha"],
        ),
    )


if __name__ == "__main__":
    run()

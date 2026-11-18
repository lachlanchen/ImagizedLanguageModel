"""Fit matched canonical and joint boundary writers using training-only pairs."""

from collections import Counter, defaultdict
import time

import torch

from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer, fit_head
from ilm.visual_lm.visual_phrase_boundary import VisualPhraseBoundary
from scripts.prepare_phrase_boundary_trial import (
    EVIDENCE,
    EVENT,
    INVENTORY,
    OUT,
    Rasterizer,
    clauses,
    load,
    order,
    read_rows,
    sha,
    write_new,
)
from scripts.prepare_visual_boundary_candidates import OTHER
from scripts.prepare_predicate_likelihood import PREDICATES
from scripts.train_visual_event_roles import PARENT, raster
from scripts.train_pictured_revision import composer
from scripts.train_paired_student import restore


def fit():
    torch.set_num_threads(2)
    start = time.monotonic()
    admission = load(EVIDENCE / "admission.json")
    assert admission["test_sha"] == sha(EVIDENCE / "test.json")
    assert admission["training_audit_sha"] == sha(OUT / "training_audit.json")
    assert admission["inventory_sha"] == sha(INVENTORY) and admission[
        "broad_sha"
    ] == sha(EVENT / "broad.json")
    rejected = {r["index"] for r in load(OUT / "training_audit.json") if not r["admit"]}
    all_rows = load(INVENTORY)["records"]
    records = [r for i, r in enumerate(all_rows) if i not in rejected]
    used = {r["source"] for r in all_rows}
    render = Rasterizer()
    engine = ContextualPhraseRealizer(restore(PARENT), composer(), {})
    model = VisualPhraseBoundary(engine)
    samples = defaultdict(list)
    counts = Counter()

    def observe(p, text, sense):
        source = raster(text + "。", render)
        i = text.index(p)
        dense, sparse = engine.features(source, [i])[0]
        ink = raster(sense, render).tobytes() if sense else None
        samples[render.tile(p)].append((dense, sparse, ink))
        counts[(p, sense)] += 1
        return source, i, ink

    for r in records:
        source, i, sense = observe(r["predicate"], r["source"], r["sense"])
        target = raster(r["target"] + "。", render)
        model.observe(
            source,
            i,
            sense,
            raster(r["expression"], render).tobytes(),
            target,
            r["target_slice"],
            r["candidates"],
        )
    pool = defaultdict(set)
    for row in read_rows("train"):
        ss, ts = clauses(row["source"]), clauses(row["new"])
        if len(ss) != len(ts):
            continue
        for s, t in zip(ss, ts):
            for p, senses in (PREDICATES | OTHER).items():
                if (
                    s in used
                    or not 2 <= len(s) <= 32
                    or s.count(p) != 1
                    or t.count(p) != 1
                    or any(g in t for g in senses)
                ):
                    continue
                i = s.index(p)
                if any(
                    len(w) == 2 and w in t
                    for w in (s[max(0, i - 1) : i + 1], s[i : i + 2])
                ):
                    pool[p].add(s)
    for p, texts in pool.items():
        for s in sorted(texts, key=order)[:96]:
            observe(p, s, None)
    for k, rows in samples.items():
        engine.heads[k] = fit_head(rows)
    engine.save(OUT / "senses.msgpack.gz")
    model.save(OUT / "boundaries.msgpack.gz")
    write_new(
        EVIDENCE / "fit.json",
        dict(
            admission_sha=sha(EVIDENCE / "admission.json"),
            seconds=time.monotonic() - start,
            sense_training_rows=len(records),
            counts=[[p, s, n] for (p, s), n in counts.items()],
            encoder_sha=sha(PARENT),
            artifacts={
                p.name: dict(sha=sha(p), bytes=p.stat().st_size)
                for p in OUT.glob("*.msgpack.gz")
            },
        ),
    )


if __name__ == "__main__":
    fit()

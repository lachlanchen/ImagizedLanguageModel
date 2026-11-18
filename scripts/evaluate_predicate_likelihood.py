"""Freely decode the held-out contrasts, cue controls and diverse source images."""

from collections import Counter
import fcntl
import random
import time

import numpy as np
from rapidfuzz.distance import Levenshtein
import torch

from scripts.prepare_predicate_likelihood import (
    EVENT,
    OUT,
    ROOT,
    PREDICATES,
    Rasterizer,
    load,
    sha,
    write_new,
)
from scripts.evaluate_paired_student import predict, save_streams
from scripts.train_causal_glyph_translator import read_rows
from scripts.train_pictured_revision import composer
from scripts.train_visual_event_roles import PARENT, raster

EVAL = OUT / "evaluation"
# Source-only controls. The evaluation roster identifies which of the two
# pictured predicates is being tested even when another occurs incidentally.
CUES = {"去": "里远近步距世", "伐": "木林树竹材薪斧斤柳柏松桑枣桧梓"}


def cue_answer(draft, r, renderer, use_cues):
    p = r["predicate"]
    text = r["source"]
    left, right = text.split(p)
    found = use_cues and any(c in left + right for c in CUES[p])
    g = ("距离" if found else "离开") if p == "去" else ("砍伐" if found else "攻打")
    pieces = []
    if left:
        pieces.append(draft.generate(raster(left, renderer)))
    pieces.append(raster(g, renderer))
    pieces.append(draft.generate(raster(right + "。", renderer)))
    return np.concatenate(pieces)


def run():
    EVAL.mkdir()
    torch.set_num_threads(2)
    fit = load(OUT / "fit.json")
    seal = load(OUT / "admission.json")
    assert (
        fit["training"] == seal["train_sha"]
        and sha(OUT / "test.json") == seal["test_sha"]
    )
    roles = load(OUT / "test.json")
    broad = load(EVENT / "broad.json")
    rows = [
        dict(r, source=r["source"] + "。", target=r["target"] + "。") for r in roles
    ] + broad
    renderer = Rasterizer()
    start = time.monotonic()
    pictures = [raster(r["source"], renderer) for r in rows]
    refs = [raster(r["target"], renderer) for r in rows]
    k = len(roles)
    engine = composer()
    results = dict(
        source=pictures, reference=refs, composer=[engine.generate(x) for x in pictures]
    )
    ends = {}
    for name, path in [
        ("parent", PARENT),
        ("replay", OUT / "replay.safetensors"),
        ("context", OUT / "context.safetensors"),
    ]:
        assert sha(path) == (
            fit["parent"] if name == "parent" else fit["arms"][name]["sha256"]
        )
        results[name], done, _, _ = predict(path, pictures, pool_routes=True)
        ends[name] = dict(held=sum(done[:k]), broad=sum(done[k:]))
    isolated, _, _, _ = predict(
        OUT / "context.safetensors",
        [raster(p, renderer) for p in PREDICATES],
        pool_routes=True,
    )
    lookup = dict(zip(PREDICATES, isolated))
    results["isolated"] = [lookup[r["predicate"]] for r in roles]
    results["exchange"] = [results["context"][i ^ 1] for i in range(k)]
    for name, use in [("cue", True), ("base_rate", False)]:
        results[name] = [cue_answer(engine, r, renderer, use) for r in roles]
    for name, seq in results.items():
        save_streams(name, seq, EVAL)
    lex = {}
    chars = set("".join(r["source"] + r["target"] for r in rows))
    for r in read_rows("train"):
        chars.update(r["new"])
    for c in sorted(chars):
        try:
            lex.setdefault(renderer.tile(c), c)
        except ValueError:
            pass

    def read(x):
        return "".join(lex.get(bytes(tile), "▧") for tile in x)

    # Same twelve general passages read in the preceding matched study.
    old_n = 2 * len(load(EVENT / "roles.json"))
    broad_ids = [
        r["row"] - old_n
        for r in load(EVENT / "evaluation/mapping.json")
        if r["row"] >= old_n
    ]
    assert len(set(broad_ids)) == 12
    review_rows = list(range(k)) + [k + i for i in broad_ids]
    rng = random.Random(871033)
    rng.shuffle(review_rows)
    packet = []
    mapping = []
    unmapped = Counter()
    for i in review_rows:
        answers = {a: read(v[i]) for a, v in results.items() if i < len(v)}
        for a, text in answers.items():
            unmapped[a] += text.count("▧")
        choices = list(dict.fromkeys(answers.values()))
        rng.shuffle(choices)
        packet.append(
            dict(
                source=rows[i]["source"],
                focus=rows[i].get("predicate", ""),
                choices=choices,
            )
        )
        mapping.append(
            dict(row=i, options={a: choices.index(w) for a, w in answers.items()})
        )
    write_new(EVAL / "review.json", packet)
    write_new(EVAL / "mapping.json", mapping)
    summary = {}
    for a, outputs in results.items():
        hits = []
        for i, r in enumerate(roles):
            seq = outputs[i]
            pair = [bytes(z) for z in seq]
            grams = set(zip(pair, pair[1:]))
            (other,) = set(PREDICATES[r["predicate"]]) - {r["gloss"]}
            hits.append(
                tuple(map(renderer.tile, r["gloss"])) in grams
                and tuple(map(renderer.tile, other)) not in grams
            )
        rec = dict(
            literal_own_exclusive=sum(hits),
            both_literal=sum(x and y for x, y in zip(hits[::2], hits[1::2])),
            hits=hits,
        )
        if len(outputs) > k:
            rec["broad_edits"] = sum(
                Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
                for x, y in zip(outputs[k:], refs[k:])
            )
        summary[a] = rec
    write_new(
        EVAL / "summary.json",
        dict(
            scores=summary,
            broad_reference_cells=sum(map(len, refs[k:])),
            stop=ends,
            unknown_reading_cells=dict(unmapped),
            roster=seal["test_sha"],
            seconds=time.monotonic() - start,
            files={p.name: sha(p) for p in EVAL.glob("*.npz")},
        ),
    )
    print("Masked packet ready", len(packet), flush=True)


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

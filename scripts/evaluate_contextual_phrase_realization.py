"""Compare learned phrase choices with literal and explicit-cue controls on new clauses."""

import fcntl
import random
import time
from rapidfuzz.distance import Levenshtein
import torch

from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer
from scripts.evaluate_predicate_likelihood import EVAL as FIRST, cue_answer
from scripts.evaluate_paired_student import save_streams, read_streams, predict
from scripts.prepare_contextual_phrase_realization import (
    OUT,
    INITIAL,
    ROOT,
    EVENT,
    load,
    sha,
    write_new,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_paired_student import restore
from scripts.train_pictured_revision import composer
from scripts.train_causal_glyph_translator import read_rows
from scripts.train_visual_event_roles import PARENT, raster

EVAL = OUT / "evaluation"


def run():
    torch.set_num_threads(2)
    fit = load(OUT / "fit.json")
    assert fit["head_sha"] == sha(OUT / "realizer.msgpack.gz") and load(
        OUT / "admission.json"
    )["test_sha"] == sha(OUT / "test.json")
    EVAL.mkdir()
    start = time.monotonic()
    render = Rasterizer()
    draft = composer()
    learner = ContextualPhraseRealizer.restore(
        OUT / "realizer.msgpack.gz", restore(PARENT), draft
    )
    new = load(OUT / "test.json")
    dev = load(INITIAL / "test.json")
    broad = load(EVENT / "broad.json")
    a = len(new)
    b = a + len(dev)
    inputs = new + dev
    rows = [
        dict(r, source=r["source"] + "。", target=r["target"] + "。") for r in inputs
    ] + broad
    x = [raster(r["source"], render) for r in rows]
    out = {"source": x, "reference": [raster(r["target"], render) for r in rows]}
    out["learned"] = [learner.generate(v) for v in x]
    out["majority"] = [learner.generate(v, majority=True) for v in x]
    out["composer"] = [draft.generate(v) for v in x[:a]] + read_streams(
        "composer", FIRST
    )
    out["cue"] = [cue_answer(draft, r, render, True) for r in inputs]
    out["switched"] = [out["learned"][i ^ 1] for i in range(b)]
    out["neural"] = predict(INITIAL / "context.safetensors", x[:a], pool_routes=True)[
        0
    ] + read_streams("context", FIRST)
    for k, seq in out.items():
        save_streams(k, seq, EVAL)
    earlier = load(FIRST / "mapping.json")
    broad_ids = [r["row"] + a for r in earlier if r["row"] >= len(dev)]
    ids = list(range(b)) + broad_ids
    rng = random.Random(936201)
    rng.shuffle(ids)
    lex = {}
    chars = set("".join(r["source"] + r["target"] for r in rows))
    for r in read_rows("train"):
        chars.update(r["new"])
    for c in sorted(chars):
        try:
            lex.setdefault(render.tile(c), c)
        except ValueError:
            pass
    packet = []
    key = []
    unknown = 0
    for i in ids:
        opts = {
            k: "".join(lex.get(bytes(tile), "▧") for tile in seq[i])
            for k, seq in out.items()
            if i < len(seq)
        }
        unique = list(dict.fromkeys(opts.values()))
        rng.shuffle(unique)
        unknown += sum(w.count("▧") for w in unique)
        packet.append(
            dict(
                source=rows[i]["source"],
                focus=rows[i].get("predicate", ""),
                choices=unique,
            )
        )
        key.append(dict(row=i, options={k: unique.index(v) for k, v in opts.items()}))
    write_new(EVAL / "review.json", packet)
    write_new(EVAL / "mapping.json", key)
    score = {}
    for name, outputs in out.items():
        if len(outputs) > b:
            score[name] = sum(
                Levenshtein.distance(tuple(map(bytes, l)), tuple(map(bytes, r)))
                for l, r in zip(outputs[b:], out["reference"][b:], strict=True)
            )
    write_new(
        EVAL / "summary.json",
        dict(
            new=a,
            development=len(dev),
            broad=len(broad),
            unknown=unknown,
            broad_edits=score,
            files={p.name: sha(p) for p in EVAL.glob("*.npz")},
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

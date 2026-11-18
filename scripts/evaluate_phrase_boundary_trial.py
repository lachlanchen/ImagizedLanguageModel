"""Measure full independent pixel answers, with method identities concealed for reading."""

import random
import time

import numpy as np
from rapidfuzz.distance import Levenshtein
import torch

from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer
from ilm.visual_lm.visual_phrase_boundary import VisualPhraseBoundary
from scripts.evaluate_paired_student import save_streams
from scripts.evaluate_predicate_likelihood import cue_answer
from scripts.prepare_phrase_boundary_trial import (
    EVIDENCE,
    EVENT,
    OUT,
    PREVIOUS,
    Rasterizer,
    load,
    sha,
    write_new,
)
from scripts.train_visual_event_roles import PARENT, raster
from scripts.train_pictured_revision import composer
from scripts.train_paired_student import restore
from scripts.train_causal_glyph_translator import read_rows


def explicit(draft, row, render):
    p, s = row["predicate"], row["source"]
    if p in "去伐":
        return cue_answer(draft, row, render, True)
    left, right = s.split(p)
    if p == "谢":
        choice = (
            "道歉"
            if any(c in left + right for c in "罪过惶恐失愧负责免冠卑")
            else "感谢"
        )
    else:
        choice = "推荐" if any(c in left + right for c in "荐贤才善知任士") else "举起"
    parts = [draft.generate(raster(left, render))] if left else []
    return np.concatenate(
        parts + [raster(choice, render), draft.generate(raster(right + "。", render))]
    )


def evaluate():
    start = time.monotonic()
    torch.set_num_threads(2)
    admission = load(EVIDENCE / "admission.json")
    fit = load(EVIDENCE / "fit.json")
    assert (
        sha(EVIDENCE / "test.json") == admission["test_sha"]
        and sha(EVENT / "broad.json") == admission["broad_sha"]
    )
    for name, receipt in fit["artifacts"].items():
        assert sha(OUT / name) == receipt["sha"]
    encoder, draft = restore(PARENT), composer()
    matched = ContextualPhraseRealizer.restore(
        OUT / "senses.msgpack.gz", encoder, draft
    )
    joint = VisualPhraseBoundary.restore(OUT / "boundaries.msgpack.gz", matched)
    prior = ContextualPhraseRealizer.restore(
        PREVIOUS / "realizer.msgpack.gz", encoder, draft
    )
    test = load(EVIDENCE / "test.json")
    n = len(test)
    broad = load(EVENT / "broad.json")
    rows = [
        dict(r, source=r["source"] + "。", target=r["target"] + "。") for r in test
    ] + broad
    render = Rasterizer()
    inputs = [raster(r["source"], render) for r in rows]
    out = dict(source=inputs, reference=[raster(r["target"], render) for r in rows])
    for name, engine in [
        ("joint", joint),
        ("single", matched),
        ("earlier", prior),
        ("composer", draft),
    ]:
        out[name] = [engine.generate(x) for x in inputs]
    out["cue"] = [explicit(draft, r, render) for r in test]
    # The preassigned swapped source preserves the predicate and changes its sense.
    out["swapped"] = [out["joint"][i ^ 1] for i in range(20)]
    for k, stream in out.items():
        save_streams(k, stream, OUT)
    old_n = 2 * len(load(EVENT / "roles.json"))
    ids = [
        r["row"] - old_n
        for r in load(EVENT / "evaluation/mapping.json")
        if r["row"] >= old_n
    ]
    assert len(ids) == len(set(ids)) == 12
    selected = list(range(n)) + [n + i for i in ids]
    rng = random.Random(916607)
    rng.shuffle(selected)
    chars = set("".join(r["source"] + r["target"] for r in rows))
    for r in read_rows("train"):
        chars.update(r["new"] + r["source"])
    lex = {}
    for ch in sorted(chars):
        try:
            lex.setdefault(render.tile(ch), ch)
        except ValueError:
            continue
    packet, mapping, images, indices = [], [], [], []
    for j, i in enumerate(selected):
        streams = {k: v[i].tobytes() for k, v in out.items() if i < len(v)}
        choices = list(dict.fromkeys(streams.values()))
        rng.shuffle(choices)
        texts = []
        for k, v in enumerate(choices):
            cells = np.frombuffer(v, np.uint8).reshape(-1, 128)
            texts.append("".join(lex.get(bytes(cell), "▧") for cell in cells))
            images.append(cells)
            indices.append((j, k))
        packet.append(
            dict(source=rows[i]["source"], sense=rows[i].get("gloss"), choices=texts)
        )
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in streams.items()})
        )
    write_new(EVIDENCE / "review.json", packet)
    write_new(EVIDENCE / "mapping.json", mapping)
    with (EVIDENCE / "review_pixels.npz").open("xb") as f:
        np.savez_compressed(
            f,
            review_choice=indices,
            packed=np.concatenate(images),
            offsets=np.cumsum([0] + list(map(len, images))),
        )
    broad_edits = {
        k: sum(
            Levenshtein.distance(tuple(map(bytes, v)), tuple(map(bytes, t)))
            for v, t in zip(vs[n:], out["reference"][n:], strict=True)
        )
        for k, vs in out.items()
        if len(vs) == len(rows)
    }
    write_new(
        EVIDENCE / "evaluation.json",
        dict(
            broad_edits=broad_edits,
            broad=len(broad),
            seconds=time.monotonic() - start,
            unknown=sum(s.count("▧") for r in packet for s in r["choices"]),
            streams={p.name: sha(p) for p in OUT.glob("*.npz")},
            packet_sha=sha(EVIDENCE / "review.json"),
            pixels_sha=sha(EVIDENCE / "review_pixels.npz"),
        ),
    )


if __name__ == "__main__":
    evaluate()

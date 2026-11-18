"""Compare the local 8B text teacher on the registered student passages, after both fits."""

import argparse
import csv
import fcntl
import random
import numpy as np
from rapidfuzz.distance import Levenshtein
from scripts.train_screened_visual_targets import (
    ROOT,
    OUT,
    EVIDENCE,
    load,
    write_new,
    sha,
)
from scripts.prepare_source_only_teacher_pilot import collect
from scripts.evaluate_paired_student import read_streams, save_streams
from scripts.evaluate_screened_visual_targets import raster, Rasterizer


def draft():
    path = OUT / "benchmark_sources.json"
    identities = [
        dict(row=i, source=r["source"])
        for i, r in enumerate(load(EVIDENCE / "registration.json")["rows"])
    ]
    if not path.exists():
        write_new(path, identities)
    assert load(path) == identities
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(
            ROOT.parent / "LocalLLM",
            sample=path,
            out=OUT,
            output=EVIDENCE / "qwen_benchmark.json",
        )


def packet():
    reg = load(EVIDENCE / "registration.json")
    d = EVIDENCE / "reading"
    bench = load(EVIDENCE / "qwen_benchmark.json")
    assert sha(OUT / "benchmark_sources.json") == bench["training_sample_sha"]
    rows = reg["rows"]
    render = Rasterizer()
    drafts = bench["drafts"]
    assert [v["row"] for v in drafts] == list(range(len(rows)))
    predicted = [raster(v["draft"], render) for v in drafts]
    save_streams("qwen", predicted, OUT / "evaluation")
    gold = read_streams("reference", OUT / "evaluation")
    edits = [
        Levenshtein.distance(tuple(map(bytes, v)), tuple(map(bytes, y)))
        for v, y in zip(predicted, gold)
    ]
    old = load(d / "review.json")
    mapping = load(d / "mapping.json")
    pixels = read_streams("review_pixels", d)
    rng = random.Random(406207)
    pairs = []
    same = []
    offset = 0
    for j, (g, m) in enumerate(zip(old, mapping)):
        i = m["row"]
        v = predicted[i]
        choices = pixels[offset : offset + len(g["choices"])]
        offset += len(choices)
        idx = next((k for k, im in enumerate(choices) if np.array_equal(im, v)), None)
        if idx is not None:
            same.append(dict(row=i, j=j, k=idx))
            continue
        which = rng.choice(["composer", "reference"])
        alt = g["choices"][m["options"][which]]
        values = [drafts[i]["draft"], alt]
        k = rng.randrange(2)
        if k == 1:
            values.reverse()
        pairs.append((dict(row=i, k=k), dict(source=rows[i]["source"], choices=values)))
    rng.shuffle(pairs)
    write_new(d / "teacher_packet.json", [v[1] for v in pairs])
    write_new(d / "teacher_mapping.json", dict(new=[v[0] for v in pairs], same=same))
    write_new(
        EVIDENCE / "teacher_costs.json",
        dict(
            edits=edits,
            training=False,
            roster_sha=sha(EVIDENCE / "registration.json"),
            complete_generation=sum(r["done_reason"] == "stop" for r in drafts),
            raw_sha=sha(OUT / "evaluation/qwen.npz"),
        ),
    )


def score():
    reg = load(EVIDENCE / "registration.json")
    a = reg["archive"]
    d = EVIDENCE / "reading"
    m = load(d / "teacher_mapping.json")
    with (d / "credits.tsv").open() as f:
        prior = list(csv.DictReader(f, delimiter="\t"))
    with (d / "teacher_credits.tsv").open() as f:
        new = list(csv.DictReader(f, delimiter="\t"))
    assert len(new) == len(m["new"])
    passed = {}
    for name in ("detail", "whole"):
        ids = set()
        for j, (g, item) in enumerate(zip(new, m["new"])):
            assert int(g["j"]) == j
            good = {int(v) for v in g[name].split(",") if v != "-"}
            assert good <= {0, 1}
            if item["k"] in good:
                ids.add(item["row"])
        for item in m["same"]:
            if str(item["k"]) in prior[item["j"]][name].split(","):
                ids.add(item["row"])
        passed[name] = ids
    write_new(
        EVIDENCE / "teacher_scores.json",
        dict(
            archive_whole=sorted(i for i in passed["whole"] if i < a),
            contrast_whole=sorted(i for i in passed["whole"] if i >= a),
            detail_pairs=[
                p for p in range(12) if {a + 2 * p, a + 2 * p + 1} <= passed["detail"]
            ],
            whole_pairs=[
                p for p in range(12) if {a + 2 * p, a + 2 * p + 1} <= passed["whole"]
            ],
            previous_identical=len(m["same"]),
            new_readings=len(new),
            archive_edits=sum(load(EVIDENCE / "teacher_costs.json")["edits"][:a]),
            teacher_reading_sha=sha(d / "teacher_credits.tsv"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("draft", "packet", "score"))
    {"draft": draft, "packet": packet, "score": score}[p.parse_args().action]()

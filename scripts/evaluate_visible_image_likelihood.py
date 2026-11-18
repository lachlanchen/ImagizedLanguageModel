"""Compare single- and multi-path image likelihood on fresh pictured chapters."""

import fcntl
import random
from collections import defaultdict
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_paired_student import (
    fresh,
    predict,
    reading_packet,
    save_streams,
    source_exchange,
)
from scripts.train_visible_image_likelihood import (
    OUT,
    ROOT,
    PARENT,
    load,
    require,
    sha,
    tiles,
    write_new,
)
from scripts.train_pictured_revision import composer

EVAL = OUT / "evaluation"


def run():
    torch.set_num_threads(2)
    raw = tiles()
    rows, prior = fresh(
        raw,
        earlier=[
            ("source_grounded_revision_v1", "evaluation", "source"),
            ("paired_gloss_student_v1", "evaluation", "source", "roster.json"),
        ],
        seed="visible-likelihood-01",
    )
    EVAL.mkdir()
    sources = [raw[r["x"]] for r in rows]
    targets = [raw[r["y"]] for r in rows]
    outputs = {"source": sources, "reference": targets}
    draft = composer()
    outputs["composer"] = [draft.generate(x) for x in sources]
    fit = load(OUT / "fit.json")
    stops = {}
    routes = {}
    seconds = {}
    for name in ("parent", "single", "marginal"):
        p = PARENT if name == "parent" else OUT / (name + ".safetensors")
        require(
            sha(p)
            == (fit["parent"] if name == "parent" else fit["arms"][name]["sha256"]),
            "Changed model",
        )
        outputs[name], stops[name], routes[name], seconds[name] = predict(
            p, sources, pool_routes=True
        )
    switch = source_exchange(sources, rows)
    outputs["exchange"] = [outputs["marginal"][j] for j in switch]
    for k, v in outputs.items():
        save_streams(k, v, EVAL)
    masked = []
    mapping = []
    rng = random.Random(79213)
    for entry in reading_packet(raw, prior, rows, outputs):
        i = entry["row"]
        opts = entry["options"]
        choices = list(dict.fromkeys(opts.values()))
        rng.shuffle(choices)
        masked.append(dict(source=rows[i]["source"], choices=choices))
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in opts.items()})
        )
    write_new(EVAL / "review.json", masked)
    write_new(EVAL / "mapping.json", mapping)
    distances = {}
    scores = {}
    for k, v in outputs.items():
        d = np.array(
            [
                Levenshtein.distance(tuple(map(bytes, p)), tuple(map(bytes, t)))
                for p, t in zip(v, targets)
            ]
        )
        exact = d == 0
        distances[k] = d
        scores[k] = dict(
            edits=int(d.sum()),
            exact=int(exact.sum()),
            changed_exact=sum(
                1
                for a, s, t in zip(exact, sources, targets)
                if a and not np.array_equal(s, t)
            ),
            cells=sum(map(len, v)),
            empty=sum(not len(x) for x in v),
            stopped=sum(stops[k]) if k in stops else None,
        )
    grouped = defaultdict(lambda: np.zeros(3))
    for i, r in enumerate(rows):
        grouped[r["g"]] += (
            distances["marginal"][i] - distances["single"][i],
            distances["marginal"][i] - distances["exchange"][i],
            len(targets[i]),
        )
    g = np.stack(list(grouped.values()))
    rng = np.random.default_rng(817)
    b = g[rng.integers(len(g), size=(2000, len(g)))].sum(axis=1)
    write_new(
        EVAL / "roster.json",
        [
            dict(
                chapter=r["g"],
                ordinal=r["i"],
                source=r["source"],
                target=r["target"],
                switched_from=switch[i],
            )
            for i, r in enumerate(rows)
        ],
    )
    write_new(EVAL / "distances.json", {k: v.tolist() for k, v in distances.items()})
    write_new(
        EVAL / "scores.json",
        dict(
            chapters=len(g),
            rows=len(rows),
            reference_cells=sum(map(len, targets)),
            scores=scores,
            ci=np.quantile(b[:, :2] / b[:, 2:], [0.025, 0.975], axis=0).T.tolist(),
            native_seconds=seconds,
            routes=routes,
            files={p.name: sha(p) for p in EVAL.iterdir()},
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

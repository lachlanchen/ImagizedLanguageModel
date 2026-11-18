"""Compare complete free-writing and participant reversals on unseen pictured passages."""

from collections import defaultdict
import fcntl
import random

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from scripts.evaluate_paired_student import predict, source_exchange, save_streams
from scripts.measure_pictured_clause_support import classify, MARKS
from scripts.prepare_visual_event_roles import (
    OUT,
    ROOT,
    RESERVED,
    swap,
    load,
    sha,
    write_new,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_pictured_revision import composer
from scripts.train_visual_event_roles import PARENT, raster
from scripts.train_causal_glyph_translator import read_rows

EVAL = OUT / "evaluation"
INVALID = {3: "partial place name", 33: "personal names"}
ARMS = ("source", "parent", "replay", "swapped", "composer", "exchange", "reference")


def role_positions(pixels, first, second):
    values = list(map(bytes, pixels))
    return values.count(first) == values.count(second) == 1 and values.index(
        first
    ) < values.index(second)


def select_pairs(rows):
    ids = []
    chapters = set()
    verbs = set()

    def add(i, r):
        if i not in INVALID and i not in ids and r["chapter"] not in chapters:
            ids.append(i)
            chapters.add(r["chapter"])
            verbs.add(r["verb"])

    for i, r in enumerate(rows):
        if r["verb"] in RESERVED:
            add(i, r)
    for diverse in [True, False]:
        for i, r in enumerate(rows):
            if len(ids) == 12:
                break
            if not diverse or r["verb"] not in verbs:
                add(i, r)
    assert len(ids) == 12
    return ids


def run():
    torch.set_num_threads(2)
    EVAL.mkdir()
    roles = load(OUT / "roles.json")
    broad = load(OUT / "broad.json")
    prep = load(OUT / "preparation.json")
    fit = load(OUT / "fit.json")
    render = Rasterizer()
    for n in ("roles", "broad"):
        assert sha(OUT / (n + ".json")) == prep["splits"][n]["sha256"]
    assert roles[3]["source"] == "周师攻楚州。" and "冯唐救魏尚" in roles[33]["source"]
    rows = []
    for i, r in enumerate(roles):
        for reverse in (False, True):
            a, b = r["a"], r["b"]
            rows.append(
                dict(
                    r,
                    base=i,
                    reverse=reverse,
                    source=swap(r["source"], a, b) if reverse else r["source"],
                    target=swap(r["target"], a, b) if reverse else r["target"],
                    first=b if reverse else a,
                    second=a if reverse else b,
                )
            )
    cut = len(rows)
    rows += broad
    sources = [raster(r["source"], render) for r in rows]
    outputs = {
        "source": sources,
        "reference": [raster(r["target"], render) for r in rows],
    }
    draft = composer()
    outputs["composer"] = [draft.generate(x) for x in sources]
    ends = {}
    timings = {}
    for k in ("parent", "replay", "swapped"):
        path = PARENT if k == "parent" else OUT / (k + ".safetensors")
        assert sha(path) == (
            fit["parent"] if k == "parent" else fit["arms"][k]["sha256"]
        )
        outputs[k], ends[k], _, timings[k] = predict(path, sources, pool_routes=True)
    perm = source_exchange(sources[cut:], [{"g": r["chapter"]} for r in broad])
    outputs["exchange"] = [outputs["swapped"][i ^ 1] for i in range(cut)] + [
        outputs["swapped"][cut + j] for j in perm
    ]
    for k, stream in outputs.items():
        save_streams(k, stream, EVAL)
    ids = select_pairs(roles)
    reading = [2 * i + b for i in ids for b in (0, 1)]
    used = set()
    for topic, _ in MARKS:
        selected = []
        for j, r in enumerate(broad):
            if r["chapter"] not in used and classify(r["source"]) == topic:
                used.add(r["chapter"])
                selected.append(j + cut)
                if len(selected) == 4:
                    break
        assert len(selected) == 4
        reading += selected
    rng = random.Random(8924)
    rng.shuffle(reading)
    text_bank = set("".join(r["source"] + r["target"] for r in rows))
    for r in read_rows("train"):
        text_bank.update(r["new"])
    lex = {}
    for c in sorted(text_bank):
        try:
            lex.setdefault(render.tile(c), c)
        except ValueError:
            pass
    packets = []
    mapping = []
    unknown = 0
    for i in reading:
        opts = {}
        for k in ARMS:
            words = []
            for cell in outputs[k][i]:
                c = lex.get(cell.tobytes())
                if c is not None:
                    assert render.tile(c) == cell.tobytes()
                else:
                    unknown += 1
                words.append(c or "▧")
            opts[k] = "".join(words)
        choices = list(dict.fromkeys(opts.values()))
        rng.shuffle(choices)
        packets.append(
            dict(
                kind="role" if i < cut else "broad",
                source=rows[i]["source"],
                choices=choices,
            )
        )
        mapping.append(
            dict(row=i, options={n: choices.index(v) for n, v in opts.items()})
        )
    write_new(EVAL / "review.json", packets)
    write_new(EVAL / "mapping.json", mapping)
    write_new(EVAL / "rows.json", rows)
    distances = {}
    stats = {}
    role_stats = {}
    for name in ARMS:
        sequence = outputs[name]
        delta = np.array(
            [
                Levenshtein.distance(tuple(map(bytes, v)), tuple(map(bytes, t)))
                for v, t in zip(sequence, outputs["reference"])
            ]
        )
        distances[name] = delta.tolist()
        exact = delta[cut:] == 0
        stats[name] = dict(
            edits=int(delta[cut:].sum()),
            exact=int(exact.sum()),
            changed_exact=sum(
                1
                for k, ok in enumerate(exact, cut)
                if ok and not np.array_equal(sources[k], outputs["reference"][k])
            ),
            cells=sum(map(len, sequence[cut:])),
            stops=sum(ends[name][cut:]) if name in ends else None,
        )
        scores = []
        for i, r in enumerate(roles):
            vals = [
                role_positions(
                    sequence[j],
                    render.tile(rows[j]["first"]),
                    render.tile(rows[j]["second"]),
                )
                for j in (2 * i, 2 * i + 1)
            ]
            scores.append(
                dict(
                    ordered=vals,
                    both=all(vals),
                    both_changed=all(vals)
                    and all(
                        not np.array_equal(sequence[j], sources[j])
                        for j in (2 * i, 2 * i + 1)
                    ),
                    both_exact=bool((delta[2 * i : 2 * i + 2] == 0).all()),
                )
            )
        role_stats[name] = {}
        for part, idx in [
            ("all", list(range(len(roles)))),
            ("primary", [j for j in range(len(roles)) if j not in INVALID]),
            (
                "reserved",
                [
                    j
                    for j, r in enumerate(roles)
                    if j not in INVALID and r["verb"] in RESERVED
                ],
            ),
        ]:
            role_stats[name][part] = {
                "pairs": len(idx),
                **{
                    k: sum(scores[j][k] for j in idx)
                    for k in ("both", "both_changed", "both_exact")
                },
                "ordered": sum(sum(scores[j]["ordered"]) for j in idx),
            }
    grouped = defaultdict(lambda: np.zeros(3))
    for j, r in enumerate(broad, cut):
        grouped[r["chapter"]] += (
            distances["swapped"][j] - distances["replay"][j],
            distances["swapped"][j] - distances["composer"][j],
            len(outputs["reference"][j]),
        )
    g = np.stack(list(grouped.values()))
    br = np.random.default_rng(61429)
    rep = g[br.integers(len(g), size=(2000, len(g)))].sum(axis=1)
    write_new(EVAL / "distances.json", distances)
    write_new(
        EVAL / "summary.json",
        dict(
            roles=role_stats,
            broad=stats,
            invalid=INVALID,
            review_pairs=ids,
            unknown_reading_cells=unknown,
            broad_reference_cells=sum(map(len, outputs["reference"][cut:])),
            ci95=np.quantile(
                rep[:, :2] / rep[:, 2:], [0.025, 0.975], axis=0
            ).T.tolist(),
            seconds=timings,
            files={p.name: sha(p) for p in EVAL.iterdir()},
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

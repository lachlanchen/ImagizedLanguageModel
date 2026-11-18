"""Run source-only students first, then obtain independent teacher targets for scoring."""

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
import random
import time

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from scripts.align_visual_translation import load, require, sha, tiles, write_new
from scripts.compare_pictured_teacher_inputs import MODEL
from scripts.evaluate_pictured_revision import fresh
from scripts.local_translation_teacher import SYSTEM, local_teacher
from scripts.measure_pictured_clause_support import classify, MARKS
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_paired_student import OUT, ROOT, PARENT, restore
from scripts.train_pictured_revision import composer

EVAL = OUT / "evaluation"


def save_streams(name, streams, directory=EVAL):
    path = directory / (name + ".npz")
    with path.open("xb") as f:
        np.savez_compressed(
            f,
            packed=np.concatenate(streams),
            offsets=np.cumsum([0] + list(map(len, streams))),
        )


def read_streams(name, directory=EVAL):
    with np.load(directory / (name + ".npz")) as f:
        data = f["packed"]
        off = f["offsets"]
        return [data[a:b] for a, b in zip(off, off[1:])]


@torch.inference_mode()
def predict(path, sources, *, pool_routes=False):
    model = restore(path).cuda().eval()
    result = []
    ends = []
    counts = Counter()
    start = time.monotonic()
    for j in range(0, len(sources), 32):
        group = sources[j : j + 32]
        lens = torch.tensor(list(map(len, group)), device="cuda")
        packed = np.zeros((len(group), int(lens.max()), 128), np.uint8)
        for i, s in enumerate(group):
            packed[i, : len(s)] = s
        pics = torch.from_numpy(np.unpackbits(packed, axis=2)).cuda().bool()
        images, sizes, done, actions = model.generate(
            pics, lens, pool_routes=pool_routes
        )
        for image, n, k in zip(images.cpu().numpy(), sizes.tolist(), actions.tolist()):
            result.append(np.packbits(image[:n], axis=1))
            counts.update(k[:n])
        ends.extend(done.tolist())
    return result, ends, counts, time.monotonic() - start


def source_exchange(sources, rows):
    group = defaultdict(list)
    for i, x in enumerate(sources):
        group[len(x)].append(i)
    switch = [None] * len(rows)
    for subset in group.values():
        # A cyclic permutation preserves the complete output multiset.
        offset = next(
            t
            for t in range(1, len(subset))
            if all(
                rows[v]["g"] != rows[subset[(j + t) % len(subset)]]["g"]
                for j, v in enumerate(subset)
            )
        )
        for j, i in enumerate(subset):
            switch[i] = subset[(j + offset) % len(subset)]
    require(sorted(switch) == list(range(len(rows))), "Exchange is not bijective")
    return switch


def reading_packet(raw, prior, rows, outputs):
    seen = set()
    ids = []
    for topic, _ in MARKS:
        part = []
        for i, r in enumerate(rows):
            if r["g"] not in seen and classify(r["source"]) == topic:
                seen.add(r["g"])
                part.append(i)
                if len(part) == 6:
                    break
        require(len(part) == 6, "Empty reading stratum")
        ids += part
    lex = defaultdict(Counter)
    for r in prior:
        for k, c in zip(r["z"], r["new"]):
            lex[bytes(raw[k])][c] += 1
    for r in rows:
        for field, col in [("source", "x"), ("target", "y")]:
            for k, c in zip(r[col], r[field]):
                lex[bytes(raw[k])][c] += 1
    preview = []
    for i in sorted(ids):
        preview.append(
            dict(
                row=i,
                options={
                    n: "".join(
                        lex[cell.tobytes()].most_common(1)[0][0]
                        if cell.tobytes() in lex
                        else "▧"
                        for cell in v[i]
                    )
                    for n, v in outputs.items()
                    if n != "source"
                },
            )
        )
    return preview


@torch.inference_mode()
def generate():
    EVAL.mkdir()
    began = time.monotonic()
    fit = load(OUT / "fit.json")
    raw = tiles()
    rows, prior = fresh(
        raw,
        earlier=[("source_grounded_revision_v1", "evaluation", "source")],
        seed="pairedstudent1",
    )
    sources = [raw[r["x"]] for r in rows]
    outputs = {"source": sources, "reference": [raw[r["y"]] for r in rows]}
    draft = composer()
    outputs["composer"] = [draft.generate(x) for x in sources]
    stops = {}
    routes = {}
    native_seconds = {}
    for name in ["parent", "archive", "teacher"]:
        path = PARENT if name == "parent" else OUT / (name + ".safetensors")
        require(
            sha(path)
            == (fit["parent"] if name == "parent" else fit["arms"][name]["checkpoint"]),
            "Changed weights",
        )
        outputs[name], stops[name], routes[name], native_seconds[name] = predict(
            path, sources
        )
    switch = source_exchange(sources, rows)
    outputs["exchange"] = [outputs["teacher"][j] for j in switch]
    for name, v in outputs.items():
        save_streams(name, v)
    preview = reading_packet(raw, prior, rows, outputs)
    write_new(
        EVAL / "roster.json",
        [
            dict(
                chapter=r["g"],
                ordinal=r["i"],
                source=r["source"],
                target=r["target"],
                switched_from=switch[j],
            )
            for j, r in enumerate(rows)
        ],
    )
    write_new(EVAL / "reading_inputs.json", preview)
    write_new(
        EVAL / "generation.json",
        dict(
            chapters=len({r["g"] for r in rows}),
            rows=len(rows),
            routes=routes,
            stops=stops,
            native_seconds=native_seconds,
            seconds=time.monotonic() - began,
            files={p.name: sha(p) for p in EVAL.iterdir()},
        ),
    )


def second_targets():
    roster = load(EVAL / "roster.json")
    gen = load(EVAL / "generation.json")
    for n, h in gen["files"].items():
        require(sha(EVAL / n) == h, "Independent source generation changed")
    work = OUT / "target_teacher"
    work.mkdir(exist_ok=True)
    if (work / "receipt.json").exists():
        raise FileExistsError("Teacher generation finished")
    path = work / "outputs.jsonl"
    previous = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    bound = load(ROOT / "references/evidence/paired_teacher_glosses_20261119.json")[
        "preparation"
    ]["manifest"]
    with local_teacher(MODEL, work) as (ask, digest, size), path.open("a") as f:
        require(bound == digest, "Teacher differs from training")
        for i, r in enumerate(previous):
            require(
                r["source"] == roster[i]["source"] and r["manifest"] == digest,
                "Wrong partial roster",
            )
        for i, row in enumerate(roster[len(previous) :], len(previous)):
            record = ask(
                SYSTEM,
                dict(role="user", content=row["source"]),
                options=dict(seed=331, temperature=0, num_ctx=4096, num_predict=160),
            )
            record.update(row=i, source=row["source"], manifest=digest)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            previous.append(record)
            if len(previous) % 128 == 0:
                print(len(previous), flush=True)
    renderer = Rasterizer()
    pictures = []
    usable = []
    for r in previous:
        try:
            pix = np.frombuffer(
                b"".join(renderer.tile(c) for c in r["text"]), np.uint8
            ).reshape(-1, 128)
        except ValueError:
            pix = np.zeros((0, 128), np.uint8)
        pictures.append(pix)
        usable.append(bool(len(pix)) and r["stop"] == "stop")
    save_streams("teacher_target", pictures)
    rng = random.Random(50242)
    blind = []
    mapping = []
    for r in load(EVAL / "reading_inputs.json"):
        i = r["row"]
        opts = dict(r["options"], teacher_target=previous[i]["text"])
        choices = list(dict.fromkeys(opts.values()))
        rng.shuffle(choices)
        blind.append(dict(source=roster[i]["source"], choices=choices))
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in opts.items()})
        )
    write_new(EVAL / "review.json", blind)
    write_new(EVAL / "mapping.json", mapping)
    write_new(
        work / "receipt.json",
        dict(
            rows=len(previous),
            manifest=digest,
            bytes=size,
            seconds=sum(r["duration_ns"] for r in previous) / 1e9,
            complete_renderable=usable,
            truncated=sum(r["stop"] == "length" for r in previous),
            outputs=sha(path),
            packed=sha(EVAL / "teacher_target.npz"),
            review=sha(EVAL / "review.json"),
        ),
    )


def score():
    gen = load(EVAL / "generation.json")
    roster = load(EVAL / "roster.json")
    rc = load(OUT / "target_teacher/receipt.json")
    streams = {
        k: [tuple(map(bytes, row)) for row in read_streams(k)]
        for k in (
            "source",
            "reference",
            "composer",
            "parent",
            "archive",
            "teacher",
            "exchange",
            "teacher_target",
        )
    }
    assert all(len(stream) == len(roster) for stream in streams.values())
    scores = {}
    edits = {}
    detailed = {}
    for ref, eligible in [
        ("reference", np.ones(len(roster), bool)),
        ("teacher_target", np.asarray(rc["complete_renderable"])),
    ]:
        results = {}
        dist = {}
        denom = sum(len(x) for x, v in zip(streams[ref], eligible) if v)
        for name, values in streams.items():
            delta = np.array(
                [Levenshtein.distance(a, b) for a, b in zip(values, streams[ref])]
            )
            dist[name] = delta
            changed = np.array(
                [a != b for a, b in zip(streams["source"], streams[ref])]
            )
            results[name] = dict(
                edits=int(delta[eligible].sum()),
                exact=int(((delta == 0) & eligible).sum()),
                changed_exact=int(((delta == 0) & eligible & changed).sum()),
                cells=sum(len(x) for x, v in zip(values, eligible) if v),
            )
        groups = defaultdict(lambda: np.zeros(3))
        for i, r in enumerate(roster):
            if eligible[i]:
                groups[r["chapter"]] += (
                    dist["teacher"][i] - dist["archive"][i],
                    dist["teacher"][i] - dist["exchange"][i],
                    len(streams[ref][i]),
                )
        grouped = np.stack(list(groups.values()))
        rng = np.random.default_rng(913)
        b = grouped[rng.integers(len(grouped), size=(2000, len(grouped)))].sum(axis=1)
        scores[ref] = dict(
            rows=int(eligible.sum()),
            denominator=denom,
            arms=results,
            ci95=np.quantile(b[:, :2] / b[:, 2:], [0.025, 0.975], axis=0).T.tolist(),
        )
        detailed[ref] = {n: d.tolist() for n, d in dist.items()}
    write_new(EVAL / "distances.json", detailed)
    write_new(
        EVAL / "scores.json",
        dict(
            scores=scores,
            stops={k: sum(v) for k, v in gen["stops"].items()},
            routes=gen["routes"],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["generate", "targets", "score"])
    a = p.parse_args().action
    torch.set_num_threads(2)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"generate": generate, "targets": second_targets, "score": score}[a]()

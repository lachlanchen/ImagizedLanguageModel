"""Evaluate conditioned phrase choice and independent answers on the registered sources."""

import argparse
import fcntl
from time import perf_counter

import numpy as np
from rapidfuzz.distance import Levenshtein
import torch

from scripts.evaluate_anchor_gap_writer import reading
from scripts.evaluate_screened_visual_targets import (
    generate,
    save_streams,
    composer,
    raster,
)
from scripts.reserve_context_phrase_trial import EVIDENCE, ROOT, WORK, load, write_new
from scripts.prepare_source_only_teacher_pilot import collect
from scripts.train_context_phrase_trial import (
    ARMS,
    MODELS,
    START,
    DATA,
    Rasterizer,
    binding,
    pack,
    four_scores,
    restore,
    sha,
)
from ilm.visual_lm.idiom_ink import visual_input


def sources():
    archive = load(EVIDENCE / "holdout.json")["rows"]
    paired = load(EVIDENCE / "test_pairs.json")
    glosses = load(EVIDENCE / "authored_glosses.json")
    rows = archive.copy()
    indices = []
    for pair in paired:
        if pair["kind"] == "archive":
            indices.append(pair["rows"])
            continue
        local = []
        for s in pair["source"]:
            local.append(len(rows))
            rows.append(dict(source=s, target=glosses[s]))
        indices.append(local)
    return rows, indices


def teacher():
    rows, _ = sources()
    p = WORK / "teacher_queries.json"
    queries = [dict(row=i, source=r["source"]) for i, r in enumerate(rows)]
    if not p.exists():
        write_new(p, queries)
    if load(p) != queries:
        raise ValueError("Wrong teacher queries")
    collect(
        ROOT.parent / "LocalLLM",
        sample=p,
        out=WORK,
        output=EVIDENCE / "teacher_answers.json",
    )


@torch.inference_mode()
def evaluate():
    reg = binding()
    assert load(EVIDENCE / "registration.json") == reg
    rows, idx = sources()
    pairs = load(EVIDENCE / "test_pairs.json")
    train = load(EVIDENCE / "training_pairs.json")
    render = Rasterizer()
    streams = {k: [raster(r[k], render) for r in rows] for k in ("source", "target")}
    start = perf_counter()
    engine = composer()
    streams["composer"] = [engine.generate(x) for x in streams["source"]]
    stats = {"composer": dict(seconds=perf_counter() - start)}
    q = load(EVIDENCE / "teacher_answers.json")
    assert q["training_sample_sha"] == sha(WORK / "teacher_queries.json") and [
        r["row"] for r in q["drafts"]
    ] == list(range(len(rows)))
    streams["teacher"] = [raster(r["draft"], render) for r in q["drafts"]]
    stats["teacher"] = dict(
        seconds=q["seconds"], stops=[r["done_reason"] == "stop" for r in q["drafts"]]
    )
    with np.load(DATA / "training.npz") as z:
        raw, records = pack(train + pairs, z["cells"])
    ids = {bytes(v): i for i, v in enumerate(raw)}
    ranking = {}

    def batch_scores(m, features, bank, records):
        return (
            torch.cat(
                [
                    four_scores(m, features, bank, records[i : i + 12])
                    for i in range(0, len(records), 12)
                ]
            )
            .cpu()
            .tolist()
        )

    replay = EVIDENCE / "training_replay"
    replay.mkdir()
    training_inputs = [raster(x, render) for rec in train for x in rec["source"]]
    save_streams("source", training_inputs, replay)
    for arm in ("parent", *ARMS):
        path = START if arm == "parent" else MODELS / (arm + ".safetensors")
        assert sha(path) == (
            reg["start_sha"]
            if arm == "parent"
            else load(EVIDENCE / (arm + "_fit.json"))["sha"]
        )
        model = restore(path, "cuda").eval()
        features = visual_input(
            torch.from_numpy(np.unpackbits(raw, axis=1)).cuda().bool(), model.landmarks
        )
        exemplars = {
            ids[bytes(v)] for v in np.packbits(model.exemplars.cpu().numpy(), axis=1)
        }
        vocab = lambda rs: sorted(
            exemplars | {g for _, ys, _ in rs for y in ys for g in y}
        )
        ranking[arm] = {}
        for group, rs, allowed in (
            ("train", records[: len(train)], records[: len(train)]),
            ("test", records[len(train) :], records),
        ):
            bank = torch.tensor(vocab(allowed), device="cuda")
            ranking[arm][group] = batch_scores(model, features, bank, rs)
        # A scalar recomputation catches padding, candidate order and normalization errors.
        scalar = four_scores(model, features, bank, [records[-1]])[0].cpu()
        torch.testing.assert_close(
            scalar, torch.tensor(ranking[arm]["test"][-1]), atol=3e-5, rtol=1e-5
        )
        del model
        start = perf_counter()
        streams[arm], stops = generate(path, streams["source"])
        stats[arm] = dict(
            seconds=perf_counter() - start, stops=stops, checkpoint_sha=sha(path)
        )

        training_outputs, _ = generate(path, training_inputs)
        save_streams(arm, training_outputs, replay)
        reproduced = []
        for i, rec in enumerate(train):
            hits = []
            for side, phrase in enumerate(rec["phrase"]):
                expected = raster(rec["prefix"] + phrase, render)
                hits.append(
                    training_outputs[2 * i + side][: len(expected)].tobytes()
                    == expected.tobytes()
                )
            reproduced.append(hits)
        stats[arm]["free_training_starts"] = reproduced
        stats[arm]["training_replay_sha"] = sha(replay / (arm + ".npz"))

    def overlap(x, y):
        bank = set(x)
        return sum(v in bank for v in y) / len(y)

    ranking["ink_overlap"] = {
        k: [
            [
                [
                    overlap(set(map(render.tile, x)), list(map(render.tile, y)))
                    for y in r["phrase"]
                ]
                for x in r["source"]
            ]
            for r in rs
        ]
        for k, rs in (("train", train), ("test", pairs))
    }
    folder = EVIDENCE / "streams"
    folder.mkdir()
    for name, ss in streams.items():
        save_streams(name, ss, folder)
        stats.setdefault(name, {}).update(
            stream_sha=sha(folder / (name + ".npz")),
            lengths=list(map(len, ss)),
            edits=[
                Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
                for x, y in zip(ss, streams["target"])
            ],
        )
    write_new(
        EVIDENCE / "evaluation.json",
        dict(
            phrase_scores=ranking,
            methods=stats,
            pair_rows=idx,
            registration_sha=sha(EVIDENCE / "registration.json"),
            gloss_sha=sha(EVIDENCE / "authored_glosses.json"),
        ),
    )
    selected = list(
        dict.fromkeys(load(EVIDENCE / "reading_roster.json") + sum(idx, []))
    )
    reading(
        rows,
        selected,
        streams,
        evidence=EVIDENCE / "reading",
        extra_chars="".join(r["draft"] for r in q["drafts"]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("teacher", "evaluate"))
    a = p.parse_args()
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        (teacher if a.action == "teacher" else evaluate)()

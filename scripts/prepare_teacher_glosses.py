"""Prepare complete teacher and archive target pictures paired to actual training inputs."""

import fcntl
import hashlib
import json
import os
import time

import numpy as np
from scripts.local_translation_teacher import local_teacher, SYSTEM
from scripts.train_pictured_revision import ROOT, read_rows, tiles, sha, write_new
from scripts.compare_pictured_teacher_inputs import MODEL
from scripts.prepare_visual_sequence_corpus import Rasterizer


def run():
    out = ROOT / "artifacts/paired_teacher_glosses_v1"
    out.mkdir(exist_ok=True)
    if (out / "preparation.json").exists():
        raise FileExistsError("The preparation has finished")
    data = read_rows("train")
    ids = sorted(
        range(len(data)),
        key=lambda i: hashlib.sha256(
            f"pairedteacher1/{data[i]['group']}/{data[i]['ordinal']}".encode()
        ).digest(),
    )
    roster = []
    chapters = set()
    for i in ids:
        r = data[i]
        if r["group"] in chapters or not (
            5 <= len(r["x"]) <= 24 and 3 <= len(r["z"]) <= 48
        ):
            continue
        roster.append(i)
        chapters.add(r["group"])
        if len(roster) == 512:
            break
    assert len(chapters) == 512
    path = out / "responses.jsonl"
    existing = (
        [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    )
    for i, r in enumerate(existing):
        assert r["train_row"] == roster[i] and r["source"] == data[roster[i]]["source"]
    archive = tiles()
    renderer = Rasterizer()
    start = time.monotonic()
    with local_teacher(MODEL, out) as (ask, manifest, size), path.open("a") as stream:
        for r in existing:
            assert r["manifest"] == manifest and r["prompt"] == SYSTEM
        for j, i in enumerate(roster[len(existing) :], len(existing)):
            row = data[i]
            answer = ask(
                SYSTEM,
                dict(role="user", content=row["source"]),
                options=dict(seed=331, temperature=0, num_ctx=4096, num_predict=160),
            )
            record = dict(
                train_row=i,
                chapter=row["group"],
                ordinal=row["ordinal"],
                source=row["source"],
                target=row["new"],
                prompt=SYSTEM,
                manifest=manifest,
                **answer,
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            existing.append(record)
            if (j + 1) % 64 == 0:
                print("Prepared", j + 1, flush=True)
    buffers = {k: [] for k in ("source", "archive", "teacher")}
    compatible = []
    for r in existing:
        row = data[r["train_row"]]
        buffers["source"].append(archive[row["x"]])
        buffers["archive"].append(archive[row["z"]])
        try:
            pixels = np.frombuffer(
                b"".join(renderer.tile(c) for c in r["text"]), np.uint8
            ).reshape(-1, 128)
            can = bool(len(pixels))
        except ValueError:
            pixels = np.zeros((0, 128), np.uint8)
            can = False
        buffers["teacher"].append(pixels)
        compatible.append(can and r["stop"] == "stop")
    arrays = {"train_rows": np.asarray(roster), "usable": np.array(compatible)}
    for k, v in buffers.items():
        arrays[k] = np.concatenate(v)
        arrays[k + "_offsets"] = np.cumsum([0] + list(map(len, v)))
    np.savez_compressed(out / "images.npz", **arrays)
    write_new(
        out / "preparation.json",
        dict(
            rows=len(existing),
            chapters=len(chapters),
            usable=sum(compatible),
            teacher=MODEL,
            manifest=manifest,
            model_bytes=size,
            original_targets=len(roster),
            teacher_cells=len(arrays["teacher"]),
            model_runtime_seconds=sum(r["duration_ns"] for r in existing) / 1e9,
            last_run_seconds=time.monotonic() - start,
            response_sha256=sha(path),
            image_sha256=sha(out / "images.npz"),
            source="artifacts/aligned_visual_translation_v1/train.jsonl",
            source_sha256=sha(
                ROOT / "artifacts/aligned_visual_translation_v1/train.jsonl"
            ),
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

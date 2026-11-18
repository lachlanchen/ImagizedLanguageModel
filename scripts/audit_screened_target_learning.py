"""Measure free writing on the 113 reviewed training sources against both sets of targets."""

import fcntl
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_screened_visual_targets import (
    generate,
    raster,
    Rasterizer,
    save_streams,
)
from scripts.train_screened_visual_targets import (
    ROOT,
    OUT,
    EVIDENCE,
    PARENT,
    load,
    write_new,
    sha,
)
from scripts.train_aligned_visual_translation import read_rows
import torch


def measure():
    rows = load(EVIDENCE / "accepted.json")
    book = read_rows("train")
    render = Rasterizer()
    streams = {
        key: [raster(text, render) for text in texts]
        for key, texts in (
            ("source", [r["source"] for r in rows]),
            ("screened_target", [r["target"] for r in rows]),
            ("book_target", [book[r["row"]]["new"] for r in rows]),
        )
    }
    scores = {}
    folder = OUT / "training_replay"
    folder.mkdir()
    for name, path in (
        ("parent", PARENT),
        ("book", OUT / "book.safetensors"),
        ("screened", OUT / "screened.safetensors"),
    ):
        prediction, stops = generate(path, streams["source"])
        streams[name] = prediction
        grades = {}
        for target in ("book_target", "screened_target"):
            costs = [
                Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
                for x, y in zip(prediction, streams[target])
            ]
            grades[target] = dict(edits=sum(costs), exact=costs.count(0), per_row=costs)
        scores[name] = dict(comparison=grades, stopped=sum(stops), checkpoint=sha(path))
    for k, v in streams.items():
        save_streams(k, v, folder)
    write_new(
        EVIDENCE / "training_replay.json",
        dict(
            scores=scores,
            rows=[r["row"] for r in rows],
            streams={k: sha(folder / (k + ".npz")) for k in streams},
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        measure()

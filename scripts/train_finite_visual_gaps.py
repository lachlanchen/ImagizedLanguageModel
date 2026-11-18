"""Fit finite target-image gaps and evaluate a fixed source planner without neural generation."""

import random
import time
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file
from ilm.visual_lm.finite_visual_gaps import FiniteVisualGaps
from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from scripts.evaluate_anchor_gap_writer import (
    EVIDENCE,
    ROSTER,
    CONTROLS,
    load,
    write_new,
    sha,
    DATA,
    OUT,
    read_streams,
    save_streams,
    Rasterizer,
    read_rows,
)
from scripts.train_aligned_visual_translation import MODELS

WEIGHTS = MODELS / "finite_visual_gaps.msgpack.gz"


def run():
    archive = DATA / "training.npz"
    assert sha(archive) == load(OUT / "fit.json")["training_sha256"]
    start = time.monotonic()
    table = FiniteVisualGaps.fit(archive)
    table.save(WEIGHTS)
    table = FiniteVisualGaps.restore(WEIGHTS)
    fit_time = time.monotonic() - start
    torch.set_num_threads(2)
    ckpt = OUT / "writer.safetensors"
    assert sha(ckpt) == load(EVIDENCE / "fit.json")["checkpoint_sha256"]
    state = load_file(str(ckpt))
    model = AnchorGapWriter(state["landmarks"], state["exemplars"])
    model.load_state_dict(state)
    model.eval()
    rows = load(ROSTER)["rows"]
    source, gold = (read_streams(k, DATA / "controls") for k in ("source", "reference"))
    outputs = [table.generate(x, model.plan_packed(x)) for x in source]
    save_streams("finite", outputs, OUT)
    costs = [
        Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
        for x, y in zip(outputs, gold)
    ]
    mapping = load(EVIDENCE / "mapping.json")
    packet = load(EVIDENCE / "review.json")
    original = read_streams("review_pixels", EVIDENCE)
    renderer = Rasterizer()
    chars = set("".join(r["source"] + r["target"] for r in rows))
    for r in read_rows("train"):
        chars.update(r["source"] + r["new"])
    lex = {}
    for ch in sorted(chars):
        try:
            lex.setdefault(renderer.tile(ch), ch)
        except ValueError:
            pass
    ext, repeated, unseen, n = [], [], [], 0
    for j, (r, p) in enumerate(zip(mapping, packet)):
        row = r["row"]
        v = outputs[row]
        same = next(
            (k for k in range(len(p["choices"])) if np.array_equal(v, original[n + k])),
            None,
        )
        if same is not None:
            repeated.append(dict(row=row, packet=j, choice=same))
        else:
            unseen.append((row, j, v))
        n += len(p["choices"])
    random.Random(532120).shuffle(unseen)
    for row, j, v in unseen:
        ext.append(
            dict(
                row=row,
                source=rows[row]["source"],
                output="".join(lex.get(bytes(c), "▧") for c in v),
            )
        )
    write_new(EVIDENCE / "finite_review.json", ext)
    write_new(
        EVIDENCE / "finite.json",
        dict(
            training_seconds=fit_time,
            rules=len(table.rules),
            checkpoint_sha256=sha(WEIGHTS),
            planner_sha256=sha(ckpt),
            edits=costs,
            reused=repeated,
            review_rows=load(CONTROLS)["reader_rows"],
            output_cells=sum(map(len, outputs)),
            stream_sha256=sha(OUT / "finite.npz"),
            runtime_seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    run()

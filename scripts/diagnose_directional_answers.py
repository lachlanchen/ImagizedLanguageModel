"""Measure what the frozen writer can express when given an accepted pictured span."""

from collections import defaultdict, Counter

import numpy as np
import torch

from scripts.evaluate_directional_reading import measure
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.prepare_directional_reading import WORK, PUB, read, dump, sha
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_directional_reading import DirectionalAnswer


def main():
    configure_cuda_cli()
    rows = read(WORK / "confirmation.json")
    assert (
        sha(WORK / "confirmation.json")
        == read(PUB / "reservation.json")["files"]["confirmation.json"]
    )
    ink = Ink(np.load(BASE / "palette.npy"))
    free = unpack(WORK / "signed.json.gz")["original"]
    visible, idx = [], []
    for i, r in enumerate(rows):
        match = next(
            ((a, j) for a in r["answers"] for j, b in enumerate(r["blocks"]) if a in b),
            None,
        )
        if match:
            a, j = match
            visible.append(dict(r, valid=[j], facts={"answer": a}))
            idx.append(i)
    with gpu_owner(), torch.inference_mode():
        guided = observed(DirectionalAnswer("signed", "cuda"), visible, ink, gold=True)
    before = measure(visible, [free[i] for i in idx], ink)
    after = measure(visible, guided, ink)
    scores = defaultdict(Counter)
    for r, a, b in zip(visible, before, after, strict=True):
        assert b["fact"]
        g = scores[r["task"]]
        g["n"] += 1
        for k in ("mentions", "exact", "stopped"):
            g["free_" + k] += a[k]
            g["supplied_" + k] += b[k]
    out = WORK / "supplied.json.gz"
    packed(out, dict(indices=idx, outputs=guided))
    dump(
        PUB / "supplied.json",
        dict(
            scores=scores,
            corpus=sha(WORK / "confirmation.json"),
            outputs=sha(out),
            selection="first accepted literal substring and first enclosing block",
        ),
    )
    print(scores)


if __name__ == "__main__":
    main()

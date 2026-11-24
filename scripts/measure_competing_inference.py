"""Meter short-answer generation and verify each emitted bitmap against the held run."""

from hashlib import sha256
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_competing_facts import dataset
from scripts.evaluate_visual_discourse import unpack
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_competing_facts import PUB, ROOT, dump
from scripts.train_paired_evidence import load_model, gpu_owner
from scripts.write_paired_evidence import generate, reader


@torch.inference_mode()
def measure(rows, enc, gold):
    start_meter = snapshot()
    start = monotonic()
    tiles = np.array([np.frombuffer(tile, np.uint8) for tile in enc.tiles])
    features = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    n = len(np.load(ROOT / "models/visual_discourse/glyphs.npy"))
    base = load_model("ink-pair", "cuda")
    model = reader("contrast", "cuda")
    torch.cuda.synchronize()
    loaded = monotonic()
    for offset in range(0, len(rows), 16):
        group = rows[offset : offset + 16]
        qs = [r["q"] for r in group]
        ss = [r["s"] for r in group]
        emitted = generate(
            base, model, features, n, qs, ss, [s + q for s, q in zip(ss, qs)], 96
        )
        for row, (answer, stopped) in zip(group, emitted, strict=True):
            expected = gold[row["key"]]
            bitmap = tiles[np.asarray(answer, dtype=int)].tobytes()
            if (
                sha256(bitmap).hexdigest() != expected["physical"]
                or stopped != expected["stopped"]
            ):
                raise ValueError(f"Output changed at {row['key']}")
    torch.cuda.synchronize()
    end = monotonic()
    return dict(
        questions=len(rows),
        batch=16,
        seconds=end - start,
        setup_seconds=loaded - start,
        inference_and_validation_seconds=end - loaded,
        **consumed(start_meter, snapshot()),
    )


def main():
    rows, enc = dataset()
    human = [r for r in rows if r["test"] == "human"]
    gold = {
        r["key"]: r
        for r in unpack(PUB / "contrast-outputs.json.gz")
        if r["test"] == "human"
    }
    subset = unpack(PUB / "qwen8vl-image-outputs.json.gz")
    index = {r["key"]: r for r in human}
    with gpu_owner():
        results = dict(
            full=measure(human, enc, gold),
            subset=measure([index[r["key"]] for r in subset], enc, gold),
        )
    dump(PUB / "optical-inference.json", results)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()

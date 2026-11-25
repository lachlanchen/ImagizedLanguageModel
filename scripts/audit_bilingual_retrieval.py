"""Replay the held translations from PNGs and count certified top-one margins."""

import tempfile
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.portable_writer import save_grid, read_image
from scripts.prepare_bilingual_ink import BASE, PUB, WORK, read, dump, sha
from scripts.retrieve_bilingual_images import load_reader, embed


def main():
    torch.set_num_threads(2)
    reader = load_reader("cpu")
    receipt = read(PUB / "corpus.json")
    if sha(WORK / "held.json") != receipt["files"]["held.json"]:
        raise ValueError("Changed holdout")
    rows = read(WORK / "held.json")
    palette = np.load(BASE / "palette.npy")
    with tempfile.TemporaryDirectory() as d:
        grid = Path(d) / "sentence.png"
        sides = []
        for t in (0, 1):
            side = []
            for r in rows:
                ids = r["cells"][t]
                save_grid(palette[ids].tobytes(), grid)
                side.append(read_image(grid, len(ids)))
            sides.append(embed(reader, side, "cpu"))
    cosine = (sides[0] @ sides[1].T).numpy()
    micro = np.rint(cosine * 1_000_000).astype(np.int64)
    scores = {}
    for direction, sim in zip(("en_to_zh", "zh_to_en"), (micro, micro.T), strict=True):
        gold = np.diag(sim).copy()
        other = sim.copy()
        np.fill_diagonal(other, -2_000_000)
        gap = gold - other.max(axis=1)
        target = read(PUB / "result.json")["retrieval"]["aligned/projected"][direction]
        if int(sum(gap > 0)) != target:
            raise ValueError("Raw PNG retrieval differs from the GPU experiment")
        scores[direction] = {
            "n": len(gap),
            "correct": target,
            "certified_at_error_millionths": {
                str(e): int(sum(gap > 2 * e)) for e in (1000, 10000, 50000)
            },
            "margin_millionths": gap.tolist(),
        }
    dump(
        PUB / "retrieval_margin.json",
        dict(
            device="cpu",
            input="256 original PNG encodings",
            unit="rounded cosine × 1,000,000",
            scores=scores,
        ),
    )
    print([(k, v["certified_at_error_millionths"]) for k, v in scores.items()])


if __name__ == "__main__":
    main()

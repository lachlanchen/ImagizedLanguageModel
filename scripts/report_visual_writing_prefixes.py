"""Recount correct prefixes using the pixels emitted by the shared writer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(run, pixel_bank):
    report = json.loads((run / "result.json").read_text())
    sources = run / "sources.npz"
    if digest(sources) != report["source_sha256"]:
        raise ValueError("The target inventory differs from the training receipt")
    bank = np.load(pixel_bank, mmap_mode="r")
    scores = {}
    with np.load(sources) as targets:
        for arm, fit in report["arms"].items():
            scores[arm] = {}
            for split, measurement in fit["measures"].items():
                path = run / arm / f"{split}.npz"
                if digest(path) != measurement["sha256"]:
                    raise ValueError(f"Changed generation archive {path}")
                with np.load(path) as saved:
                    truth = bank[targets[split]]
                    slots = truth.shape[1]
                    present = np.arange(slots)[None, :] < saved["length"][:, None]
                    exact = (saved["packed"][:, :slots] == truth).all(axis=2) & present
                    prefix = np.logical_and.accumulate(exact, axis=1)
                    full_length = saved["length"] == slots
                    stopped_exact = prefix[:, -1] & full_length & saved["stopped"]
                    if not np.array_equal(stopped_exact, saved["exact"]):
                        raise ValueError(f"Recount disagrees with complete outputs for {arm}/{split}")
                    scores[arm][split] = {
                        "position_exact": exact.sum(axis=0).tolist(),
                        "prefix_exact": prefix.sum(axis=0).tolist(),
                        "length_four": int(full_length.sum()),
                        "forms": len(exact),
                    }
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/shared_visual_writing_control_v1"))
    parser.add_argument("--pixel-bank", type=Path, default=Path("artifacts/passage_diversity_v1/pool/pixels.npy"))
    args = parser.parse_args()
    print(json.dumps(recount(args.run, args.pixel_bank), indent=2))

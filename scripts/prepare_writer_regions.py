"""Restore and authenticate the selected training regions from the released selector."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import torch
from scripts.prepare_region_writer import WORK, MODEL, main as prepare
from scripts.train_region_writer import Corpus, select_all, gpu_owner
from scripts.write_region_writer import model
from scripts.prepare_open_exam import sha, read
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    prepare()
    target = WORK / "selector/train-regions.npy"
    expected = read(MODEL / "selector/fit.json")["train_region_hash"]
    if target.exists():
        if sha(target) != expected:
            raise ValueError("Cached training regions differ")
        return
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("high")
        d = Corpus("train", "cuda")
        picks = select_all(model("selector", "cuda"), d, "cuda")
        target.parent.mkdir(exist_ok=True, parents=True)
        with NamedTemporaryFile(dir=target.parent) as f:
            np.save(f, picks)
            f.flush()
            if sha(Path(f.name)) != expected:
                raise ValueError(
                    "Restored selector picks differ from the training data"
                )
            # The temporary file stays owned by its context manager.
            os.link(f.name, target)


if __name__ == "__main__":
    configure_cuda_cli()
    main()

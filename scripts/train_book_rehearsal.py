"""Train the matched optical book writer with shape-preserving role rehearsal."""

import numpy as np
import torch
from scripts.prepare_book_rehearsal import ROOT, BOOK, BOOK_PUB, WORK, PUB, sha, read
from scripts.train_book_breadth import fit
from scripts.train_paired_ink import material
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    configure_cuda_cli()
    design = read(PUB / "design.json")
    if (
        sha(WORK / "replacement.npz") != design["replacement"]
        or sha(BOOK_PUB / "corpus.json") != design["control"]
    ):
        raise ValueError("Substitution design changed")
    control = read(ROOT / "models/book_breadth/broad/fit.json")
    for f, h in control["binding"].items():
        if sha(ROOT / f) != h:
            raise ValueError(f)
    for f, h in read(BOOK_PUB / "corpus.json")["files"].items():
        if sha(BOOK / f) != h:
            raise ValueError(f)
    cells = np.load(BOOK / "cells.npy")
    lengths = np.count_nonzero(cells, axis=2).astype(np.uint8)
    with np.load(WORK / "replacement.npz") as z:
        cells[z["indices"]] = z["cells"]
    if not np.array_equal(np.count_nonzero(cells, axis=2), lengths):
        raise ValueError("Unmatched exposure")
    if len(
        np.unique(cells.reshape(len(cells), -1).view(np.dtype((np.void, 384))))
    ) != len(cells):
        raise ValueError("Repeated visual pair")
    binding = dict(control["binding"])
    for p in (
        ROOT / "references/book_rehearsal_protocol.md",
        ROOT / "scripts/prepare_book_rehearsal.py",
        ROOT / "scripts/train_book_rehearsal.py",
        PUB / "design.json",
    ):
        binding[str(p.relative_to(ROOT))] = sha(p)
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 6 * 2**30:
            raise RuntimeError("Insufficient GPU memory")
        result = fit(
            "rehearsal",
            torch.as_tensor(cells.astype(np.int32), device="cuda"),
            lengths,
            np.arange(len(cells)),
            read(BOOK / "dev.json"),
            material("cuda"),
            binding,
        )
    for k in ("updates", "positions"):
        assert [r[k] for r in result["epochs"]] == [r[k] for r in control["epochs"]]


if __name__ == "__main__":
    main()

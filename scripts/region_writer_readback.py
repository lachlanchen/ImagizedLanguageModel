"""Publish the training bitmap bank and a transcript for external evaluation only."""

from collections import Counter
import numpy as np
from scripts.prepare_region_writer import WORK, MODEL, Ink, inventory
from scripts.prepare_open_exam import dump
from scripts.prepare_evidence_views import ROOT, read, sha
from shutil import copyfile


def main():
    palette = np.load(WORK / "palette.npy")
    ink = Ink(palette)
    train, dev = inventory()
    counts = Counter(
        c
        for row in train + dev
        for s in [row["q"], *row["blocks"], *row["answers"]]
        for c in set(s)
    )
    chars = [""] * len(palette)
    for c, _ in counts.most_common():
        v = ink.encode(c)[0]
        if v < len(chars) and not chars[v] and c not in ink.bad:
            chars[v] = c
    assert all(chars[2:]), [i for i in range(2, len(chars)) if not chars[i]]
    MODEL.mkdir(parents=True, exist_ok=True)
    assert (
        sha(WORK / "palette.npy")
        == read(ROOT / "references/evidence/region_writer_v1/corpus.json")["files"][
            "palette.npy"
        ]
    )
    copyfile(WORK / "palette.npy", MODEL / "palette.npy")
    dump(MODEL / "readback.json", chars)


if __name__ == "__main__":
    main()

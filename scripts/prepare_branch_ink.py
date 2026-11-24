"""Reserve a second document-disjoint exam evaluation and the physical training divergences."""

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scripts.prepare_open_exam import (
    build as exam_build,
    MODEL as EXAM,
    PUB as EXAM_PUB,
    WORK as EXAM_WORK,
)
from scripts.prepare_open_exam import ROOT, dump as dump, read, sha
from scripts.prepare_multiple_ink_runs import prepared_files

PUB = ROOT / "references/evidence/branch_ink_v1"
WORK = ROOT / "artifacts/branch_ink_v1"
MODEL = ROOT / "models/branch_ink"


def positions(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["doc"]].append(i)
    cuts = np.full(len(rows), -1, np.int32)
    for group in groups.values():
        for index, i in enumerate(group):
            a = rows[i]
            for j in group[index + 1 :] + group[:index]:
                b = rows[j]
                if a["pi"] == b["pi"] and a["qi"] != b["qi"] and a["y"] != b["y"]:
                    cuts[i] = next(
                        k for k, (u, v) in enumerate(zip(a["y"], b["y"])) if u != v
                    )
                    break
    return cuts


def build(stage):
    original = read(EXAM_PUB / "corpus.json")
    if sha(EXAM_WORK / "held.json") != original["files"]["held.json"]:
        raise ValueError("Earlier evaluation cache differs")
    receipt = exam_build(stage, read(EXAM_WORK / "held.json"))
    for name in ("train.json", "tiles.npy"):
        if receipt["files"][name] != original["files"][name]:
            raise ValueError("Changed physical training data")
        (stage / name).unlink()
    rows = read(EXAM_WORK / "train.json")
    if sha(EXAM_WORK / "train.json") != original["files"]["train.json"]:
        raise ValueError("Earlier training cache differs")
    cut = positions(rows)
    np.save(stage / "cuts.npy", cut)
    return dict(
        source=sha(EXAM_PUB / "corpus.json"),
        stats=receipt["stats"],
        fresh=receipt["held"],
        cut_counts=dict(sorted(Counter(map(int, cut)).items())),
        files={s: sha(stage / s) for s in ("cuts.npy", "held.json")},
    )


if __name__ == "__main__":
    prepared_files(
        WORK,
        PUB / "corpus.json",
        {s: ("files", s) for s in ("held.json", "cuts.npy")},
        build,
    )

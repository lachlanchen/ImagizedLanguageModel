"""Translate two forms of each classical passage with the same frozen writer."""

import argparse
from collections import Counter
import numpy as np
import torch

from scripts.prepare_book_breadth import (
    BASE,
    ROOT,
    PUB,
    WORK,
    Ink,
    rank,
    read,
    key,
    dump,
    sha,
)
from scripts.evaluate_book_breadth import generate, MODES
from scripts.evaluate_proposition_translation import semantic
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def prepare():
    ink = Ink(np.load(BASE / "palette.npy"))
    inv = [set(ids) for ids in read(BASE / "inventories.json")]
    candidates = []
    totals = Counter()
    for book in ("daodejing", "shanhaijing"):
        local = []
        for f in sorted(
            (ROOT.parent / "ZhJpBook/books" / book).glob(
                "work/*/interlinear/chunks/*.json"
            )
        ):
            for para in read(f)["paragraphs"]:
                for j, u in enumerate(para["units"]):
                    total = totals[book + "_examined"] + 1
                    totals[book + "_examined"] = total
                    texts = [
                        "".join(x["t"] for x in u.get(k, [])).strip()
                        for k in ("en", "wenyan", "zh_modern")
                    ]
                    if not all(2 <= len(s) <= 96 for s in texts) or key(
                        texts[1]
                    ) == key(texts[2]):
                        continue
                    cells = [ink.encode(s) for s in texts]
                    if cells[1] == cells[2] or any(
                        not set(z) <= inv[min(t, 1)] or ink.bad.intersection(texts[t])
                        for t, z in enumerate(cells)
                    ):
                        continue
                    local.append(
                        dict(
                            doc=book,
                            id=f"{book}/{para['id']}/{j}",
                            text=texts,
                            cells=cells,
                        )
                    )
        totals[book + "_supported"] = len(local)
        candidates.extend(local)
    wanted = {tuple(c) for r in candidates for c in r["cells"]}
    forbidden = set()

    def screen(sequence):
        k = tuple(sequence)
        if k in wanted:
            forbidden.add(k)

    for batch in np.array_split(np.load(WORK / "cells.npy", mmap_mode="r"), 128):
        for seq in batch.reshape(-1, 96):
            screen(seq[: np.count_nonzero(seq)].tolist())
    for split in ("dev", "test"):
        for r in read(WORK / (split + ".json")):
            for seq in r["cells"]:
                screen(seq)
    selected = []
    for r in sorted(candidates, key=lambda z: rank(z["id"])):
        if totals[r["doc"] + "_selected"] >= 12 or any(
            tuple(c) in forbidden for c in r["cells"]
        ):
            continue
        selected.append(r)
        forbidden.update(tuple(c) for c in r["cells"])
        totals[r["doc"] + "_selected"] += 1
    rows = []
    for i, r in enumerate(selected):
        other = next(
            z
            for z in selected[i + 1 :] + selected[:i]
            if z["doc"] == r["doc"] and z["id"] != r["id"]
        )
        for j, group in ((1, "classical"), (2, "modern")):
            rows.append(
                dict(
                    group=group,
                    id=r["id"],
                    doc=r["doc"],
                    t=0,
                    source=r["text"][j],
                    ref=r["text"][0],
                    foil=other["text"][0],
                    input=r["cells"][j],
                    target=r["cells"][0],
                )
            )
    dump(WORK / "script-test.json", rows)
    dump(
        PUB / "script-split.json",
        dict(counts=totals, test=sha(WORK / "script-test.json")),
    )
    print(totals, flush=True)


def evaluate():
    configure_cuda_cli()
    rows = read(WORK / "script-test.json")
    if sha(WORK / "script-test.json") != read(PUB / "script-split.json")["test"]:
        raise ValueError("Supplemental cohort changed")
    with gpu_owner():
        out = generate(rows, WORK / "script-pixels")
        dump(WORK / "script-generated.json", dict(rows=rows, answers=out))
        torch.cuda.empty_cache()
        grades = semantic(rows, out, tuple(MODES))
    dump(WORK / "script-graded.json", dict(rows=rows, answers=out, grades=grades))
    both = Counter()
    for k in range(0, len(rows), 2):
        assert rows[k]["id"] == rows[k + 1]["id"]
        for m in (*MODES, "qwen", "reference", "foil"):
            both[m] += all(
                out[i][m]["stop"] and all(grades["votes"][2 * i + j][m] for j in (0, 1))
                for i in (k, k + 1)
            )
    dump(
        PUB / "script-result.json",
        dict(
            groups=grades["groups"],
            both_forms=both,
            units=len(rows) // 2,
            votes=grades["votes"],
            generated=sha(WORK / "script-generated.json"),
            graded=sha(WORK / "script-graded.json"),
        ),
    )
    print(grades["groups"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "evaluate"))
    if p.parse_args().action == "prepare":
        prepare()
    else:
        evaluate()

"""Estimate whole-answer fit on real training documents with the same two-question geometry."""

from collections import defaultdict
from hashlib import sha256
import torch
from scripts.train_open_exam import corpus
from scripts.prepare_branch_ink import MODEL, EXAM, PUB
from scripts.prepare_open_exam import dump
from scripts.check_branch_ink_transfer import write
from scripts.write_open_exam import load
from scripts.write_visual_discourse import Writer
from scripts.train_paired_evidence import gpu_owner


def pairs(rows):
    docs = defaultdict(list)
    for row in rows:
        docs[row["doc"]].append(row)
    chosen = []
    for genre, quota in (("m", 64), ("d", 16)):
        seen = 0
        for name in sorted(
            docs, key=lambda k: sha256(("branch training audit " + k).encode()).digest()
        ):
            group = docs[name]
            if group[0]["genre"] != genre:
                continue
            a = group[0]
            b = next(
                (r for r in group[1:] if r["y"] != a["y"] and r["qi"] != a["qi"]), None
            )
            if b is None:
                continue
            chosen.extend([a, b])
            seen += 1
            if seen == quota:
                break
        assert seen == quota
    return chosen


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    rows, tiles, _ = corpus("train")
    selected = pairs(rows)
    report = {}
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        for arm, root in (("matched", EXAM), ("branch", MODEL)):
            net = load(arm, "cuda", root)
            outs = write(
                writer, net, tiles, [dict(q=r["qi"], s=r["pi"]) for r in selected]
            )
            exact = [
                end and ids == r["y"][:-1]
                for (ids, end), r in zip(outs, selected, strict=True)
            ]
            report[arm] = dict(
                questions=len(selected),
                exact=sum(exact),
                both=sum(a and b for a, b in zip(exact[::2], exact[1::2], strict=True)),
                right_first=sum(
                    bool(ids) and ids[0] == r["y"][0]
                    for (ids, _), r in zip(outs, selected, strict=True)
                ),
            )
    dump(PUB / "training-sample.json", report)

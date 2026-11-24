"""Evaluate ordered source-copy decoding on the original questions with frozen weights."""

import json
import gzip
import torch
from scripts.prepare_chinese_two_fact import (
    WORK,
    PUB,
    MODEL,
    MARKS,
    FIELDS,
    DOMAINS,
    Rasterizer,
    read,
    dump,
)
from scripts.train_chinese_two_fact import archive, load, gpu_owner, output_bank
from scripts.write_chinese_two_fact import keys_for, write_plans
from scripts.evaluate_chinese_two_fact import (
    surfaces,
    observations,
    challenge,
    grade,
    totals,
    expand_text,
)


@torch.inference_mode()
def evaluate(mode="ink", aligned=False):
    filename = ("alignment-" + mode + ".json") if aligned else "coverage.json"
    if (PUB / filename).exists():
        raise FileExistsError(filename)
    meta = read(PUB / "corpus.json")
    bank = archive(WORK / "bank.npz", meta["bank"])
    held = {s: read(WORK / (s + ".json")) for s in ("val", "test")}
    held["swap"] = challenge(held["test"])
    if aligned:
        for name, frame in (
            ("fresh-list", "请按顺序介绍：{a}、{b}。"),
            ("fresh-delay", "把{b}留到第二项，先回答{a}。"),
        ):
            held[name] = [
                dict(r, q=frame.format(a=r["keys"][0], b=r["keys"][1]))
                for r in held["swap"]
                if len(r["card"]) > 2
            ]
    arms = json.loads(gzip.decompress((PUB / "replies.json.gz").read_bytes()))
    raster = Rasterizer()
    chars = read(MODEL / "alphabet.json")
    scores = {}
    with gpu_owner():
        from scripts.train_field_alignment import load_alignment

        aligner = load_alignment(mode, "cuda") if aligned else None
        net = load(mode + "-joined", "cuda")
        palette = output_bank("cuda")
        obs = torch.as_tensor(observations(bank, mode), device="cuda")
        labels = net.encode_rows(obs[bank["labels"]])
        for split, rows in held.items():
            lines = []
            inputs = surfaces(rows, mode, raster)
            for start in range(0, len(rows), 32):
                part = rows[start : start + 32]
                q = torch.as_tensor(inputs[start : start + len(part)], device="cuda")
                dom = torch.tensor(
                    [list(DOMAINS).index(r["domain"]) for r in part], device="cuda"
                )
                av = torch.tensor(
                    [[k in r["card"] for k in FIELDS] for r in part], device="cuda"
                )
                override = aligner.choose(q, dom, av) if aligned else None
                plan, seq = write_plans(
                    net,
                    q,
                    labels,
                    dom,
                    av,
                    palette,
                    chars.index(MARKS[-1]),
                    teacher=override,
                    cover=True,
                )
                for j, (r, p, s) in enumerate(zip(part, plan, seq)):
                    keys = keys_for(p)
                    line = expand_text(s, chars, keys, r)
                    if (
                        not aligned
                        and keys != arms[split]["ink-joined"][start + j]["plan"]
                    ):
                        raise ValueError("Plan replay differs")
                    lines.append(grade(line, keys, r))
            groups = {
                str(n): totals([d for d, r in zip(lines, rows) if len(r["keys"]) == n])
                for n in (1, 2)
            }
            pairs = [d for d, r in zip(lines, rows) if len(r["keys"]) == 2]
            scores[split] = dict(groups=groups, lines=pairs, all_lines=lines)
            if split == "swap" or split.startswith("fresh-"):
                scores[split]["paired_order"] = sum(
                    a["answer_order"] and b["answer_order"]
                    for a, b in zip(lines[::2], lines[1::2])
                )
    dump(PUB / filename, scores)
    print(scores["test"]["groups"])


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", choices=("ink", "byte"))
    args = ap.parse_args()
    evaluate(args.align or "ink", bool(args.align))

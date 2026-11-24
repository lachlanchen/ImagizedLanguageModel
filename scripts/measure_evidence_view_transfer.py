"""Count correct-prefix probability transferred from evidence to an entire page."""

import argparse
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_evidence_views import groups, official_index
from scripts.prepare_evidence_views import Alphabet, MODEL, PUB
from scripts.prepare_open_exam import dump
from scripts.write_open_exam import load
from scripts.write_recurrent_evidence import generate
from scripts.write_visual_discourse import Writer
from scripts.train_paired_evidence import gpu_owner


def main(root, report):
    judge, _, _, dev, _ = official_index()
    data = groups(judge, dev)
    a = Alphabet()
    items = []
    # First 64 eligible first turns in official order, including their learned stop.
    for full, crop in zip(data["full"], data["evidence"], strict=True):
        answer = full["refs"][0]
        if not answer or len(answer) >= 64:
            continue
        assert full["doc"] == crop["doc"]
        items.append(
            (
                a.encode(full["question"]),
                a.encode(full["page"]),
                a.encode(crop["page"]),
                a.encode(answer) + [1],
            )
        )
        if len(items) == 64:
            break
    features = torch.from_numpy(observations({"tiles": a.array()}, "ink")).cuda()
    report.mkdir(parents=True, exist_ok=True)
    results = {}
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        for arm in ("parent", "whole", "highlight"):
            net = load(arm, "cuda", root)
            counts = dict(
                positions=0,
                crop_confident=0,
                excess_drop=0,
                certified=0,
                free_prefix_decisions_certified=0,
                whole_answers_certified=0,
                exact=0,
            )
            for i in range(0, len(items), 8):
                qs, full, crop, targets = map(list, zip(*items[i : i + 8]))
                steps = max(map(len, targets))
                forced = torch.ones((len(qs), steps), device="cuda", dtype=torch.long)
                for j, y in enumerate(targets):
                    forced[j, : len(y)] = torch.tensor(y, device="cuda")
                probs = []
                for p in (full, crop):
                    lp = generate(
                        writer.model,
                        net,
                        features,
                        a.n,
                        qs,
                        p,
                        [x + y for x, y in zip(p, qs)],
                        steps,
                        forced=forced,
                    )
                    probs.append(lp.exp().cpu().numpy())
                free = generate(
                    writer.model,
                    net,
                    features,
                    a.n,
                    qs,
                    full,
                    [p + q for p, q in zip(full, qs)],
                    64,
                )
                for j, (y, (ids, end)) in enumerate(zip(targets, free, strict=True)):
                    f, c = (p[j, : len(y)] for p in probs)
                    cert = (c > 0.60001) & (c - f <= 0.1)
                    if not np.all(f[cert] > 0.5):
                        raise AssertionError("Probability certificate fails")
                    out = ids + ([1] if end else [])
                    counts["positions"] += len(y)
                    counts["certified"] += int(cert.sum())
                    counts["crop_confident"] += int((c > 0.60001).sum())
                    counts["excess_drop"] += int((c - f > 0.1).sum())
                    counts["whole_answers_certified"] += int(cert.all())
                    counts["exact"] += out == y
                    for k, mark in enumerate(y):
                        if cert[k]:
                            assert k < len(out) and out[k] == mark
                            counts["free_prefix_decisions_certified"] += 1
                        if k >= len(out) or out[k] != mark:
                            break
            results[arm] = dict(questions=len(items), **counts)
    dump(report / "conditional-transfer.json", results)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--output", type=Path, default=PUB)
    a = p.parse_args()
    main(a.model_dir, a.output)

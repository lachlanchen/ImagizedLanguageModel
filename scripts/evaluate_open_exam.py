"""Assess freely written exam phrases independently from teacher-forced option ranking."""

import argparse
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.train_open_exam import corpus, likelihood
from scripts.write_open_exam import load
from scripts.write_visual_discourse import Writer
from scripts.write_recurrent_evidence import generate
from scripts.prepare_open_exam import WORK, PUB, MODEL, dump
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import packed

ARMS = ("parent", "exchanged", "matched")
CONDITIONS = ("intact", "other_question", "other_page")


def assignments(rows):
    paired = defaultdict(list)
    for i, r in enumerate(rows):
        if r["cohort"] == "paired":
            paired[r["doc"]].append(i)
    q = {i: j for a, b in paired.values() for i, j in ((a, b), (b, a))}
    page = {}
    for g in ("d", "m"):
        all_ids = [i for i, r in enumerate(rows) if r["genre"] == g]
        single = sorted(
            (i for i in all_ids if i not in q), key=lambda i: len(rows[i]["qi"])
        )
        for idx, i in enumerate(single):
            q[i] = next(
                j
                for j in single[idx + 1 :] + single[:idx]
                if rows[j]["qi"] != rows[i]["qi"]
            )
        ids = sorted(all_ids, key=lambda i: len(rows[i]["pi"]))
        for idx, i in enumerate(ids):
            page[i] = next(
                j
                for j in ids[idx + 1 :] + ids[:idx]
                if rows[j]["doc"] != rows[i]["doc"] and rows[j]["pi"] != rows[i]["pi"]
            )
    return q, page


def summarize(items):
    paired = defaultdict(list)
    for r in items:
        if r["cohort"] == "paired":
            paired[r["doc"]].append(r)
    return dict(
        n=len(items),
        free=sum(r["exact"] for r in items),
        stops=sum(r["stop"] for r in items),
        paired_n=len(paired),
        both_free=sum(all(v["exact"] for v in g) for g in paired.values()),
        ranked=sum(r["ranked"] for r in items),
        both_ranked=sum(all(v["ranked"] for v in g) for g in paired.values()),
        absent_multi_n=sum(r["absent_multi"] for r in items),
        absent_multi_free=sum(r["exact"] and r["absent_multi"] for r in items),
        no_choice_cue_n=sum(not r["cue"] for r in items),
        no_choice_cue_exact=sum(not r["cue"] and r["exact"] for r in items),
        expected_uniform_ranked=sum(1 / r["choices"] for r in items),
    )


@torch.inference_mode()
def main(arms=ARMS, model_dir=MODEL, work=WORK, report=PUB, *, samples=None):
    work.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    rows, tiles, n = corpus("held") if samples is None else samples
    qswap, pswap = assignments(rows)
    metrics = {}
    with gpu_owner():
        writer = Writer(device="cuda")
        x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        bk = writer.model.encode(x)
        for arm in arms:
            net = load(arm, "cuda", model_dir)
            metrics[arm] = {}
            original = {}
            private = []
            public = []
            for cond in CONDITIONS:
                for i in range(0, len(rows), 12):
                    group = rows[i : i + 12]
                    qs = [
                        (rows[qswap[i + j]] if cond == "other_question" else r)["qi"]
                        for j, r in enumerate(group)
                    ]
                    ps = [
                        (rows[pswap[i + j]] if cond == "other_page" else r)["pi"]
                        for j, r in enumerate(group)
                    ]
                    generated = generate(
                        writer.model,
                        net,
                        x,
                        n,
                        qs,
                        ps,
                        [p + q for p, q in zip(ps, qs)],
                        64,
                    )
                    # Candidate phrases enter only this separate likelihood call after free generation.
                    cand = [
                        (q, p, y)
                        for q, p, r in zip(qs, ps, group, strict=True)
                        for y in r["ys"]
                    ]
                    scores = []
                    for k in range(0, len(cand), 16):
                        rr = list(zip(*cand[k : k + 16], strict=True))
                        scores.extend(
                            likelihood(net, writer.model, x, bk, n, *rr).tolist()
                        )
                    at = 0
                    for j, ((ids, stop), r) in enumerate(
                        zip(generated, group, strict=True)
                    ):
                        k = len(r["ys"])
                        vals = scores[at : at + k]
                        at += k
                        correct = stop and ids == r["y"][:-1]
                        maxv = max(vals)
                        best = [k for k, v in enumerate(vals) if abs(v - maxv) <= 1e-6]
                        ranked = len(best) == 1 and r["ys"][best[0]] == r["y"]
                        pix = tiles[np.asarray(ids, int)]
                        rec = dict(
                            id=r["id"],
                            doc=r["doc"],
                            genre=r["genre"],
                            cohort=r["cohort"],
                            condition=cond,
                            choices=len(vals),
                            cue=r["choice_cue"],
                            absent_multi=r["absent"] and len(r["y"]) >= 3,
                            exact=correct,
                            stop=stop,
                            cells=len(ids),
                            ranked=ranked,
                            tied=len(best) > 1,
                            output_sha256=sha256(pix.tobytes()).hexdigest(),
                        )
                        public.append(rec)
                        private.append(
                            dict(**rec, ids=ids, choice_log_likelihoods=vals)
                        )
                items = [r for r in public if r["condition"] == cond]
                metrics[arm][cond] = summarize(items)
                if cond == "intact":
                    original = {r["id"]: r for r in items}
                else:
                    metrics[arm][cond]["changed_output"] = sum(
                        r["output_sha256"] != original[r["id"]]["output_sha256"]
                        for r in items
                    )
                    metrics[arm][cond]["lost_original_exact"] = sum(
                        original[r["id"]]["exact"] and not r["exact"] for r in items
                    )
                    metrics[arm][cond]["lost_original_absent_multi"] = sum(
                        original[r["id"]]["exact"]
                        and r["absent_multi"]
                        and not r["exact"]
                        for r in items
                    )
                print(arm, cond, metrics[arm][cond], flush=True)
            packed(work / (arm + "-responses.json.gz"), private)
            packed(report / (arm + "-outcomes.json.gz"), public)
    dump(report / "results.json", metrics)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument(
        "--output", type=Path, help="Separate directory for this replication"
    )
    args = parser.parse_args()
    main(args.arm or ARMS, args.model_dir, args.output or WORK, args.output or PUB)

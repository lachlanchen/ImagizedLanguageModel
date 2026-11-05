#!/usr/bin/env python3
"""Verify completed V63 score tables and bind a compact receipt, without CUDA.

This independently checks score aggregation and the registered bootstrap. It
does not rerun generation or reinterpret the original bitmap scoring function.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    if not np.allclose(actual, expected, rtol=0, atol=2e-14):
        raise ValueError(f"aggregate differs: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT/"artifacts/question_assignment_v63_20261105")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    files = {}

    def read(path):
        path = run/path
        files[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    report = read(Path("development/report.json"))
    preflight = read(Path("preflight/report.json"))
    if not all(preflight["checks"].values()):
        raise ValueError("preflight failed")
    for name, expected in preflight["provenance"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    metrics = tuple(report["measurements"]["control"])
    routes = {}
    for route in ("control", "control_blank", "aligned", "aligned_blank"):
        rows = read(Path(f"development/{route}_answers.json"))
        if len(rows) != 256 or [r["id"] for r in rows] != report["primary_ids"]:
            raise ValueError("primary cases changed")
        if any(not np.isfinite(float(r[m])) for r in rows for m in metrics):
            raise ValueError("nonfinite primary score")
        for metric in metrics:
            close(np.mean([r[metric] for r in rows]), report["measurements"][route][metric])
        routes[route] = rows

    # Resample entire source articles, preserving their unequal example counts.
    articles = [r["article"] for r in routes["control"]]
    unique = sorted(set(articles))
    groups = [[i for i, name in enumerate(articles) if name == article] for article in unique]
    draws = np.random.default_rng(20266329).integers(0, len(groups), (2000, len(groups)))
    sizes = np.array([len(g) for g in groups])
    for left, right in (("aligned", "control"), ("aligned", "aligned_blank"),
                        ("control", "control_blank")):
        if [r["article"] for r in routes[left]] != [r["article"] for r in routes[right]]:
            raise ValueError("paired articles changed")
        for metric in ("exact_answer", "raster_f1", "pixel_foreground_f1"):
            differences = np.array([a[metric]-b[metric] for a, b in zip(routes[left], routes[right])])
            sums = np.array([sum(differences[g]) for g in groups])
            means = sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
            expected = report["comparisons"][f"{left}_minus_{right}_{metric}"]
            close(differences.mean(), expected["mean"])
            close(np.quantile(means, (.025, .975)), expected["ci95"])
            if (expected["articles"], expected["examples"], expected["replicates"], expected["seed"]) != (198, 256, 2000, 20266329):
                raise ValueError("registered bootstrap differs")

    training = {}
    for arm in ("control", "aligned"):
        row = read(Path(f"{arm}/training_report.json"))
        if row["update"] != 2000 or row["smoke_only"] or row["assignment_weight"] != float(arm == "aligned"):
            raise ValueError("complete registered training required")
        if row["provenance"] != preflight["provenance"]:
            raise ValueError("training identity differs")
        weight = run/arm/"model_final.pt"
        files[str(weight.relative_to(ROOT))] = sha(weight)
        if files[str(weight.relative_to(ROOT))] != report["checkpoint_sha256"][arm]:
            raise ValueError("final model changed")
        replay = read(Path(f"standalone/{arm}/verification.json"))
        if not all(replay["checks"].values()) or replay["checkpoint_sha256"] != report["checkpoint_sha256"][arm]:
            raise ValueError("standalone replay failed")
        likelihood = read(Path(f"development/{arm}_likelihood.json"))
        if len(likelihood) != 1321:
            raise ValueError("likelihood cases changed")
        close(np.mean([r["answer_pixel_nll"] for r in likelihood]), report["likelihood"][arm])
        pairs = read(Path(f"development/{arm}_pair_results.json"))
        if len(pairs) != 393:
            raise ValueError("pair cases changed")
        for pair in pairs:
            e = np.asarray(pair["energies"])
            if e.shape != (2, 2) or not np.isfinite(e).all():
                raise ValueError("invalid reference energies")
            margin = e[0, 1]+e[1, 0]-e[0, 0]-e[1, 1]
            # The evaluator forms its margin in FP32, not Python FP64.
            close(pair["assignment_margin"], float(np.float32(np.float32(e[0, 1])+np.float32(e[1, 0])
                  -np.float32(e[0, 0])-np.float32(e[1, 1]))))
            if pair["diagonal_assignment"] != float(margin > 0):
                raise ValueError("reference assignment differs")
            if pair["both_energy_rank_correct"] != float(e[0, 0] < e[0, 1] and e[1, 1] < e[1, 0]):
                raise ValueError("individual reference ranks differ")
        for metric, value in report["reference_assignment"][arm].items():
            close(np.mean([r[metric] for r in pairs]), value)
        close(np.mean([r["both_generated_exact"] for r in pairs]), report["paired_both_exact"][arm])
        measured = report["measurements"][arm]
        gates = dict(report["free_reading_gates"][arm])
        for key, value in dict(exact_answers=measured["exact_answer"] >= .1,
                raster_f1=measured["raster_f1"] >= .25,
                question_dependence=report["comparisons"][f"{arm}_minus_{arm}_blank_raster_f1"]["ci95"][0] > 0,
                paired_answers=report["paired_both_exact"][arm] >= .02,
                capped_outputs=measured["capped"] < .2).items():
            if gates[key] != value:
                raise ValueError("reading quality decision differs")
        if all(gates.values()) != report["free_reading_pass"][arm]:
            raise ValueError("reading decision differs")
        training[arm] = {k: row[k] for k in ("update", "elapsed_seconds", "peak_allocated_vram_gib")}

    journals = [(run/arm/"sample_order.jsonl").read_bytes() for arm in ("control", "aligned")]
    journal_rows = [json.loads(line) for line in journals[0].splitlines()]
    if journals[0] != journals[1] or [r["update"] for r in journal_rows] != list(range(1, 2001)):
        raise ValueError("matched sample order differs")
    if any(len(r["pair_indices"]) != 4 for r in journal_rows):
        raise ValueError("pair draw count differs")
    for arm in ("control", "aligned"):
        path = run/arm/"sample_order.jsonl"
        files[str(path.relative_to(ROOT))] = sha(path)
    gain = report["comparisons"]["aligned_minus_control_raster_f1"]["ci95"][0] > 0
    decision = "assignment_intervention_pass" if gain and report["free_reading_pass"]["aligned"] else "reject_assignment_intervention"
    if report["decision"] != decision or report["matched_raster_gain"] != gain:
        raise ValueError("intervention decision differs")
    receipt = dict(experiment="question-assignment-v63", decision=decision,
        free_reading_pass=report["free_reading_pass"], measurements=report["measurements"],
        comparisons=report["comparisons"], reference_assignment=report["reference_assignment"],
        paired_both_exact=report["paired_both_exact"], likelihood=report["likelihood"], training=training,
        updates_per_arm=2000, pair_draws_per_arm=8000, matched_journals=True,
        offline_replay_both_arms=True, registered_aggregates_and_bootstrap_verified=True,
        frozen_training_sources_unchanged=True, primary_examples=256, pairs=393,
        likelihood_examples=1321, general_ilm_goal_complete=False,
        scope="Checks saved scores and artifact identity; does not rerun generation or raster scoring.",
        files_sha256=files, verifier_sha256=sha(Path(__file__)))
    encoded = json.dumps(receipt, indent=2, sort_keys=True)+"\n"
    if args.out.exists() and args.out.read_text() != encoded:
        raise FileExistsError("preserve the existing receipt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        args.out.write_text(encoded)
    print(json.dumps(dict(decision=decision, receipt_sha256=sha(args.out), verified_files=len(files))))


if __name__ == "__main__":
    main()

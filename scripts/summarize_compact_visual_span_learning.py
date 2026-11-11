#!/usr/bin/env python3
"""Authenticate completed readings and summarize the fixed language trajectory."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha


SEED = 202611154
REPLICATES = 10000
EXAMPLES = ("qa_01_a_normal", "qa_01_b_normal", "book_01_normal")


def load(path):
    return json.loads(path.read_text())


def clustered_interval(values):
    """Resample whole, equally sized passage groups, retaining paired questions."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Supply a finite rectangular matrix of passage groups")
    means = values.mean(axis=1)
    draws = np.random.default_rng(SEED).integers(0, len(means), (REPLICATES, len(means)))
    interval = np.quantile(means[draws].mean(axis=1), [.025, .975])
    return {"mean": float(values.mean()), "ci95": interval.tolist(),
            "groups": len(means), "observations": values.size}


def qa_values(records, references, condition, paired=False):
    groups = defaultdict(list)
    for row in records:
        reference = references[row["case"].removesuffix("_" + row["condition"])]
        if row["condition"] == condition and reference["kind"] == "qa":
            groups[reference["paired_group"]].append(row)
    if set(groups) != set(range(1, 17)) or any(len(rows) != 2 for rows in groups.values()):
        raise ValueError("Keep all sixteen complete question pairs in the denominator")
    values = []
    for group in sorted(groups):
        rows = sorted(groups[group], key=lambda row: row["case"])
        if any(type(row["complete_exact"]) is not bool for row in rows):
            raise ValueError("Complete-answer decisions must be explicit booleans")
        if paired:
            distinct = rows[0].get("image_sha256") != rows[1].get("image_sha256")
            # The native-text reference has no generated images. Its distinct,
            # disjoint canonical answers already imply distinct correct strings.
            distinct = distinct or all(row.get("native_text_reference") for row in rows)
            values.append([all(row["complete_exact"] for row in rows) and distinct])
        else:
            values.append([row["complete_exact"] for row in rows])
    return np.asarray(values, dtype=float)


def metric_rows(records, references, step):
    conditions = sorted({row["condition"] for row in records if row["kind"] == "qa"})
    result = []
    for condition in conditions:
        for paired in (False, True):
            values = qa_values(records, references, condition, paired)
            result.append({"step": step, "condition": condition,
                           "metric": "qa_both_correct" if paired else "qa_complete_exact",
                           "count": int(values.sum()), "cases": values.size,
                           **clustered_interval(values)})
    return result


def authenticate_registration(evaluation):
    preparation = load(evaluation / "preparation.json")
    for name, key in (("references.json", "references_sha256"),
                      ("inputs/manifest.json", "input_manifest_sha256")):
        if sha(evaluation / name) != preparation[key]:
            raise ValueError(f"Registered evaluation changed: {name}")
    references = {row["id"]: row for row in load(evaluation / "references.json")["reference_rows"]}
    if preparation["paired_qa_groups"] != 16 or preparation["normal_inputs"] != 44:
        raise ValueError("This analysis is fixed to the registered 44-prompt cohort")
    return preparation, references


def authenticate_visual(evaluation, pipeline, reading_path):
    # Recompute the original scorer in a temporary directory. The saved score
    # file and its hashes alone cannot certify a changed interpretation.
    import torch
    from scripts.review_compact_visual_span import collect_run, registered_inputs, score

    torch.set_num_threads(2)
    fit_path = Path(load(pipeline / "request.json")["fit"])
    fit = load(fit_path / "report.json")
    snapshots = sorted(fit_path.glob("image_heads_step_*.pt"))
    if (len(snapshots) != 5 or snapshots[0].stem != "image_heads_step_0000000"
            or snapshots[-1].name != fit["final_image_heads"]
            or sha(snapshots[-1]) != fit["final_image_heads_sha256"]):
        raise ValueError("Require the five registered snapshots from the completed fit")
    review = pipeline / "review"
    scores = load(review / "scores.json")
    with tempfile.TemporaryDirectory(prefix="ilm-language-analysis-") as temporary:
        check = Path(temporary)
        (check / "mapping.json").symlink_to((review / "mapping.json").resolve())
        (check / "pages").symlink_to((review / "pages").resolve())
        with redirect_stdout(io.StringIO()):
            score(evaluation, check, reading_path)
        if load(check / "scores.json") != scores:
            raise ValueError("Saved scores differ from the authenticated raw readings")
    _, registered = registered_inputs(evaluation)
    steps = {}
    for index, snapshot in enumerate(snapshots):
        run = pipeline / snapshot.stem
        request, report, _ = collect_run(run, registered)
        expected = (set(registered) if index in {0, 4} else
                    {key for key, row in registered.items() if row["condition"] == "normal"})
        if {row["id"] for row in report["records"]} != expected:
            raise ValueError("Keep all endpoint controls and all intermediate normal prompts")
        if request["checkpoint_sha256"] != sha(snapshot):
            raise ValueError("Generation used a different checkpoint")
        step = report["optimizer_updates"]
        if step in steps or snapshot.stem != f"image_heads_step_{step:07d}":
            raise ValueError("Snapshot and actual optimizer step differ")
        steps[step] = {"run": run, "checkpoint_sha256": request["checkpoint_sha256"],
                       "generation_report_sha256": sha(run / "report.json")}
    if sorted(steps)[-1] != fit["updates"]:
        raise ValueError("Final evaluated update differs from the completed fit")
    if {row["step"] for row in scores["records"]} != steps.keys():
        raise ValueError("Readings must cover every registered checkpoint")
    return scores, steps, {"scores_sha256": sha(review / "scores.json"),
                           "readings_sha256": sha(reading_path), "evaluator": scores["evaluator"],
                           "fit_report_sha256": sha(fit_path / "report.json"),
                           "pipeline_request_sha256": sha(pipeline / "request.json")}


def authenticate_baseline(evaluation, generation):
    from scripts.compare_compact_visual_span_qwen import score

    expected = load(generation / "scores.json")
    with tempfile.TemporaryDirectory(prefix="ilm-qwen-analysis-") as temporary:
        parent = Path(temporary)
        (parent / "prompts.json").symlink_to((generation.parent / "prompts.json").resolve())
        check = parent / "generation"
        check.mkdir()
        for path in generation.glob("*.json"):
            if path.name != "scores.json":
                (check / path.name).symlink_to(path.resolve())
        with redirect_stdout(io.StringIO()):
            score(evaluation, check)
        if load(check / "scores.json") != expected:
            raise ValueError("Baseline scores differ from its authenticated raw responses")
    rows = [{**row, "case": row["id"], "condition": "normal", "complete_exact": row["exact"],
             "native_text_reference": True} for row in expected["results"]]
    return rows, {"scores_sha256": sha(generation / "scores.json"),
                  "report_sha256": sha(generation / "report.json"),
                  "model": expected["baseline"], "same_gpu_efficiency_measured": False}


def visual_summary(evaluation, pipeline, readings, references, output):
    scores, steps, provenance = authenticate_visual(evaluation, pipeline, readings)
    rows = []
    by_step = {}
    for step in sorted(steps):
        records = [row for row in scores["records"] if row["step"] == step]
        by_step[step] = records
        rows.extend(metric_rows(records, references, step))
    first, final = min(steps), max(steps)
    effects = []
    for paired in (False, True):
        final_values = qa_values(by_step[final], references, "normal", paired)
        for label, step, condition in (("final_minus_initial", first, "normal"),
                                       ("final_minus_question_blank", final, "question_blank"),
                                       ("final_minus_passage_blank", final, "passage_blank")):
            other = qa_values(by_step[step], references, condition, paired)
            effects.append({"comparison": label,
                            "metric": "qa_both_correct" if paired else "qa_complete_exact",
                            **clustered_interval(final_values - other)})
    books = [row for row in scores["results"] if row["kind"] == "book"]
    # Copies retain the exact original PNG bytes, including empty placeholders.
    inputs = {row["id"]: row for row in load(evaluation / "inputs/manifest.json")["inputs"]}
    example_rows = []
    for step, state in sorted(steps.items()):
        for case in EXAMPLES:
            source = state["run"] / case / "answer.png"
            record = next(row for row in by_step[step] if row["case"] == case)
            name = f"{case}/step_{step:07d}.png"
            path = output / "raw_examples" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
            if sha(path) != record["image_sha256"]:
                raise ValueError("Copied example differs from its generated pixels")
            input_path = evaluation / "inputs" / inputs[case]["image"]
            if sha(input_path) != inputs[case]["image_sha256"]:
                raise ValueError("Prospectively fixed example input changed")
            shutil.copyfile(input_path, path.parent / "input.png")
            example_rows.append({"case": case, "step": step, "file": "raw_examples/" + name,
                                 "image_sha256": record["image_sha256"],
                                 "input_sha256": inputs[case]["image_sha256"],
                                 "emitted_glyphs": record["emitted_glyphs"], "stopped": record["stopped"]})
    return {"mode": "visual_trajectory", "metrics": rows, "paired_effects": effects,
            "book_counts": books, "raw_examples": example_rows,
            "checkpoint_bindings": [{"step": step, **{key: value for key, value in row.items() if key != "run"}}
                                    for step, row in sorted(steps.items())], **provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/compact_visual_span_evaluation_v1")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "artifacts/compact_visual_span_pipeline_v1")
    parser.add_argument("--readings", type=Path)
    parser.add_argument("--baseline", type=Path, help="Summarize this completed native-text generation instead")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Hide CUDA for CPU-only analysis")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.baseline is None and args.readings is None:
        parser.error("Visual analysis requires the completed blind readings")
    if args.baseline is not None and args.readings is not None:
        parser.error("Select visual readings or the native-text baseline")
    prep, references = authenticate_registration(args.evaluation)
    # A temporary sibling prevents an interrupted analysis looking complete.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ilm-learning-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "complete"
        stage.mkdir()
        if args.baseline is not None:
            records, provenance = authenticate_baseline(args.evaluation, args.baseline)
            report = {"mode": "native_text_reference", "metrics": metric_rows(records, references, None),
                      "book_coherence_rated": False, **provenance}
        else:
            report = visual_summary(args.evaluation, args.pipeline, args.readings, references, stage)
        report.update(evaluation_preparation_sha256=sha(args.evaluation / "preparation.json"),
                      source_sha256=sha(Path(__file__)), bootstrap_seed=SEED, bootstrap_replicates=REPLICATES,
                      interval_method="pointwise percentile bootstrap of sixteen paired passage groups",
                      interval_limitations="Small development cohort; no simultaneous coverage or pretraining-exposure guarantee. Constant observations give a degenerate bootstrap interval, not proof of zero population error.",
                      book_interval_reported=False, book_interval_reason="Only two held-out works",
                      qualification_verdict="not assessed by this analysis",
                      trained_visual_language_result=report["mode"] == "visual_trajectory")
        save_json(stage / "summary.json", report)
        with (stage / "language_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "condition", "metric", "count", "cases", "rate", "ci95_low", "ci95_high"])
            writer.writeheader()
            for row in report["metrics"]:
                writer.writerow({key: row[key] for key in ("step", "condition", "metric", "count", "cases")} |
                                {"rate": row["mean"], "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1]})
        stage.rename(args.output)
    print(json.dumps({"mode": report["mode"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify raw answer images, prepare blind review, and score completed readings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.infer_compact_visual_span import check_output, read_image
from scripts.prepare_compact_visual_span_evaluation import normalized, order


def load(path):
    return json.loads(path.read_text())


def registered_inputs(evaluation):
    prep = load(evaluation / "preparation.json")
    manifest = evaluation / "inputs/manifest.json"
    if sha(manifest) != prep["input_manifest_sha256"]:
        raise ValueError("Registered prompt manifest changed")
    return prep, {row["id"]: row for row in load(manifest)["inputs"]}


def verified_answer(directory, record):
    raw, png = directory / "ink.npz", directory / "answer.png"
    if sha(raw) != record["raw_sha256"] or sha(png) != record["image_sha256"]:
        raise ValueError(f"Raw or displayed answer changed: {record['id']}")
    with np.load(raw, allow_pickle=False) as archive:
        tiles = torch.from_numpy(archive["ink"])
    if len(tiles) != record["emitted_glyphs"]:
        raise ValueError("Saved output has the wrong tile count")
    check_output({**record, "ink": tiles}, record["input_glyphs"])
    if len(tiles):
        if not torch.equal(read_image(png, len(tiles)), tiles):
            raise ValueError("Displayed answer differs from the raw glyph stream")
    else:
        with Image.open(png) as image:
            if image.size != (32, 32) or image.convert("L").getextrema() != (255, 255):
                raise ValueError("A zero-glyph response must display as a white placeholder")
    return png


def collect_run(run, registered):
    request_path = run / "request.json"
    report_path = run / "report.json"
    request, report = load(request_path), load(report_path)
    if (report["request_sha256"] != sha(request_path) or report["reference_inputs"] or report["output_repaired"]):
        raise ValueError("Inference report does not identify raw reference-free generation")
    observed = {row["id"]: row for row in report["records"]}
    ids = {row["id"] for row in request["inputs"]}
    normal = {name for name, row in registered.items() if row["condition"] == "normal"}
    if (len(observed) != len(report["records"]) or len(ids) != len(request["inputs"])
            or observed.keys() != ids or ids not in (normal, set(registered))):
        raise ValueError("Complete the registered input set or every normal prompt before review")
    rows = []
    for binding in request["inputs"]:
        key = binding["id"]
        expected, actual = registered[key], observed[key]
        if (binding["sha256"] != expected["image_sha256"] or binding["glyphs"] != expected["glyphs"]
                or actual["input_sha256"] != binding["sha256"] or actual["input_glyphs"] != binding["glyphs"]):
            raise ValueError(f"Inference input differs from registration: {key}")
        png = verified_answer(run / key, actual)
        opaque = order(request["checkpoint_sha256"] + "/" + key)[:20]
        rows.append((opaque, actual, expected["condition"], png))
    return request, report, rows


def review(evaluation, runs, output):
    _, registered = registered_inputs(evaluation)
    cohorts = [collect_run(run, registered) for run in runs]
    steps = {report["optimizer_updates"] for _, report, _ in cohorts}
    setting_keys = ("seed", "flow_steps", "maximum_glyphs", "device", "source_sha256")
    configurations = {json.dumps({key: request[key] for key in setting_keys}, sort_keys=True)
                      for request, _, _ in cohorts}
    if len(steps) != len(cohorts) or len(configurations) != 1:
        raise ValueError("Review requires distinct checkpoint steps with the same generation settings")
    output.mkdir(parents=True, exist_ok=False)
    pages = output / "pages"
    pages.mkdir()
    readings = {}
    mapping = {}
    for (request, report, rows), run in zip(cohorts, runs):
        for opaque, actual, condition, png in rows:
            if opaque in mapping:
                raise ValueError("Duplicate blind review image identifier")
            name = f"{opaque}.png"
            shutil.copyfile(png, pages / name)
            key = actual["id"]
            if key.startswith("book_") and condition == "normal":
                source = evaluation / "inputs" / registered[key]["image"]
                if sha(source) != registered[key]["image_sha256"]:
                    raise ValueError("Registered book prefix changed")
                shutil.copyfile(source, pages / f"{opaque}_prefix.png")
            readings[opaque] = {"text": None, "uncertain": None, "coherent_continuation": None}
            mapping[opaque] = {
                "case": key, "condition": condition, "step": report["optimizer_updates"],
                "checkpoint_sha256": request["checkpoint_sha256"], "run_report_sha256": sha(run / "report.json"),
                "image_sha256": actual["image_sha256"], "stopped": actual["stopped"],
                "emitted_glyphs": actual["emitted_glyphs"], "elapsed_seconds": actual["elapsed_seconds"],
            }
    save_json(output / "mapping.json", {"preparation_sha256": sha(evaluation / "preparation.json"), "rows": mapping})
    save_json(output / "readings_template.json", {"evaluator": None, "readings": dict(sorted(readings.items()))})
    print(json.dumps({"actual_answer_images": len(mapping), "checkpoint_steps": sorted(steps), "references_opened": False}))


def score(evaluation, review_directory, reading_path):
    prep, registered = registered_inputs(evaluation)
    mapping_path = review_directory / "mapping.json"
    mapping = load(mapping_path)
    review_receipt = load(reading_path)
    evaluator, readings = review_receipt["evaluator"], review_receipt["readings"]
    if not isinstance(evaluator, str) or not evaluator.strip():
        raise ValueError("Identify the reviewer or the transcription method")
    if (mapping["preparation_sha256"] != sha(evaluation / "preparation.json")
            or readings.keys() != mapping["rows"].keys()):
        raise ValueError("Review must cover every registered generated image")
    covered = defaultdict(set)
    for row in mapping["rows"].values():
        key, step = row["case"], row["step"]
        if key in covered[step] or registered[key]["condition"] != row["condition"]:
            raise ValueError("Review repeats or relabels a registered case")
        covered[step].add(key)
    normal = {key for key, row in registered.items() if row["condition"] == "normal"}
    if not covered or any(ids not in (normal, set(registered)) for ids in covered.values()):
        raise ValueError("A reviewed checkpoint is missing registered cases")
    ref_path = evaluation / "references.json"
    if sha(ref_path) != prep["references_sha256"]:
        raise ValueError("Registered evaluation references changed")
    references = {row["id"]: row for row in load(ref_path)["reference_rows"]}
    measurements, groups, pairs = [], defaultdict(list), defaultdict(list)
    for opaque, row in mapping["rows"].items():
        if sha(review_directory / "pages" / f"{opaque}.png") != row["image_sha256"]:
            raise ValueError(f"Reviewed answer changed: {opaque}")
        reading = readings[opaque]
        if type(reading["uncertain"]) is not bool or not isinstance(reading["text"], str):
            raise ValueError(f"Supply a complete transcription or mark it uncertain: {opaque}")
        reference = references[row["case"].removesuffix("_" + row["condition"])]
        coherent = reading["coherent_continuation"]
        if reference["kind"] == "book" and row["condition"] == "normal" and type(coherent) is not bool:
            raise ValueError(f"Supply a coherence judgment using the visible book prefix: {opaque}")
        clear = not reading["uncertain"] and bool(normalized(reading["text"])) and row["emitted_glyphs"] > 0
        exact = clear and row["stopped"] and normalized(reading["text"]) == normalized(reference["answer"])
        measured = {**row, "kind": reference["kind"], "reading": reading, "readable": clear,
                    "complete_exact": exact, "coherent_complete": clear and row["stopped"] and coherent is True}
        measurements.append(measured)
        groups[row["step"], reference["kind"], row["condition"]].append(measured)
        if reference["kind"] == "qa":
            pairs[row["step"], row["condition"], reference["paired_group"]].append(measured)
    results = []
    for (step, kind, condition), rows in sorted(groups.items()):
        paired = None
        if kind == "qa":
            pair_rows = [pairs[step, condition, index] for index in range(1, prep["paired_qa_groups"] + 1)]
            if any(len(pair) != 2 for pair in pair_rows):
                raise ValueError("A paired context is missing an answer")
            paired = sum(all(r["complete_exact"] for r in pair) and pair[0]["image_sha256"] != pair[1]["image_sha256"]
                         for pair in pair_rows)
        results.append({"step": step, "kind": kind, "condition": condition, "cases": len(rows),
                        "readable": sum(r["readable"] for r in rows), "stopped": sum(r["stopped"] for r in rows),
                        "complete_exact": sum(r["complete_exact"] for r in rows), "paired_both_correct": paired,
                        "coherent_complete": sum(r["coherent_complete"] for r in rows) if kind == "book" and condition == "normal" else None})
    destination = review_directory / "scores.json"
    if destination.exists():
        raise FileExistsError(destination)
    save_json(destination, {"results": results, "records": measurements, "evaluator": evaluator,
                            "review_mapping_sha256": sha(mapping_path),
                            "readings_sha256": sha(reading_path), "references_sha256": sha(ref_path)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/compact_visual_span_evaluation_v1")
    stages = parser.add_subparsers(dest="stage", required=True)
    make = stages.add_parser("prepare-review")
    make.add_argument("--runs", type=Path, nargs="+", required=True)
    make.add_argument("--output", type=Path, required=True)
    grade = stages.add_parser("score")
    grade.add_argument("--review", type=Path, required=True)
    grade.add_argument("--readings", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare-review":
        review(args.evaluation, args.runs, args.output)
    else:
        score(args.evaluation, args.review, args.readings)

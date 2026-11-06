#!/usr/bin/env python3
"""Separate CPU-only OCR calibration and observation for paired-image outputs.

This program is never imported by the deployed generator. Reference OCR
calibration does not guarantee recognition of generated typography.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.evaluate_image_edit_baseline import engine, normalize, recognize


def implementation():
    return {name: sha(ROOT / name) for name in
        ("scripts/score_paired_writing.py", "scripts/evaluate_image_edit_baseline.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "score"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--generations", type=Path,
        help="JSON list of {id, path, sha256}; contains no repaired images")
    parser.add_argument("--task", choices=("copy", "qa"), default="copy")
    parser.add_argument("--train-limit", type=int, default=32)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicitly hide CUDA for the separate offline evaluator")
    all_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    train = [r for r in all_rows if r["task"] == args.task and r["split"] == "train"][:args.train_limit]
    rows = train + [r for r in all_rows if r["task"] == args.task and r["split"] == "validation"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique nonempty evaluation cohort required")
    binary, source = engine()
    if args.stage == "calibrate":
        selected = [{**r, "observed_path": (args.manifest.parent / r["target"]).resolve(),
            "observed_sha256": r["target_sha256"]} for r in rows]
        calibration = None
    else:
        if args.calibration is None or args.generations is None:
            raise ValueError("score requires both reference calibration and an output index")
        calibration = json.loads(args.calibration.read_text())
        if (calibration["stage"] != "calibrate" or calibration["engine"] != source
                or calibration["implementation"] != implementation()
                or calibration["manifest_sha256"] != sha(args.manifest)
                or calibration["cohort"] != [r["id"] for r in rows]):
            raise ValueError("owning unchanged reference calibration required")
        supplied = json.loads(args.generations.read_text())
        by_id = {r["id"]: r for r in rows}
        if not supplied or len({r["id"] for r in supplied}) != len(supplied):
            raise ValueError("unique nonempty generated-output index required")
        selected = []
        for result in supplied:
            original = by_id[result["id"]]
            selected.append({**original,
                "observed_path": (args.generations.parent / result["path"]).resolve(),
                "observed_sha256": result["sha256"]})

    def observe(row):
        if sha(row["observed_path"]) != row["observed_sha256"]:
            raise ValueError("observed image changed: " + row["id"])
        raw, diagnostic = recognize(binary, row["observed_path"])
        text = normalize(raw, row["language"])
        return dict(id=row["id"], split=row["split"], language=row["language"],
            expected=row["reference_text"], raw_transcript=raw, normalized=text,
            ocr_stderr=diagnostic, png_sha256=row["observed_sha256"],
            ocr_exact=text == normalize(row["reference_text"], row["language"]))

    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as workers:
        observed = list(workers.map(observe, selected))
    summary = {split: dict(cases=sum(r["split"] == split for r in observed),
        ocr_exact=sum(r["split"] == split and r["ocr_exact"] for r in observed))
        for split in ("train", "validation")}
    report = dict(stage=args.stage, engine=source, implementation=implementation(),
        manifest_sha256=sha(args.manifest), cohort=[r["id"] for r in rows], rows=observed,
        summary=summary, complete_cohort={r["id"] for r in selected} == {r["id"] for r in rows},
        generated_images_opened=args.stage == "score", neural_generator_loaded=False,
        reference_calibration_sha256=sha(args.calibration) if calibration else None,
        reference_calibration_summary=calibration["summary"] if calibration else None,
        normalization="NFC; English whitespace collapsed; Chinese whitespace removed; case and punctuation retained",
        limitations="Exploratory OCR observations; generated typography and uncertain readings require raw-image review",
        language_quality_qualified=False, general_ilm_goal_complete=False)
    save_json(args.output / "report.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

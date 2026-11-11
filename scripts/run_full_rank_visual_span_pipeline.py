#!/usr/bin/env python3
"""Serialize the controlled wider-writer fit and its unchanged evaluation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.prepare_full_rank_visual_span import validate_registration
from scripts.run_compact_visual_span_pipeline import Pipeline, load


def execute(pipeline, args, registered):
    fit_script = ROOT / "scripts/train_full_rank_visual_span.py"
    command = [fit_script, "--teacher-run", registered["teacher_run"], "--run", args.fit,
               "--registration", args.registration]
    pipeline.resumable_stage("fit", command, args.fit, args.fit / "report.json")
    fit = load(args.fit / "report.json")
    snapshots = sorted(args.fit.glob("image_heads_step_*.pt"))
    if (len(snapshots) != 5 or snapshots[0].name != "image_heads_step_0000000.pt"
            or snapshots[-1].name != fit["final_image_heads"]
            or sha(snapshots[-1]) != fit["final_image_heads_sha256"]
            or fit["updates"] != 8172 or fit["sequence_exposures"] != registered["training_sequences"]):
        raise ValueError("The controlled fit does not contain the complete registered pass and five snapshots")
    evaluation = Path(registered["evaluation"])
    manifest = evaluation / "inputs/manifest.json"
    expected = load(evaluation / "preparation.json")["input_manifest_sha256"]
    runs = []
    for index, snapshot in enumerate(snapshots):
        validate_registration(args.registration)
        if sha(manifest) != expected:
            raise ValueError("Registered evaluation input manifest changed")
        output = args.pipeline / snapshot.stem
        runs.append(output)
        if not (output / "report.json").exists():
            command = [ROOT / "scripts/infer_full_rank_visual_span.py", "--manifest", manifest,
                       "--checkpoint", snapshot, "--output", output]
            if index not in {0, 4}:
                command.append("--normal-only")
            code, log = pipeline.child(snapshot.stem, command)
            if code or not (output / "report.json").is_file():
                raise RuntimeError(f"Evaluation failed; inspect {log.name}")
        if load(output / "request.json")["checkpoint_sha256"] != sha(snapshot):
            raise ValueError("Generated answers use a different checkpoint")
    review = args.pipeline / "review"
    if not (review / "mapping.json").is_file():
        command = [ROOT / "scripts/review_compact_visual_span.py", "--evaluation", evaluation,
                   "prepare-review", "--output", review, "--runs", *runs]
        code, log = pipeline.child("review", command, gpu=False)
        if code or not (review / "mapping.json").is_file():
            raise RuntimeError(f"Review preparation failed; inspect {log.name}")
    pipeline.status("awaiting_blind_readings", review=str(review), training_updates=fit["updates"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=ROOT / "artifacts/compact_visual_span_full_rank_v1/registration.json")
    parser.add_argument("--fit", type=Path, default=ROOT / "artifacts/compact_visual_span_full_rank_fit_v1")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "artifacts/compact_visual_span_full_rank_pipeline_v1")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        parser.error("Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0")
    args.registration, args.fit, args.pipeline = (p.resolve() for p in (args.registration, args.fit, args.pipeline))
    registered = validate_registration(args.registration)
    identity = {"registration_sha256": sha(args.registration), "fit": str(args.fit),
                "sources_sha256": registered["sources_sha256"]}
    pipeline = Pipeline(args.pipeline, identity)
    with pipeline.lock:
        fcntl.flock(pipeline.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = args.pipeline / "request.json"
        if request.exists() and load(request) != identity:
            raise ValueError("Pipeline inputs or registered sources changed")
        save_json(request, identity)
        def terminate(_signum, _frame):
            raise SystemExit("Pipeline termination requested")
        signal.signal(signal.SIGTERM, terminate)
        try:
            execute(pipeline, args, registered)
        except BaseException as error:
            pipeline.status("failed", reason=repr(error))
            raise
        finally:
            status = args.pipeline / "status.json"
            if status.is_file():
                save_json(status, {**load(status), "pid": None})


if __name__ == "__main__":
    main()

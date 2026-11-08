#!/usr/bin/env python3
"""Prepare and sequentially execute complete native QA training and assessment."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.qa_learning import (CONDITION, CONTROLS, INITIAL, INITIAL_SHA,
    PAGES, SNAPSHOT, SOURCE, dataset, digest, identity, schedule)


def prepare(run):
    code = identity(); train, held = dataset()
    preparation = json.loads((CONTROLS / "preparation.json").read_text())
    if preparation["manifest_sha256"] != sha(PAGES / "pairs.jsonl"):
        raise ValueError("source preparation and actual manifest differ")
    run.mkdir(parents=True, exist_ok=False)
    indices = {}
    for name, originals in dict(normal=["normal_seen_inputs.json", "normal_held_inputs.json"],
            question_erased=["question_erased_held_inputs.json"],
            passage_erased=["passage_erased_held_inputs.json"]).items():
        rows = []
        for original in originals:
            path = CONTROLS / original
            if sha(path) != preparation["index_files_sha256"][original]:
                raise ValueError("complete prepared input index differs")
            for row in json.loads(path.read_text()):
                if set(row) != {"id", "input", "input_sha256", "seed"} or sha(CONTROLS / row["input"]) != row["input_sha256"]:
                    raise ValueError("input-only image binding differs")
                rows.append({**row, "input": str((CONTROLS / row["input"]).resolve())})
        expected = train + held if name == "normal" else held
        if [r["id"] for r in rows] != [r["id"] for r in expected]:
            raise ValueError("complete prospective evaluation cohort required")
        path = run / (name + "_inputs.json"); save_json(path, rows); indices[name] = sha(path)
    if sha(INITIAL) != INITIAL_SHA:
        raise ValueError("prospectively fixed initial checkpoint differs")
    save_json(run / "schedule.json", schedule(train))
    record = dict(code_sha256=code, initial_checkpoint_sha256=INITIAL_SHA,
        training_ids=[r["id"] for r in train], held_ids=[r["id"] for r in held],
        schedule_sha256=digest(schedule(train)), manifest_sha256=sha(PAGES / "pairs.jsonl"),
        original_source_sha256=sha(SOURCE), control_preparation_sha256=sha(CONTROLS / "preparation.json"),
        index_files_sha256=indices, complete_qa_assessment_calls=256, final_copying_calls=64,
        optimizer_updates=128, pretrained_initial_selected_before_QA=True,
        general_ilm_goal_complete=False)
    save_json(run / "preparation.json", record)
    print(json.dumps(record, indent=2), flush=True)


def execute_run(run, resume):
    code = identity(); prepared = json.loads((run / "preparation.json").read_text())
    if code != prepared["code_sha256"] or sha(INITIAL) != INITIAL_SHA or sha(PAGES / "pairs.jsonl") != prepared["manifest_sha256"] or sha(SOURCE) != prepared["original_source_sha256"]:
        raise ValueError("exact prepared source/data/initial weights required")
    for name, digest_ in prepared["index_files_sha256"].items():
        if sha(run / (name + "_inputs.json")) != digest_: raise ValueError("registered input index changed")
    if (run / "runtime.json").exists() and not resume:
        raise ValueError("preserve existing runtime; explicit unchanged --resume required")
    completed = json.loads((run / "completed_stages.json").read_text()) if (run / "completed_stages.json").exists() else {}
    phase, child = "starting", None; started = time.time()

    def status(**extra):
        save_json(run / "runtime.json", dict(phase=phase, coordinator_pid=None if phase in ("failed", "ready_for_raw_review") else os.getpid(),
            child_pid=None if child is None or child.poll() is not None else child.pid,
            started_unix=started, code_sha256=code, gpu=0, novnc_url=None,
            general_ilm_goal_complete=False, **extra))

    def interrupted(signum, frame): raise InterruptedError("QA coordinator signal " + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(signum, interrupted)

    def stage(name, script, arguments, gpu, primary):
        nonlocal phase, child
        if identity() != code: raise ValueError("registered live implementation changed")
        primary = Path(primary)
        if name in completed:
            if sha(primary) != completed[name]["primary_sha256"]:
                raise ValueError("completed stage receipt changed")
            return
        phase = name
        arguments = list(map(str, arguments))
        if name == "training" and (run / "training/checkpoint.pt").exists():
            if not resume: raise ValueError("explicit resume required")
            arguments.append("--resume")
        command = [str(Path.home() / ".local/bin/with-current-nvidia"), "env",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=" + ("0" if gpu else ""),
            "PYTHONNOUSERSITE=1", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=4",
            "PYTHONPATH=.", str(ROOT.parent / "LocalLLM/.local/image-generation/venv/bin/python"), script, *arguments]
        number = 1 + len(list(run.glob(name + ".attempt_*.log")))
        with (run / f"{name}.attempt_{number:03}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(child_command=command)
            try:
                if child.wait(): raise RuntimeError(name + " failed; no automatic restart or quality selection")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        if identity() != code: raise ValueError("bound stage implementation changed")
        completed[name] = dict(primary=str(primary.relative_to(run)), primary_sha256=sha(primary), command=command)
        save_json(run / "completed_stages.json", completed)

    try:
        status()
        stage("training", "scripts/train_native_paired_qa.py", ["--output", run / "training"], True, run / "training/report.json")
        trained = json.loads((run / "training/report.json").read_text())
        if trained["updates"] != 128 or trained["checkpoint_sha256"] != sha(run / "training/checkpoint.pt"):
            raise ValueError("all 128 updates and unchanged final checkpoint required")
        for arm, checkpoint in (("final", run / "training/checkpoint.pt"), ("initial", INITIAL)):
            for variant in ("normal", "question_erased", "passage_erased"):
                name = arm + "_" + variant
                stage("infer_" + name, "scripts/infer_paired_image_cohort_v2.py",
                    ["--snapshot", SNAPSHOT, "--condition", CONDITION, "--checkpoint", checkpoint,
                     "--inputs", run / (variant + "_inputs.json"), "--output", run / (name + "_inference")], True,
                    run / (name + "_inference/report.json"))
                stage("score_" + name, "scripts/score_paired_writing.py",
                    ["--stage", "score", "--task", "qa", "--train-limit", "32", "--manifest", PAGES / "pairs.jsonl",
                     "--calibration", CONTROLS / "reference_ocr/report.json", "--generations", run / (name + "_inference/outputs.json"),
                     "--output", run / (name + "_ocr")], False, run / (name + "_ocr/report.json"))
        stage("infer_final_copy", "scripts/infer_paired_image_cohort_v2.py",
            ["--snapshot", SNAPSHOT, "--condition", CONDITION, "--checkpoint", run / "training/checkpoint.pt",
             "--inputs", ROOT / "artifacts/image_edit_learning/full_strip_inputs.json", "--output", run / "final_copy_inference"], True,
            run / "final_copy_inference/report.json")
        stage("score_final_copy", "scripts/score_paired_writing.py",
            ["--stage", "score", "--manifest", ROOT / "artifacts/image_edit_learning/strip_pages/pairs.jsonl",
             "--calibration", ROOT / "artifacts/balanced_autonomous_writer_v1/reference_ocr/report.json",
             "--generations", run / "final_copy_inference/outputs.json", "--output", run / "final_copy_ocr"], False,
            run / "final_copy_ocr/report.json")
        stage("assessment", "scripts/assess_native_paired_qa.py", ["--run", run], False, run / "assessment.json")
        phase = "ready_for_raw_review"; status(owned_children_exited=True)
    except BaseException as error:
        phase = "failed"; status(reason=str(error), owned_children_exited=child is None or child.poll() is not None); raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "run"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(); run = args.output.resolve()
    prepare(run) if args.stage == "prepare" else execute_run(run, args.resume)

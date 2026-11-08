#!/usr/bin/env python3
"""Run one registered sequential two-arm experiment and complete CPU scoring.

No automatic retries or checkpoint selection. Actual raw review and tensor
authentication are still required after this worker finishes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    code_paths = ("scripts/run_ink_mass_comparison.py", "scripts/train_ink_mass_writer.py",
        "ilm/image_edit/ink_mass.py", "references/ink_mass_writing_protocol.md",
        "scripts/infer_paired_image_cohort_v2.py", "scripts/score_paired_writing.py",
        "scripts/evaluate_image_edit_baseline.py", "ilm/image_edit/baseline.py",
        "ilm/image_edit/paired.py", "ilm/image_edit/recognition.py", "ilm/image_edit/autonomous.py")
    if subprocess.check_output(["git", "status", "--porcelain", "--", *code_paths], cwd=ROOT, text=True):
        raise ValueError("commit the exact experiment sources first")
    code = {p: sha(ROOT / p) for p in code_paths}
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    phase, child = "starting", None

    def status(**extra):
        save_json(run / "runtime.json", dict(phase=phase,
            coordinator_pid=None if phase in ("failed", "ready_for_export_and_raw_review") else os.getpid(),
            child_pid=None if child is None or child.poll() is not None else child.pid,
            started_unix=started, code_sha256=code, gpu=0, novnc_url=None,
            general_ilm_goal_complete=False, **extra))

    def interrupted(signum, frame):
        raise InterruptedError("comparison worker received signal " + str(signum))
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)

    def execute(name, script, arguments, gpu):
        nonlocal phase, child
        if any(sha(ROOT / p) != v for p, v in code.items()):
            raise ValueError("actual experiment sources changed")
        phase = name
        python = ROOT.parent / "LocalLLM/.local/image-generation/venv/bin/python"
        command = [str(Path.home() / ".local/bin/with-current-nvidia"), "env",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=" + ("0" if gpu else ""),
            "PYTHONNOUSERSITE=1", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=4",
            "PYTHONPATH=.", str(python), script, *map(str, arguments)]
        with (run / (name + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(child_command=command)
            try:
                if child.wait():
                    raise RuntimeError(name + " failed; preserve all partial artifacts")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        if any(sha(ROOT / p) != v for p, v in code.items()):
            raise ValueError("bound experiment implementation changed")

    snapshot = Path("artifacts/external/image_edit_learning/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d")
    condition = Path("artifacts/image_edit_baseline/condition/condition.pt")
    manifest = Path("artifacts/image_edit_learning/strip_pages/pairs.jsonl")
    inputs = Path("artifacts/image_edit_learning/full_strip_inputs.json")
    calibration = Path("artifacts/balanced_autonomous_writer_v1/reference_ocr/report.json")
    common = ["--snapshot", snapshot, "--condition", condition, "--manifest", manifest, "--inputs", inputs,
        "--teacher", "artifacts/external/writing_reader/hub/models--stepfun-ai--GOT-OCR-2.0-hf/snapshots/d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
        "--admission", "artifacts/recognition_guided_writing_v1/calibration_r2/report.json",
        "--initial", "artifacts/balanced_autonomous_writer_v1/training/checkpoint.pt",
        "--baseline", "artifacts/balanced_autonomous_writer_v1/inference/outputs.json",
        "--audit", "references/evidence/writing_physical_loss_audit.json"]
    try:
        status()
        for arm in ("control", "ink_mass"):
            execute("train_" + arm, "scripts/train_ink_mass_writer.py",
                [*common, "--arm", arm, "--output", run / (arm + "_training")], True)
            trained = json.loads((run / (arm + "_training/report.json")).read_text())
            if (trained["updates"] != 32 or trained["stage"] != "completed"
                    or trained["checkpoint_sha256"] != sha(run / (arm + "_training/checkpoint.pt"))):
                raise ValueError("complete registered training required")
        scores = {}
        for arm in ("control", "ink_mass"):
            execute("infer_" + arm, "scripts/infer_paired_image_cohort_v2.py",
                ["--snapshot", snapshot, "--condition", condition, "--inputs", inputs,
                 "--checkpoint", run / (arm + "_training/checkpoint.pt"), "--output", run / (arm + "_inference")], True)
            execute("score_" + arm, "scripts/score_paired_writing.py",
                ["--stage", "score", "--manifest", manifest, "--calibration", calibration,
                 "--generations", run / (arm + "_inference/outputs.json"), "--output", run / (arm + "_ocr")], False)
            score = json.loads((run / (arm + "_ocr/report.json")).read_text())
            if len(score["rows"]) != 64 or not score["complete_cohort"]:
                raise ValueError("complete fixed-cohort scoring required")
            scores[arm] = dict(summary=score["summary"], language_summary={
                split + "/" + language: sum(r["ocr_exact"] for r in score["rows"] if r["split"] == split and r["language"] == language)
                for split in ("train", "validation") for language in ("zh", "en")},
                report_sha256=sha(run / (arm + "_ocr/report.json")))
        a, b = scores["control"], scores["ink_mass"]
        gates = dict(Chinese=b["language_summary"]["validation/zh"] >= max(6,a["language_summary"]["validation/zh"]+2),
            seen=b["summary"]["train"]["ocr_exact"] >= max(13,a["summary"]["train"]["ocr_exact"]),
            English=b["language_summary"]["validation/en"] >= max(13,a["language_summary"]["validation/en"]))
        save_json(run / "comparison.json", dict(scores=scores, exploratory_gates=gates,
            exploratory_rule_met=all(gates.values()), raw_review_completed=False,
            tensor_authentication_pending=True, writing_qualified=False, semantic_QA_evaluated=False,
            general_ilm_goal_complete=False))
        phase = "ready_for_export_and_raw_review"
        status(owned_children_exited=True)
    except BaseException as error:
        phase = "failed"; status(reason=str(error), owned_children_exited=child is None or child.poll() is not None)
        raise


if __name__ == "__main__":
    main()

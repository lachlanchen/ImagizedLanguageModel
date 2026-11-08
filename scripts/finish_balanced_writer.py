#!/usr/bin/env python3
"""Wait for the exact owned trainer, then assess its final checkpoint once."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, save_json, sha


def owner(pid):
    path = Path("/proc") / str(pid)
    try:
        command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        start = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None
    return dict(pid=pid, start_ticks=start, command=command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "snapshot", "condition", "manifest", "inputs", "pilot", "calibration"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--training-pid", type=int, required=True)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError("completion worker received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    run = args.run.resolve()
    if not run.is_dir(): raise ValueError("existing owned training run required")
    lock = (run / "assessment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run / "assessment.json"
    if status_path.exists():
        raise ValueError("preserve earlier assessment state; do not automatically rerun model outputs")
    paths = ("scripts/finish_balanced_writer.py", "scripts/infer_paired_image_cohort_v2.py",
        "scripts/export_balanced_autonomous_writer.py", "scripts/score_paired_writing.py",
        "scripts/evaluate_image_edit_baseline.py", "ilm/image_edit/baseline.py", "ilm/image_edit/paired.py")
    code = {name: sha(ROOT / name) for name in paths}
    if subprocess.check_output(["git", "status", "--porcelain", "--", *code], cwd=ROOT, text=True):
        raise ValueError("commit the exact completion implementation before launch")
    identity = owner(args.training_pid)
    if identity is not None and ("scripts/train_balanced_autonomous_writer.py" not in identity["command"]
            or str(args.run / "training") not in identity["command"]):
        raise ValueError("specified live PID is not this run's trainer")
    started = time.time(); phase = "waiting_training"

    def status(**extra):
        save_json(status_path, dict(phase=phase, coordinator_pid=os.getpid(), training_owner=identity,
            implementation=code, started_unix=started, novnc_url=None, **extra))

    def unchanged():
        if any(sha(ROOT / name) != digest for name, digest in code.items()):
            raise ValueError("bound completion implementation changed")

    def execute(name, script, arguments, *, gpu):
        nonlocal phase
        unchanged(); phase = name
        python = ROOT.parent / "LocalLLM/.local/image-generation/venv/bin/python"
        command = [str(Path.home() / ".local/bin/with-current-nvidia"), "env",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=" + ("0" if gpu else ""),
            "PYTHONNOUSERSITE=1", "OMP_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=4",
            "PYTHONPATH=.", str(python), script, *map(str, arguments)]
        with (run / (name + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(child_pid=child.pid, child_command=command)
            try:
                result = child.wait()
                if result: raise RuntimeError(name + " exited with status " + str(result))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        unchanged()

    try:
        status(child_pid=None)
        while identity is not None and owner(args.training_pid) == identity:
            state = json.loads((run / "training/status.json").read_text())
            status(child_pid=None, training_status=state)
            time.sleep(15)
        state = json.loads((run / "training/status.json").read_text())
        report = json.loads((run / "training/report.json").read_text()) if (run / "training/report.json").exists() else None
        if (state != {"stage": "completed", "pid": None} or not report or report["stage"] != "completed"
                or report["updates"] != 128 or sha(run / "training/checkpoint.pt") != report["checkpoint_sha256"]):
            raise RuntimeError("owned training is terminal without a verified complete 128-update checkpoint")
        execute("inference", "scripts/infer_paired_image_cohort_v2.py", ["--snapshot", args.snapshot,
            "--condition", args.condition, "--inputs", args.inputs, "--checkpoint", run / "training/checkpoint.pt",
            "--output", run / "inference"], gpu=True)
        reference = run / "reference_ocr"; reference.mkdir(exist_ok=False)
        # Only a small immutable calibration report is copied, never model weights.
        (reference / "report.json").write_bytes(args.calibration.read_bytes())
        execute("output_ocr", "scripts/score_paired_writing.py", ["--stage", "score", "--manifest", args.manifest,
            "--calibration", reference / "report.json", "--generations", run / "inference/outputs.json",
            "--output", run / "output_ocr"], gpu=False)
        execute("export", "scripts/export_balanced_autonomous_writer.py", ["--run", run, "--manifest", args.manifest,
            "--inputs", args.inputs, "--pilot", args.pilot, "--output", run / "export"], gpu=False)
        result = json.loads((run / "export/result.json").read_text())
        phase = "ready_for_raw_review"
        status(child_pid=None, complete_generations=result["complete_outputs"], summary=result["summary"],
            raw_review_completed=False, writing_qualification=False, goal_complete=False)
        print("Complete final checkpoint exported; inspect every raw page before qualification.", flush=True)
    except BaseException as error:
        failed_phase = phase; phase = "failed"
        status(child_pid=None, failed_phase=failed_phase, exception=type(error).__name__, reason=str(error))
        raise


if __name__ == "__main__":
    main()

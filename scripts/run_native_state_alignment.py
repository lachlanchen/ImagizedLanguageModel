#!/usr/bin/env python3
"""Run the fixed reader experiment sequentially and retain live child identity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from ilm.image_edit.baseline import save_json, sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_state_alignment"))
    p.add_argument("--python", default="artifacts/external/native_unified_source/runtime/bin/python")
    p.add_argument("--resume", type=Path)
    p.add_argument("--training-output", type=Path)
    p.add_argument("--after-output", type=Path)
    p.add_argument("--score-output", type=Path, default=Path("references/evidence/native_state_alignment_result"))
    args = p.parse_args()
    learning = args.training_output or args.root / "learning_v1"
    after = args.after_output or args.root / "after_v1"
    status = args.root / "controller_status.json"
    args.root.mkdir(parents=True, exist_ok=True)
    script = "scripts/train_native_state_alignment.py"
    common = ["--prepared", str(args.root / "prepared_v1"), "--cache", str(args.root / "cache_v1")]
    stages = [("cache", [args.python, script, "--stage", "cache", *common], args.root / "cache_v1"),
        ("preflight", [args.python, script, "--stage", "preflight", *common], args.root / "preflight_v1"),
        ("copy_before", [args.python, script, "--stage", "evaluate", "--inputs", str(args.root / "assessment_v1/copy_inputs.json")], args.root / "copy_before_v1")]
    training_command = [args.python, script, "--stage", "train", *common]
    if args.resume:
        if args.training_output is None or args.after_output is None:
            raise ValueError("resume requires fresh explicit training and after output paths")
        training_command = [args.python, "scripts/resume_native_state_alignment.py", *common, "--resume", str(args.resume)]
    stages.extend((("train", training_command, learning),
        ("after", [args.python, script, "--stage", "evaluate", "--inputs", str(args.root / "assessment_v1/inputs.json"), "--checkpoint", str(learning / "checkpoint.pt")], after),
        ("source_replay_complete", [args.python, "scripts/check_native_state_source.py", "--inputs", str(args.root / "assessment_v1/en_inputs.json")], args.root / "source_replay_complete_v1"),
        ("score", [args.python, "scripts/score_native_state_alignment.py", "--root", str(args.root), "--learning", str(learning), "--after", str(after)], args.score_output)))
    child = None
    try:
        for stage, command, output in stages:
            if output.exists():
                if stage == "score":
                    result = read(output / "result.json")
                    if result["training"]["checkpoint_sha256"] != sha(learning / "checkpoint.pt"):
                        raise ValueError("existing score belongs to another model")
                else:
                    state = read(output / "status.json")
                    if state["state"] != "completed":
                        pid = state.get("pid")
                        live = bool(pid and Path(f"/proc/{pid}/cmdline").exists())
                        raise RuntimeError(f"preserve existing {stage}: state={state['state']} pid={pid} live={live}; attach or use explicit fresh resume paths")
                    report = read(output / ("manifest.json" if stage == "cache" else "report.json"))
                    from train_native_state_alignment import binding
                    actual = report.get("implementation", report.get("metadata", {}).get("implementation"))
                    if actual != binding():
                        raise ValueError("existing stage code binding differs: " + stage)
                print("reuse completed", stage, flush=True)
                continue
            env = dict(os.environ, PYTHONPATH=".", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                CUDA_VISIBLE_DEVICES="" if stage == "score" else "0")
            with (args.root / (stage + ".controller.log")).open("a") as log:
                child = subprocess.Popen([*command, "--output", str(output)], env=env, stdout=log, stderr=subprocess.STDOUT)
                print("started", stage, "pid", child.pid, flush=True)
                while child.poll() is None:
                    progress_path = output / "status.json"
                    try:
                        progress = read(progress_path) if progress_path.exists() else None
                    except json.JSONDecodeError:
                        progress = None
                    save_json(status, dict(state="running", stage=stage, controller_pid=os.getpid(), child_pid=child.pid,
                        output=str(output), progress=progress, goal_complete=False))
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"{stage} failed with exit {child.returncode}; inspect retained stage log")
                print("completed", stage, flush=True)
                child = None
        save_json(status, dict(state="completed", controller_pid=None, child_pid=None, learning=str(learning),
            after=str(after), score_output=str(args.score_output), goal_complete=False))
    except BaseException as error:
        # Preserve any independently live child; an observation/controller
        # failure is not permission to start a replacement optimizer.
        live_pid = child.pid if child is not None and child.poll() is None else None
        save_json(status, dict(state="failed", controller_pid=None, child_pid=live_pid, error=repr(error),
            traceback=traceback.format_exc(), goal_complete=False))
        raise


if __name__ == "__main__":
    main()

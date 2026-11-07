#!/usr/bin/env python3
"""Run the declared attention learner sequentially, retaining child identity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from ilm.image_edit.baseline import save_json, sha
from train_native_reader_attention import binding


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_reader_attention"))
    p.add_argument("--python", default="artifacts/external/native_unified_source/runtime/bin/python")
    p.add_argument("--resume", type=Path)
    p.add_argument("--learning", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--score-output", type=Path, default=Path("references/evidence/native_reader_attention_result"))
    args = p.parse_args()
    learning = args.learning or args.root / "learning_v1"
    after = args.after or args.root / "after_v1"
    if args.resume and (args.learning is None or args.after is None):
        raise ValueError("resume requires fresh learning and assessment paths")
    source = "scripts/train_native_reader_attention.py"
    common = ["--cache", str(args.root / "cache_v1")]
    training = [args.python, source, "--stage", "train", *common]
    if args.resume:
        training.extend(("--resume", str(args.resume)))
    stages = (("cache", [args.python, source, "--stage", "cache"], args.root / "cache_v1"),
        ("preflight", [args.python, source, "--stage", "preflight", *common], args.root / "preflight_v1"),
        ("train", training, learning),
        ("after", [args.python, source, "--stage", "evaluate", "--checkpoint", str(learning / "checkpoint.pt")], after),
        ("score", [args.python, "scripts/score_native_reader_attention.py", "--root", str(args.root), "--learning", str(learning), "--after", str(after)], args.score_output))
    status = args.root / "controller_status.json"
    args.root.mkdir(parents=True, exist_ok=True)
    child = None
    try:
        for stage, command, output in stages:
            if output.exists():
                if stage == "score":
                    if read(output / "result.json")["training"]["checkpoint_sha256"] != sha(learning / "checkpoint.pt"):
                        raise ValueError("existing score belongs to another checkpoint")
                else:
                    state = read(output / "status.json")
                    if state["state"] != "completed":
                        pid = state.get("pid")
                        raise RuntimeError(f"preserve {stage}: {state['state']} pid={pid} live={bool(pid and Path(f'/proc/{pid}/cmdline').exists())}")
                    report = read(output / ("manifest.json" if stage == "cache" else "report.json"))
                    if report.get("implementation", report.get("metadata", {}).get("implementation")) != binding():
                        raise ValueError("completed stage code binding changed: " + stage)
                print("reuse completed", stage, flush=True)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if stage == "score" else "0", PYTHONPATH=".", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            with (args.root / (stage + ".controller.log")).open("a") as log:
                child = subprocess.Popen([*command, "--output", str(output)], env=env, stdout=log, stderr=subprocess.STDOUT)
                print("started", stage, "pid", child.pid, flush=True)
                while child.poll() is None:
                    path = output / "status.json"
                    try:
                        progress = read(path) if path.exists() else None
                    except json.JSONDecodeError:
                        progress = None
                    save_json(status, dict(state="running", stage=stage, controller_pid=os.getpid(), child_pid=child.pid,
                        output=str(output), progress=progress, goal_complete=False))
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"{stage} failed with exit {child.returncode}; inspect retained log")
                print("completed", stage, flush=True)
                child = None
        save_json(status, dict(state="completed", controller_pid=None, child_pid=None, learning=str(learning), after=str(after), goal_complete=False))
    except BaseException as error:
        live = child.pid if child is not None and child.poll() is None else None
        save_json(status, dict(state="failed", controller_pid=None, child_pid=live, error=repr(error), traceback=traceback.format_exc(), goal_complete=False))
        raise


if __name__ == "__main__":
    main()

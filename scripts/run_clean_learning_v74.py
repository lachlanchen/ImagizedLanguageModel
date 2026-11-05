#!/usr/bin/env python3
"""Run the paired image-language learning control serially on physical GPU 0."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.cache_semantic_images_v72 import ROOT, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=ROOT/"artifacts/semantic_ilm_v72_20261105")
    args = parser.parse_args()
    out = args.run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", PYTHONPATH=str(ROOT))
    completed = []
    # This is a coordinator lock. Each GPU child separately holds the shared
    # project GPU lock, checks host memory, and checks physical GPU occupancy.
    with (out/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(out/"runtime_handoff.json", dict(session="ilm-clean-learning-v74",
            runner_pid=os.getpid(), gpu=0, noVNC_url=None, owned_gui_processes=[],
            status="running", cleanup="each child exits before the next stage"))
        for stage in ("prepare", "train", "audit", "verify"):
            started = time.time()
            env = environment if stage in ("train", "audit") else dict(environment, CUDA_VISIBLE_DEVICES="")
            command = [sys.executable, "scripts/probe_clean_learning_v74.py", "--stage", stage,
                "--out", str(out), "--source-run", str(args.source_run.resolve())]
            with (out/(stage+".log")).open("a") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(out/"pipeline_status.json", dict(stage=stage, runner_pid=os.getpid(),
                    child_pid=child.pid, started=started, completed=completed))
                try:
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if code:
                atomic_json(out/"pipeline_status.json", dict(stage="failed", failed_phase=stage,
                    exit_code=code, runner_pid=None, child_pid=None, completed=completed, finished=time.time()))
                atomic_json(out/"runtime_handoff.json", dict(session="ilm-clean-learning-v74",
                    runner_pid=None, status="failed", failed_phase=stage, gpu=0,
                    noVNC_url=None, owned_gui_processes=[], owned_model_processes=[]))
                raise SystemExit(code)
            completed.append(dict(stage=stage, wall_seconds=time.time()-started))
        verification = json.loads((out/"verification.json").read_text())
        atomic_json(out/"pipeline_status.json", dict(stage="completed_verified",
            paired_learning_control_pass=verification["paired_learning_control_pass"],
            runner_pid=None, child_pid=None, completed=completed, finished=time.time(),
            language_quality_claim=False, general_ilm_goal_complete=False))
        atomic_json(out/"runtime_handoff.json", dict(session="ilm-clean-learning-v74",
            runner_pid=None, status="completed", gpu=0, noVNC_url=None,
            owned_gui_processes=[], owned_model_processes=[]))


if __name__ == "__main__":
    main()

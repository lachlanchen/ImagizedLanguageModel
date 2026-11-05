#!/usr/bin/env python3
"""Serial full-corpus clean training and original autonomous language audit."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json


def complete(out, stage):
    if stage=="preflight" and (out/"preflight/preflight.json").exists():
        report = json.loads((out/"preflight/preflight.json").read_text())
        if not report["passed"]:
            raise ValueError("retained execution preflight failed")
        return True
    if stage=="train" and (out/"training/training_report.json").exists():
        report = json.loads((out/"training/training_report.json").read_text())
        if (report["updates"]!=40000 or report["draws"]!=640000
                or report["identity"]["purpose"]!="full_corpus_clean"
                or sha(out/"training/checkpoint_latest.pt")!=report["checkpoint_sha256"]):
            raise ValueError("completed production state changed")
        return True
    if stage=="audit" and (out/"development/report.json").exists():
        report = json.loads((out/"development/report.json").read_text())
        if (report["generations"]!=384
                or sha(out/"training/training_report.json")!=report["identity"]["training_report_sha256"]):
            raise ValueError("completed audit changed")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((ROOT/"references/evidence/clean_learning_v74_result.json").read_text())
    source_run = ROOT/receipt["source_run"]
    if (not receipt["verification"]["paired_learning_control_pass"]
            or sha(source_run/"verification.json")!=receipt["verification_sha256"]
            or sha(source_run/"audit/report.json")!=receipt["report_sha256"]):
        raise ValueError("verified paired learning prerequisite required")
    source_cache = ROOT/"artifacts/semantic_ilm_v72_20261105/cache"
    if sha(source_cache/"report.json")!="1da155a5476f59f0c6cff1d0e897534f371dba7e8bf7e07188157176204b4aeb":
        raise ValueError("full training cache changed")
    if not (out/"cache").exists():
        (out/"cache").symlink_to(source_cache, target_is_directory=True)
    if not (out/"cache").is_symlink() or (out/"cache").resolve()!=source_cache.resolve():
        raise ValueError("reuse the exact existing cache link")
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", PYTHONPATH=str(ROOT))
    phases = [
        ("preflight", ["scripts/train_clean_corpus_v75.py", "--cache", str(out/"cache"),
            "--output", str(out/"preflight"), "--preflight"]),
        ("train", ["scripts/train_clean_corpus_v75.py", "--cache", str(out/"cache"),
            "--output", str(out/"training"), "--preflight-receipt", str(out/"preflight/preflight.json")]),
        ("audit", ["scripts/audit_clean_corpus_v75.py", "--run", str(out)]),
        ("verify", ["scripts/verify_semantic_ilm_v72.py", "--run", str(out)]),
    ]
    completed = []
    with (out/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(out/"runtime_handoff.json", dict(session="ilm-clean-corpus-v75", runner_pid=os.getpid(),
            status="running", gpu=0, noVNC_url=None, owned_gui_processes=[],
            cleanup="each child exits before the next phase; reuse the source cache"))
        for stage, command in phases:
            if complete(out, stage):
                completed.append(dict(stage=stage, reused_completed_prerequisite=True))
                continue
            started = time.time()
            with (out/(stage+".log")).open("a") as log:
                child = subprocess.Popen([sys.executable, *command], cwd=ROOT,
                    env=env if stage!="verify" else dict(env, CUDA_VISIBLE_DEVICES=""),
                    stdout=log, stderr=subprocess.STDOUT)
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
                atomic_json(out/"runtime_handoff.json", dict(session="ilm-clean-corpus-v75", runner_pid=None,
                    status="failed", failed_phase=stage, gpu=0, noVNC_url=None,
                    owned_gui_processes=[], owned_model_processes=[]))
                raise SystemExit(code)
            completed.append(dict(stage=stage, wall_seconds=time.time()-started))
        verification = json.loads((out/"verification.json").read_text())
        atomic_json(out/"pipeline_status.json", dict(stage="completed_verified", runner_pid=None,
            child_pid=None, completed=completed, finished=time.time(),
            bounded_chinese_development_pass=verification["bounded_chinese_development_pass"],
            general_ilm_goal_complete=False))
        atomic_json(out/"runtime_handoff.json", dict(session="ilm-clean-corpus-v75", runner_pid=None,
            status="completed", gpu=0, noVNC_url=None, owned_gui_processes=[], owned_model_processes=[]))


if __name__=="__main__":
    main()

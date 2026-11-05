#!/usr/bin/env python3
"""Persistent, serial cache -> train -> autonomous-audit pipeline on GPU 0."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic(path,value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def phase_already_complete(out,stage):
    """Never rewrite completed prerequisite reports during pipeline restart."""
    if stage=="cache" and (out/"cache/report.json").exists():
        report = json.loads((out/"cache/report.json").read_text())
        if report["complete"]:
            if report["completed"]!=12976 or len(report["files_sha256"])!=12976:
                raise ValueError("completed cache has the wrong admitted inventory")
            if sha(out/"cache/recipe.json")!=report["recipe_sha256"]:
                raise ValueError("completed cache recipe changed")
            for name,digest in report["files_sha256"].items():
                if sha(out/"cache"/name)!=digest:
                    raise ValueError("completed cache payload changed")
            return True
    if stage=="train" and (out/"training/training_report.json").exists():
        report = json.loads((out/"training/training_report.json").read_text())
        if (report["updates"]!=40000 or report["draws"]!=640000
                or sha(out/"training/checkpoint_latest.pt")!=report["checkpoint_sha256"]
                or sha(out/"cache/report.json")!=report["identity"]["cache_report_sha256"]):
            raise ValueError("completed training identity changed")
        return True
    if stage=="audit" and (out/"development/report.json").exists():
        report = json.loads((out/"development/report.json").read_text())
        if (report["generations"]!=384 or sha(out/"training/training_report.json")!=report["identity"]["training_report_sha256"]):
            raise ValueError("completed audit identity changed")
        # The next mandatory CPU verification authenticates every raw output.
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    out.mkdir(parents=True,exist_ok=True)
    receipt = out/"preflight/preflight.json"
    if not json.loads(receipt.read_text())["passed"]:
        raise ValueError("real-example execution preflight must pass first")
    environment = dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",PYTHONPATH=str(ROOT))
    phases = [
        ("cache",["scripts/cache_semantic_images_v72.py","--output",str(out/"cache")]),
        ("train",["scripts/train_semantic_ilm_v72.py","--cache",str(out/"cache"),"--output",str(out/"training"),
            "--preflight-receipt",str(receipt)]),
        ("audit",["scripts/audit_semantic_ilm_v72.py","--run",str(out)]),
        ("verify",["scripts/verify_semantic_ilm_v72.py","--run",str(out)]),
    ]
    atomic(out/"runtime_handoff.json",dict(session="ilm-semantic-v72",runner_pid=os.getpid(),
        gpu=0,noVNC_url=None,owned_gui_processes=[],cleanup="each model subprocess exits before the next phase",
        status="running",run=str(out)))
    completed = []
    for stage,command in phases:
        begun = time.time()
        if phase_already_complete(out,stage):
            completed.append(dict(stage=stage,reused_verified_prerequisite=True))
            continue
        env = environment if stage!="verify" else dict(environment,CUDA_VISIBLE_DEVICES="")
        with (out/(stage+".log")).open("a") as log:
            child = subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic(out/"pipeline_status.json",dict(stage=stage,runner_pid=os.getpid(),child_pid=child.pid,
                started=begun,completed=completed))
            code = child.wait()
        if code:
            atomic(out/"pipeline_status.json",dict(stage="failed",failed_phase=stage,exit_code=code,
                runner_pid=None,child_pid=None,finished=time.time(),completed=completed))
            atomic(out/"runtime_handoff.json",dict(session="ilm-semantic-v72",runner_pid=None,status="failed",
                failed_phase=stage,gpu=0,noVNC_url=None,owned_gui_processes=[]))
            raise SystemExit(code)
        completed.append(dict(stage=stage,wall_seconds=time.time()-begun))
    atomic(out/"pipeline_status.json",dict(stage="completed_verified",runner_pid=None,child_pid=None,
        completed=completed,finished=time.time()))
    atomic(out/"runtime_handoff.json",dict(session="ilm-semantic-v72",runner_pid=None,status="completed",
        gpu=0,noVNC_url=None,owned_gui_processes=[],owned_model_processes=[]))


if __name__=="__main__":
    main()

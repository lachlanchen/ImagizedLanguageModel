#!/usr/bin/env python3
"""Persistent, serial cache -> train -> autonomous-audit pipeline on GPU 0."""

import argparse
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

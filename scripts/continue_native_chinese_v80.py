#!/usr/bin/env python3
"""Wait for the owned training process, then run the fixed final image audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.cache_semantic_images_v72 import ROOT, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args=parser.parse_args(); run=args.run.resolve()
    while not (run/"training/training_report.json").exists():
        if (run/"failure.json").exists():
            raise RuntimeError("training failed; preserve failure and do not audit an incomplete checkpoint")
        time.sleep(10)
    phases=[("audit",False),("verify",True)]
    for stage,cpu in phases:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="" if cpu else "0")
        command=[sys.executable,"scripts/audit_native_chinese_v80.py","--run",str(run)]
        if cpu: command.append("--verify")
        with (run/(stage+".log")).open("a") as output:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            atomic_json(run/"audit_pipeline.json",dict(stage=stage,watcher_pid=os.getpid(),child_pid=process.pid))
            code=process.wait()
        if code:
            atomic_json(run/"audit_pipeline.json",dict(stage="failed",failed_phase=stage,
                exit_code=code,watcher_pid=None,child_pid=None))
            raise SystemExit(code)
    atomic_json(run/"audit_pipeline.json",dict(stage="completed",watcher_pid=None,child_pid=None))


if __name__=="__main__": main()

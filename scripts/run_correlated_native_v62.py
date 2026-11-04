#!/usr/bin/env python3
"""Persistent sequential V62 training/replay/audit with exact checkpoint resume."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/correlated_native_density_v62_20261105")
    args=parser.parse_args();os.chdir(ROOT)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("physical GPU 0 required")
    if sys.version_info[:2]!=(3,12):raise ValueError("use the verified Python 3.12 runtime")
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    lock=(run/"pipeline.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=json.loads((run/"preflight/report.json").read_text())
    if not all(preflight["checks"].values()):raise ValueError("completed preflight required")
    child=None;stopping=False

    def record(status,stage,**values):
        row=dict(status=status,stage=stage,pid=os.getpid(),physical_gpu=0,
            child_pid=child.pid if child is not None else None,
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        temporary=run/"pipeline_status.json.partial"
        temporary.write_text(json.dumps(row,indent=2)+"\n");temporary.replace(run/"pipeline_status.json")
        print(json.dumps(row),flush=True)

    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)

    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)

    def execute(stage,arguments,logpath):
        nonlocal child
        if stopping:raise InterruptedError("pipeline stopped before next stage")
        with logpath.open("a") as log:
            child=subprocess.Popen([sys.executable,*arguments],stdout=log,stderr=subprocess.STDOUT)
            record("running",stage,arguments=arguments)
            result=child.wait()
        child=None
        if result!=0:
            record("failed",stage,exit_code=result)
            raise RuntimeError(f"{stage} exited {result}; inspect its log before resuming")
        if stopping:
            record("interrupted",stage,exit_code=result)
            raise InterruptedError("child stopped; any saved training state is resumable")

    try:
        for k in (1,8):
            arm=run/f"k{k}";arm.mkdir(exist_ok=True)
            if not (arm/"model_final.pt").exists():
                command=["scripts/train_correlated_native_density_v62.py","--components",str(k),"--out",str(arm)]
                if (arm/"checkpoint_latest.pt").exists():command.extend(["--resume",str(arm/"checkpoint_latest.pt")])
                execute(f"train_k{k}",command,arm/"console.log")
            if not (arm/"training_report.json").exists():
                record("interrupted",f"train_k{k}",reason="no final report despite zero process exit")
                return 2
            report=json.loads((arm/"training_report.json").read_text())
            if report["smoke_only"] or report["update"]!=1000 or report["components"]!=k:
                raise ValueError("incomplete or mismatched production arm")
            if report["provenance"]!=preflight["provenance"]:raise ValueError("trained source identity differs")
        if not all((run/f"standalone/k{k}/verification.json").exists() for k in (1,8)):
            execute("offline_replay",["scripts/verify_correlated_native_export_v62.py","--run",str(run)],run/"offline_replay.log")
        if not (run/"development/report.json").exists():
            execute("locked_audit",["scripts/eval_correlated_native_density_v62.py","--run",str(run)],run/"locked_audit.log")
        report=json.loads((run/"development/report.json").read_text())
        record("audit_complete","finished",decision=report["decision"],free_reading_pass=report["free_reading_pass"],
               overall_ilm_goal_complete=False)
        return 0
    except BaseException as error:
        record("interrupted" if stopping else "failed","pipeline",error=repr(error))
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=="__main__":raise SystemExit(main())

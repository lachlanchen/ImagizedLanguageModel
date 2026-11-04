#!/usr/bin/env python3
"""Persistent sequential V63 training, standalone replay and locked audit."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/question_assignment_v63_20261105")
    args=parser.parse_args();os.chdir(ROOT);run=Path(args.run)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID":
        raise ValueError("explicit physical GPU 0 with PCI ordering required")
    if sys.version_info[:2]!=(3,12):raise ValueError("use the verified Python 3.12 runtime")
    preflight=json.loads((run/"preflight/report.json").read_text())
    if not all(preflight["checks"].values()):raise ValueError("complete real preflight required")
    lock=(run/"pipeline.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;stopping=False
    def record(status,stage,**values):
        row=dict(status=status,stage=stage,pid=os.getpid(),child_pid=child.pid if child else None,
            physical_gpu=0,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        temporary=run/"pipeline_status.json.partial";temporary.write_text(json.dumps(row,indent=2)+"\n");temporary.replace(run/"pipeline_status.json")
        print(json.dumps(row),flush=True)
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def resources(stage):
        while True:
            if stopping:raise InterruptedError("stopped while waiting for resources")
            memory={line.split(":",1)[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[1].isdigit()}
            available=memory["MemAvailable"]/1024**2
            swap=(memory["SwapTotal"]-memory["SwapFree"])/memory["SwapTotal"] if memory["SwapTotal"] else 0.
            resident=int(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
            if available>=24 and swap<=.75 and resident<=256:return
            record("waiting_resources",stage,available_gib=available,swap_used_fraction=swap,gpu0_resident_mib=resident)
            time.sleep(15)
    def execute(stage,arguments,logpath):
        nonlocal child
        resources(stage)
        with logpath.open("a") as log:
            child=subprocess.Popen([sys.executable,*arguments],stdout=log,stderr=subprocess.STDOUT)
            record("running",stage,arguments=arguments);result=child.wait();child=None
        if stopping:raise InterruptedError("active child stopped; inspect its saved state")
        if result:raise RuntimeError(f"{stage} exited {result}; inspect its log before resume")
    try:
        for arm in ("control","aligned"):
            folder=run/arm;folder.mkdir(exist_ok=True)
            if not (folder/"model_final.pt").exists():
                command=["scripts/train_question_assignment_v63.py","--arm",arm,"--out",str(folder)]
                if (folder/"checkpoint_latest.pt").exists():command.extend(["--resume",str(folder/"checkpoint_latest.pt")])
                execute("train_"+arm,command,folder/"console.log")
            if not (folder/"training_report.json").exists():
                record("interrupted","train_"+arm,reason="no final report despite zero process exit");return 2
            report=json.loads((folder/"training_report.json").read_text())
            if report["smoke_only"] or report["update"]!=2000 or report["arm"]!=arm or report["longest_pair"]:
                raise ValueError("incomplete production state")
            if report["provenance"]!=preflight["provenance"] or report["data_sha256"]!=preflight["data_sha256"]:
                raise ValueError("preflight/training source or data changed")
        if not all((run/"standalone"/arm/"verification.json").exists() for arm in ("control","aligned")):
            execute("offline_replay",["scripts/verify_question_assignment_export_v63.py","--run",str(run)],run/"offline_replay.log")
        if not (run/"development/report.json").exists():
            execute("locked_audit",["scripts/eval_question_assignment_v63.py","--run",str(run)],run/"locked_audit.log")
        report=json.loads((run/"development/report.json").read_text())
        record("audit_complete","finished",decision=report["decision"],free_reading_pass=report["free_reading_pass"],overall_ilm_goal_complete=False)
        return 0
    except BaseException as error:
        record("interrupted" if stopping else "failed","pipeline",error=repr(error));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=="__main__":raise SystemExit(main())

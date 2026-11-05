#!/usr/bin/env python3
"""Wait for the exact live full-corpus coordinator, then run the copy diagnostic."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def atomic(path,value):
    temporary=path.with_suffix(".partial")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        folder=Path("/proc")/str(pid)
        argv=folder.joinpath("cmdline").read_bytes().decode().split("\0")
        ticks=folder.joinpath("stat").read_text().rsplit(")",1)[1].split()[19]
        return dict(pid=pid,start_ticks=ticks,argv=argv,cwd=str(folder.joinpath("cwd").resolve()))
    except (FileNotFoundError,ProcessLookupError):
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/clean_corpus_v75_20261106")
    parser.add_argument("--data",type=Path,default=ROOT/"data/general_instructions/copy_transfer_v75_20261106")
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/copy_transfer_v75_20261106")
    args=parser.parse_args()
    args.run,args.data,args.out=(p.resolve() for p in (args.run,args.data,args.out))
    args.out.mkdir(parents=True,exist_ok=True)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    name=str(Path(__file__).resolve().relative_to(ROOT))
    original=subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)
    if original!=Path(__file__).read_bytes():
        raise ValueError("commit the queue implementation before execution")
    with (args.out/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((args.run/"pipeline_status.json").read_text())
        if state["stage"]=="completed_verified":
            watched=None
        else:
            watched=process_identity(state.get("runner_pid"))
            if (watched is None or not any(a.endswith("scripts/run_clean_corpus_v75.py") for a in watched["argv"])
                    or "--run" not in watched["argv"]):
                raise ValueError("the original live coordinator handle is missing")
            value=watched["argv"][watched["argv"].index("--run")+1]
            if (Path(watched["cwd"])/value).resolve()!=args.run:
                raise ValueError("coordinator belongs to another run")
        child=None
        def interrupted(signum,_frame):
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        try:
            while watched is not None and process_identity(watched["pid"])==watched:
                atomic(args.out/"queue_status.json",dict(stage="waiting_for_original",pid=os.getpid(),
                    source_commit=revision,watched=watched,observed_alive_at=time.time(),
                    gpu_job_started=False,noVNC_url=None))
                time.sleep(15)
            state=json.loads((args.run/"pipeline_status.json").read_text())
            if state["stage"]!="completed_verified":
                raise RuntimeError("original coordinator stopped before verified completion")
            verification=json.loads((args.run/"verification.json").read_text())
            digest=hashlib.sha256((args.run/"development/report.json").read_bytes()).hexdigest()
            if not verification["offline_checks_pass"] or verification["report_sha256"]!=digest:
                raise ValueError("original verification is incomplete")
            environment=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",PYTHONPATH=str(ROOT))
            for stage in ("evaluate","verify"):
                command=[sys.executable,"scripts/audit_copy_transfer_v75.py","--stage",stage,
                    "--run",str(args.run),"--data",str(args.data),"--out",str(args.out)]
                with (args.out/(stage+".log")).open("a") as log:
                    child=subprocess.Popen(command,cwd=ROOT,
                        env=environment if stage=="evaluate" else dict(environment,CUDA_VISIBLE_DEVICES=""),
                        stdout=log,stderr=subprocess.STDOUT)
                    atomic(args.out/"queue_status.json",dict(stage=stage,pid=os.getpid(),child_pid=child.pid,
                        source_commit=revision,original_run_verified=True,noVNC_url=None))
                    code=child.wait()
                child=None
                if code:
                    raise RuntimeError(stage+" exited with "+str(code))
            result=json.loads((args.out/"verification.json").read_text())
            atomic(args.out/"queue_status.json",dict(stage="completed_verified",pid=None,child_pid=None,
                summary=result["summary"],descriptive_diagnostic_only=True,language_quality_claim=False))
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()
            atomic(args.out/"queue_status.json",dict(stage="failed",pid=None,child_pid=None,
                error=str(error),source_commit=revision,language_quality_claim=False))
            raise


if __name__=="__main__":
    main()

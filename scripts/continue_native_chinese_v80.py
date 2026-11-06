#!/usr/bin/env python3
"""Wait for the owned training process, then run the fixed final image audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def atomic_json(path,value):
    temporary=path.with_suffix(".partial")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def owned_training_alive(run):
    runtime=run/"runtime_handoff.json"
    if not runtime.exists():
        return False,None
    pid=json.loads(runtime.read_text()).get("model_pid")
    if not isinstance(pid,int) or pid<=1:
        return False,pid
    try:
        arguments=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        state=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0]
    except FileNotFoundError:
        return False,pid
    expected=any(arg.endswith(b"scripts/train_native_chinese_v80.py") for arg in arguments)
    # Resolve the --run argument against this known project's working directory.
    try:
        position=arguments.index(b"--run")
        target=Path(os.fsdecode(arguments[position+1]))
        expected=expected and (ROOT/target).resolve()==run
    except (ValueError,IndexError):
        expected=False
    return expected and state!="Z",pid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args=parser.parse_args(); run=args.run.resolve()
    with (run/"audit_coordinator.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        coordinate(run)


def coordinate(run):
    missing=0
    while not (run/"training/training_report.json").exists():
        alive,pid=owned_training_alive(run)
        missing=0 if alive else missing+1
        atomic_json(run/"audit_pipeline.json",dict(stage="waiting_for_training",watcher_pid=os.getpid(),
            confirmed_live_training_pid=pid if alive else None,consecutive_missing_process_checks=missing))
        if missing>=2:
            raise RuntimeError("owned training process is absent; preserve its state and do not audit an incomplete checkpoint")
        pending=[folder for folder in (run/"diagnostics").glob("update_*")
            if not (folder/"pixel_scores.json").exists()
            and len(list(folder.glob("[0-9][0-9]/source.json")))==4]
        if pending:
            child(run,"training_diagnostics",True,"scripts/score_native_training_samples_v80.py")
        time.sleep(10)
    phases=[("audit",False,"scripts/audit_native_chinese_v80.py"),
        ("verify",True,"scripts/audit_native_chinese_v80.py"),
        ("standalone",False,"scripts/check_completed_native_prompt_routes_v80.py")]
    for stage,cpu,script in phases:
        child(run,stage,cpu,script,verify=stage=="verify")
    atomic_json(run/"audit_pipeline.json",dict(stage="completed",watcher_pid=None,child_pid=None))


def child(run,stage,cpu,script,*,verify=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="" if cpu else "0")
    command=[sys.executable,script,"--run",str(run)]
    if verify: command.append("--verify")
    with (run/(stage+".log")).open("a") as output:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True)
        atomic_json(run/"audit_pipeline.json",dict(stage=stage,watcher_pid=os.getpid(),child_pid=process.pid))
        try:
            code=process.wait()
        finally:
            # Standalone inference owns CLI grandchildren in this process
            # group. Clean up this exact group, never another project's job.
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            if process.poll() is None:
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
    if code:
        atomic_json(run/"audit_pipeline.json",dict(stage="failed",failed_phase=stage,
            exit_code=code,watcher_pid=None,child_pid=None))
        raise SystemExit(code)


if __name__=="__main__":
    def stop(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    try:
        main()
    except BaseException as error:
        if "--run" in sys.argv:
            path=Path(sys.argv[sys.argv.index("--run")+1]).resolve()
            if path.exists():
                atomic_json(path/"audit_pipeline.json",dict(stage="watcher_stopped",watcher_pid=None,
                    child_pid=None,error_type=type(error).__name__,message=str(error)))
        raise

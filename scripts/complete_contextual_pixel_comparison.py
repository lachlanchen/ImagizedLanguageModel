#!/usr/bin/env python3
"""Wait for the registered controller, then run standalone checks and export."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT=Path(__file__).absolute().parents[1]
EXTRA_SOURCES=("scripts/complete_contextual_pixel_comparison.py",
    "scripts/infer_adapted_pixel_spans.py","ilm/visual_lm/adapted_pixel_runtime.py",
    "scripts/export_contextual_pixel_comparison.py","scripts/audit_contextual_pixel_data.py",
    "ilm/visual_lm/clean_image_actor.py")


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path):return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write(path,value):
    temporary=path.with_suffix(".partial")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def process(pid):
    if pid is None:return None
    path=Path("/proc")/str(pid)
    try:
        raw=(path/"stat").read_text();fields=raw[raw.rfind(")")+2:].split()
        return dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],
            argv=(path/"cmdline").read_bytes().decode().strip("\0").split("\0"),
            cwd=str((path/"cwd").resolve()))
    except (FileNotFoundError,ProcessLookupError):return None


def controller_identity(pid,run):
    value=process(pid)
    if value is None or value["state"] in ("Z","X"):return None
    arguments=value["argv"]
    if "--run" not in arguments:raise ValueError("selected PID is not the expected controller")
    path=Path(arguments[arguments.index("--run")+1])
    if not path.is_absolute():path=Path(value["cwd"])/path
    scripts=[Path(s).name for s in arguments]
    if path.resolve()!=run.resolve() or "compare_contextual_pixel_learning.py" not in scripts:
        raise ValueError("selected live PID belongs to another runtime")
    return value


def source_identity(run):
    value=read(run/"identity.json");sources=dict(value["sources"])
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for name in EXTRA_SOURCES:
        raw=subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)
        digest=hashlib.sha256(raw).hexdigest()
        if sha(ROOT/name)!=digest:raise ValueError("commit completion implementation before launching")
        sources[name]=digest
    verify_sources(sources)
    return dict(completion_source_commit=revision,sources=sources,training_identity_sha256=sha(run/"identity.json"))


def verify_sources(sources):
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise ValueError("bound training or completion implementation changed: "+name)


def resource_snapshot():
    values={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,rest=line.split(":",1);values[key]=int(rest.split()[0])
    gpu=int(subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=memory.used",
        "--format=csv,noheader,nounits"],text=True,timeout=10).strip())
    tmux=subprocess.run(["tmux","list-sessions","-F","#{session_name}"],capture_output=True,text=True,timeout=5)
    sockets=subprocess.check_output(["ss","-ltnH"],text=True,timeout=5)
    ports=[]
    for line in sockets.splitlines():
        address=line.split()[3]
        if address.startswith("127.0.0.1:") and 5900<=int(address.rsplit(":",1)[1])<=6199:
            ports.append(int(address.rsplit(":",1)[1]))
    pids=[]
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:info=process(int(path.name))
        except (PermissionError,OSError):continue
        if info and info["cwd"]==str(ROOT) and info["state"] not in ("Z","X"):
            names=[Path(a).name for a in info["argv"]]
            if any(n.startswith(("train_","infer_","compare_")) and n.endswith(".py") for n in names):
                pids.append(info["pid"])
    swap=1-values["SwapFree"]/values["SwapTotal"] if values["SwapTotal"] else 0.
    return dict(available_ram_gib=values["MemAvailable"]/1024**2,swap_used_fraction=swap,
        gpu0_used_mib=gpu,project_model_pids=pids,tmux_session_count=len(tmux.stdout.splitlines()),
        loopback_novnc_ports=sorted(set(ports)),owned_gui_sessions=[],novnc_url=None)


class Completion:
    def __init__(self,args):
        self.args=args;self.child=None;self.last_stage=None;self.last_print=0.;self.owns_workdir=False

    def status(self,stage,**fields):
        record=dict(stage=stage,supervisor_pid=os.getpid(),child_pid=self.child.pid if self.child and self.child.poll() is None else None,
            controller_pid=self.args.controller_pid,**fields)
        write(self.args.work_dir/"status.json",record)
        if stage!=self.last_stage or time.monotonic()-self.last_print>=60:
            print(json.dumps(record),flush=True);self.last_stage=stage;self.last_print=time.monotonic()

    def await_resources(self,gpu):
        while True:
            snapshot=resource_snapshot();write(self.args.work_dir/"resources.json",snapshot)
            host_ok=snapshot["available_ram_gib"]>=24 and snapshot["swap_used_fraction"]<=.75
            card_ok=snapshot["gpu0_used_mib"]<=256 and not snapshot["project_model_pids"]
            if host_ok and (not gpu or card_ok):
                return
            self.status("waiting_for_resources",resources=snapshot);time.sleep(10)

    def invoke(self,name,arguments,gpu):
        verify_sources(self.recipe["sources"])
        self.await_resources(gpu)
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0" if gpu else "",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONPATH=str(ROOT))
        executable=ROOT/"artifacts/external/mixar_env_py312/bin/python"
        # Preserve this executable path: resolving its symlink changes the environment.
        command=[str(executable),*arguments]
        with (self.args.work_dir/(name+".log")).open("w") as log:
            self.child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            began=time.monotonic()
            self.status(name,command=command)
            while self.child.poll() is None:
                self.status(name,runtime_seconds=time.monotonic()-began);time.sleep(5)
            code=self.child.returncode;self.child=None
        if code:raise RuntimeError(name+" failed; inspect its preserved log")

    def run(self):
        run=self.args.run;self.args.work_dir.mkdir(parents=True,exist_ok=False)
        self.owns_workdir=True
        self.recipe=source_identity(run)
        original=controller_identity(self.args.controller_pid,run)
        if original is None and read(run/"status.json")["stage"]!="completed":
            raise ValueError("controller is missing without a completed experiment")
        self.recipe.update(controller_identity=original,run=local(run))
        write(self.args.work_dir/"recipe.json",self.recipe)
        while original is not None:
            current=process(original["pid"])
            if current is None or current["state"] in ("Z","X") or current["start_ticks"]!=original["start_ticks"]:break
            self.status("waiting_for_controller",controller_status=read(run/"status.json"));time.sleep(10)
        if read(run/"status.json")["stage"]!="completed":
            raise ValueError("controller exited without completing the registered comparison")
        verify_sources(self.recipe["sources"])
        result=read(run/"result.json")
        if result["full_native_assessment_calls"]!=512:raise ValueError("all registered native assessments required")
        previous=ROOT/"artifacts/contextual_pixel_spans_v1/prepared"
        if sha(previous/"inputs.json")!=read(run/"identity.json")["development_inputs_sha256"]:
            raise ValueError("development input index changed")
        rows=read(previous/"inputs.json")
        chosen=[r for r in rows if r["split"]=="development" and r["task"]=="qa" and r["condition"]=="normal"][:2]
        chosen += [r for r in rows if r["split"]=="development" and r["task"]=="copy"][:2]
        if len(chosen)!=4 or [r["language"] for r in chosen[2:]]!=["zh","en"]:
            raise ValueError("fixed standalone input cohort changed")
        images=[]
        for row in chosen:
            path=previous/row["input"]
            if sha(path)!=row["input_sha256"]:raise ValueError("standalone input changed")
            images.append(local(path))
        for arm in ("frozen","adapted"):
            report=read(run/arm/"training_report.json");checkpoint=run/arm/"checkpoint.pt"
            if sha(checkpoint)!=report["checkpoint_sha256"] or report["updates"]!=3257:
                raise ValueError("complete final checkpoint required")
            out=run/arm/"standalone_replay"
            if out.exists():
                if not (out/"report.json").exists():raise ValueError("partial standalone output requires review; preserve it")
                saved=read(out/"report.json")
                if saved["checkpoint_sha256"]!=report["checkpoint_sha256"] or len(saved["outputs"])!=4:
                    raise ValueError("existing standalone output identity differs")
                for observed,wanted in zip(saved["outputs"],chosen):
                    if observed["input_sha256"]!=wanted["input_sha256"] or sha(out/observed["output"])!=observed["output_sha256"]:
                        raise ValueError("existing standalone output changed")
            else:
                arguments=["scripts/infer_adapted_pixel_spans.py","--checkpoint",local(checkpoint),
                    "--checkpoint-sha256",report["checkpoint_sha256"],"--images",*images,
                    "--output",local(out)]
                self.invoke(arm+"_standalone",arguments,gpu=True)
        output=self.args.work_dir/"verified_export"
        self.invoke("independent_export",["scripts/export_contextual_pixel_comparison.py","--run",local(run),
            "--output",local(output)],gpu=False)
        exported=read(output/"result.json")
        if not exported["independent_png_scores_exact"] or not exported["all512_raw_answers_retained"]:
            raise ValueError("independent export is incomplete")
        self.status("completed",verified_export=local(output),
            standalone_calls=len(exported["standalone_native_replays"]),general_ilm_goal_complete=False,
            publication_review_pending=True)

    def stop_child(self):
        if self.child and self.child.poll() is None:
            self.child.terminate()
            try:self.child.wait(timeout=15)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait(timeout=5)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--controller-pid",type=int,required=True)
    p.add_argument("--work-dir",type=Path,required=True)
    args=p.parse_args();args.run=args.run.absolute();args.work_dir=args.work_dir.absolute()
    supervisor=Completion(args)
    def interrupted(signum,_frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (ROOT/"artifacts/contextual_pixel_completion.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:supervisor.run()
        except BaseException as error:
            supervisor.stop_child()
            if supervisor.owns_workdir:
                write(args.work_dir/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),general_ilm_goal_complete=False))
                supervisor.status("failed",error=repr(error))
            raise

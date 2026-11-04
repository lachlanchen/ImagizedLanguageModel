"""Answer-image recognition outside the ILM, using one owned local evaluator."""

from __future__ import annotations

import base64
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import requests

MODEL="qwen3-vl:8b-instruct-q4_K_M"
MANIFEST_SHA="0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
MODEL_SHA="1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466"
PROMPT=('Transcribe only the writing visible in the image. Copy the characters and '
    'punctuation exactly. Do not answer a question, complete missing writing, or '
    'guess damaged characters. Return only a JSON object with keys "text" and '
    '"readable". If there is no legible writing, return {"text":"","readable":false}.')
OPTIONS=dict(temperature=0,top_p=1,seed=20266371,num_ctx=2048,num_predict=256)
SCHEMA=dict(type="object",properties=dict(text=dict(type="string"),readable=dict(type="boolean")),
            required=["text","readable"],additionalProperties=False)


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(16*1024*1024),b""):value.update(block)
    return value.hexdigest()


def atomic_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");temporary.replace(path)


def request_body(png_bytes):
    # The function accepts image bytes alone. No record, gold text, question,
    # filename, actor name or condition can be added to the recognition prompt.
    return dict(model=MODEL,messages=[dict(role="user",content=PROMPT,
        images=[base64.b64encode(png_bytes).decode("ascii")])],
        stream=False,think=False,format=copy.deepcopy(SCHEMA),options=dict(OPTIONS),keep_alive="10m")


def parse_response(response):
    try:
        if not isinstance(response,dict) or not isinstance(response.get("message"),dict):
            raise ValueError("missing response message")
        content=response["message"].get("content","")
        value=json.loads(content)
        if not isinstance(value,dict) or set(value)!={"text","readable"}:
            raise ValueError("response must contain exactly text and readable")
        if not isinstance(value["text"],str) or type(value["readable"]) is not bool:
            raise ValueError("response field types differ from the schema")
        return dict(valid_json=True,**value)
    except (ValueError,TypeError) as error:
        return dict(valid_json=False,text=None,readable=None,error=str(error))


def gpu_memory():
    return int(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used",
        "--format=csv,noheader,nounits"],text=True).strip())


def check_host_memory():
    values={line.split(":",1)[0]:int(line.split()[1]) for line in
        Path("/proc/meminfo").read_text().splitlines() if line.split()[1].isdigit()}
    available=values["MemAvailable"]/1024**2
    swap_total=values["SwapTotal"]
    swap_used=(swap_total-values["SwapFree"])/swap_total if swap_total else 0.
    if available<24 or swap_used>.75:
        raise RuntimeError("shared host memory limit: wait or clean up only obsolete owned work")
    return dict(available_gib=available,swap_used_fraction=swap_used)


def _children(pid):
    pairs=[]
    for row in subprocess.check_output(["ps","-e","-o","pid=,ppid="],text=True).splitlines():
        a,b=map(int,row.split());pairs.append((a,b))
    result={pid};changed=True
    while changed:
        prior=len(result);result.update(a for a,b in pairs if b in result);changed=len(result)!=prior
    return result


def _starttime(pid):
    try:
        fields=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return None if fields[0]=="Z" else fields[19]
    except FileNotFoundError:return None


def _process_group(pid):
    rows=subprocess.check_output(["ps","-e","-o","pid=,pgid=,sid="],text=True).splitlines()
    return {int(a) for a,b,c in (row.split() for row in rows) if int(b)==pid and int(c)==pid}


class LocalImageReader:
    def __init__(self,server,base_url,cache):
        self.server,self.base_url,self.cache=server,base_url,Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session();self.session.trust_env=False
        self.sampled_gpu0_memory_mib_max=0;self.actual_calls=0
        self.request_sha256=hashlib.sha256(json.dumps(dict(model=MODEL,manifest=MANIFEST_SHA,
            prompt=PROMPT,options=OPTIONS,schema=SCHEMA),sort_keys=True).encode()).hexdigest()

    def recognize(self,path):
        path=Path(path);digest=sha256(path);result=self.cache/(digest+".json")
        if result.exists():
            saved=json.loads(result.read_text())
            if saved["image_sha256"]!=digest or saved["request_sha256"]!=self.request_sha256:
                raise ValueError("recognition cache image/request identity differs")
            return saved
        if self.server.poll() is not None:raise RuntimeError("owned evaluator is no longer running")
        begun=time.perf_counter()
        try:
            response=self.session.post(self.base_url+"/api/chat",json=request_body(path.read_bytes()),timeout=(5,180))
            response.raise_for_status();raw=response.json()
        except Exception as error:
            # Preserve the failed attempt, without substituting an empty answer.
            atomic_json(dict(image_sha256=digest,error=repr(error)),self.cache/(digest+"_transport_error.json"))
            raise
        parsed=parse_response(raw);self.actual_calls+=1
        resident=gpu_memory();self.sampled_gpu0_memory_mib_max=max(self.sampled_gpu0_memory_mib_max,resident)
        saved=dict(image_sha256=digest,request_sha256=self.request_sha256,**parsed,raw_response=raw,elapsed_seconds=time.perf_counter()-begun,
                   gpu0_resident_memory_mib=resident)
        atomic_json(saved,result)
        if resident>=20*1024:raise RuntimeError("owned evaluator exceeded the GPU 0 memory ceiling")
        return saved


@contextmanager
def _start_owned_reader(run,out,local_root,port=11437):
    run,out,local_root=Path(run),Path(out),Path(local_root).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("explicit physical GPU 0 required")
    status=json.loads((run/"pipeline_status.json").read_text())
    if status["status"]!="audit_complete" or not (run/"development/report.json").exists():
        raise RuntimeError("finish both actors, replay and primary audit before evaluator startup")
    host_memory=check_host_memory()
    if gpu_memory()>256:raise RuntimeError("GPU 0 is occupied; wait without stopping other work")
    # CUDA's PCI ordering must map ordinal zero to nvidia-smi physical GPU 0.
    buses=subprocess.check_output(["nvidia-smi","--query-gpu=index,pci.bus_id","--format=csv,noheader"],text=True).splitlines()
    if sorted((row.split(",")[1].strip(),row.split(",")[0].strip()) for row in buses)[0][1]!="0":
        raise RuntimeError("PCI order does not map CUDA ordinal zero to physical GPU 0")
    with socket.socket() as check:check.bind(("127.0.0.1",port))
    binary=local_root/".local/ollama/bin/ollama";models=local_root/".local/models/ollama"
    manifest=models/"manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M"
    blob=models/("blobs/sha256-"+MODEL_SHA)
    if sha256(manifest)!=MANIFEST_SHA or sha256(blob)!=MODEL_SHA:
        raise ValueError("pinned evaluator weights differ; never pull a substitute")
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",CUDA_DEVICE_ORDER="PCI_BUS_ID",OLLAMA_VULKAN="0",
        OLLAMA_HOST=f"127.0.0.1:{port}",OLLAMA_MODELS=str(models),OLLAMA_CONTEXT_LENGTH="2048",
        OLLAMA_NUM_PARALLEL="1",OLLAMA_MAX_LOADED_MODELS="1",OLLAMA_KEEP_ALIVE="10m",
        OLLAMA_NO_CLOUD="1",OLLAMA_NOPRUNE="1",OLLAMA_SCHED_SPREAD="0")
    log=(out/"owned_server.log").open("a")
    server=subprocess.Popen([str(binary),"serve"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    base_url=f"http://127.0.0.1:{port}";reader=None
    started=time.perf_counter()
    try:
        reader=LocalImageReader(server,base_url,out/"recognizer_outputs")
        atomic_json(dict(server_pid=server.pid,physical_gpu=0,base_url=base_url,
            host_memory_before=host_memory,binary_sha256=sha256(binary),
            model_manifest_sha256=MANIFEST_SHA,model_weight_sha256=MODEL_SHA,
            options=OPTIONS,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest()),out/"owned_runtime.json")
        while True:
            if server.poll() is not None:raise RuntimeError("owned evaluator server failed to start")
            try:
                response=reader.session.get(base_url+"/api/version",timeout=1);response.raise_for_status()
                atomic_json(response.json(),out/"server_version.json");break
            except requests.RequestException:
                if time.perf_counter()-started>60:raise RuntimeError("owned evaluator startup timeout")
                time.sleep(.25)
        yield reader
    finally:
        pids=_process_group(server.pid)
        if server.poll() is None:pids.update(_children(server.pid))
        owned={pid:_starttime(pid) for pid in pids}
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
        # Terminate a surviving descendant only if its process identity still
        # matches this owned server tree. Never target the shared Ollama service.
        for pid,birth in owned.items():
            if pid!=server.pid and birth is not None and _starttime(pid)==birth:
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.perf_counter()+10
        while any(_starttime(pid)==birth for pid,birth in owned.items() if birth is not None) and time.perf_counter()<deadline:
            time.sleep(.2)
        remaining=[pid for pid,birth in owned.items() if birth is not None and _starttime(pid)==birth]
        for pid in remaining:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if remaining:time.sleep(.5)
        still_live=[pid for pid,birth in owned.items() if birth is not None and _starttime(pid)==birth]
        if reader is not None:reader.session.close()
        log.close()
        with socket.socket() as probe:port_open=probe.connect_ex(("127.0.0.1",port))==0
        atomic_json(dict(server_pid=server.pid,owned_pids=sorted(owned),remaining_owned_live_pids=still_live,
            port_open_after_cleanup=port_open,
            gpu0_memory_mib_after=gpu_memory(),actual_recognition_calls=reader.actual_calls if reader else 0,
            sampled_gpu0_memory_mib_max=reader.sampled_gpu0_memory_mib_max if reader else 0,
            elapsed_seconds=time.perf_counter()-started),out/"cleanup.json")
        if still_live:raise RuntimeError("an owned evaluator process survived cleanup")


@contextmanager
def owned_reader(run,out,local_root,port=11437):
    with (Path(run)/"vision_evaluator.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with _start_owned_reader(run,out,local_root,port) as reader:yield reader
        finally:fcntl.flock(lock,fcntl.LOCK_UN)

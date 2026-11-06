#!/usr/bin/env python3
"""Prepare and run calibrated, image-only reading after the original V80 pipeline."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps
import requests

from ilm.visual_lm.blind_vision_transcription import (
    MODEL, MODEL_SHA, MANIFEST_SHA, PROMPT, SCHEMA, check_host_memory,
    gpu_memory, _children, _process_group, _starttime,
)
from scripts.audit_correlated_native_transcripts_v62 import bootstrap
from scripts.audit_native_pixel_transcripts_v61 import normalize, scores
from scripts.cache_semantic_images_v72 import ROOT, committed_identity, sha

PLAN = "references/native_neural_transcription_v80_protocol.md"
CAL_SHA = "5fed1616f03b9ca4fd6791dda00c5bfb08b8ed0a93c944d2c45a574bc996adf7"
OPTIONS = dict(temperature=0,top_p=1,seed=20268040,num_ctx=8192,num_predict=1024)
SOURCES = ("scripts/audit_native_neural_transcripts_v80.py", PLAN,
    "ilm/visual_lm/blind_vision_transcription.py", "ilm/visual_lm/codec_language_audit.py",
    "scripts/audit_correlated_native_transcripts_v62.py", "scripts/audit_native_pixel_transcripts_v61.py",
    "scripts/cache_semantic_images_v72.py", "scripts/audit_native_chinese_v80.py",
    "scripts/score_native_training_samples_v80.py")


class PipelinePending(RuntimeError):
    """The existing pipeline still owns the GPU; no diagnostic was attempted."""


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");temporary.replace(path)


def view(pixels,folder):
    if pixels.dtype!=np.uint8 or pixels.ndim!=2 or pixels.shape[0]!=32 or not pixels.shape[1] or not np.isin(pixels,(0,255)).all():
        raise ValueError("complete nonempty binary writing strip required")
    spans=[];start=0;width=min(1024,pixels.shape[1])
    while start<pixels.shape[1]:
        end=min(start+1024,pixels.shape[1])
        if end<pixels.shape[1]:
            lower=start+512;white=np.flatnonzero(np.all(pixels[:,lower:end]==255,axis=0))
            if len(white):end=lower+int(white[-1])+1
        spans.append((start,end));start=end
    page=np.full((len(spans)*36-4,width),255,np.uint8)
    for i,(a,b) in enumerate(spans):page[i*36:i*36+32,:b-a]=pixels[:,a:b]
    assert np.array_equal(np.concatenate([page[i*36:i*36+32,:b-a] for i,(a,b) in enumerate(spans)],axis=1),pixels)
    image=ImageOps.expand(Image.fromarray(page).resize((width*4,len(page)*4),Image.Resampling.NEAREST),border=16,fill=255)
    assert np.array_equal(np.asarray(image)[16:-16:4,16:-16:4],page)
    buffer=io.BytesIO();image.save(buffer,format="PNG");raw=buffer.getvalue()
    key=hashlib.sha256(raw).hexdigest();folder.mkdir(parents=True,exist_ok=True);path=folder/(key+".png")
    if path.exists() and path.read_bytes()!=raw:raise ValueError("view hash collision")
    path.write_bytes(raw)
    return key,dict(source_shape=list(pixels.shape),page_shape=list(page.shape),spans=spans,source_pixels_preserved=True)


def prepare(run,out):
    cal=run/"cpu_transcription_v80r2";manifest=cal/"private_calibration.json"
    if sha(manifest)!=CAL_SHA:raise ValueError("the fixed 288-case calibration changed")
    identity=dict(**committed_identity(SOURCES),training_identity_sha256=sha(run/"identity.json"),
        calibration_manifest_sha256=CAL_SHA,model=MODEL,manifest_sha256=MANIFEST_SHA,body_sha256=MODEL_SHA,
        prompt=PROMPT,schema=SCHEMA,options=OPTIONS,thinking=False,
        questions_visible=False,references_visible=False,shared_foundation_with_actor=True)
    if (out/"identity.json").exists():
        previous=json.loads((out/"identity.json").read_text())
        if {k:v for k,v in previous.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("earlier neural diagnostic identity differs")
        identity=previous
    elif out.exists() and any(out.iterdir()):raise ValueError("preserve the nonempty output directory")
    write(out/"identity.json",identity);rows=[]
    for row in json.loads(manifest.read_text()):
        path=cal/"anonymous_calibration"/(row["image_sha256"]+".png")
        if sha(path)!=row["image_sha256"]:raise ValueError("original calibration image changed")
        enlarged=np.asarray(Image.open(path));pixels=enlarged[16:-16:4,16:-16:4].copy()
        assert np.array_equal(np.repeat(np.repeat(pixels,4,axis=0),4,axis=1),enlarged[16:-16,16:-16])
        key,layout=view(pixels,out/"calibration_images")
        rows.append(dict(kind=row["kind"],expected_text=row["expected_text"],original_image_sha256=row["image_sha256"],image_sha256=key,layout=layout))
    if len(rows)!=288:raise ValueError("all calibration cases required")
    destination=out/"private_calibration.json"
    if destination.exists() and json.loads(destination.read_text())!=rows:raise ValueError("prepared calibration changed")
    write(destination,rows)
    write(out/"preparation.json",dict(identity_sha256=sha(out/"identity.json"),manifest_sha256=sha(destination),
        cases=288,unique_images=len({r["image_sha256"] for r in rows}),source_pixels_preserved=True,
        max_png_width=max(Image.open(out/"calibration_images"/(r["image_sha256"]+".png")).width for r in rows),
        gpu_calls=0,actor_calls=0,neural_recognition_performed=False))
    return identity,rows


def ready(run):
    pipeline=run/"audit_pipeline.json"
    if not pipeline.exists() or json.loads(pipeline.read_text()).get("stage")!="completed":
        raise PipelinePending("original training, fixed audit, CPU verification and standalone CLI pipeline must finish first")
    from scripts.audit_native_chinese_v80 import training_state
    training,checkpoint=training_state(run)
    verification=json.loads((run/"verification.json").read_text());report=json.loads((run/"development/report.json").read_text())
    cli=json.loads((run/"standalone_inference/verification.json").read_text())
    if (not verification["offline_checks_pass"] or verification["report_sha256"]!=sha(run/"development/report.json")
            or report["generations"]!=384 or report["identity"]["checkpoint_sha256"]!=sha(checkpoint)
            or cli["actual_completed_checkpoint_calls"]!=3 or not cli["typed_image_input_and_generation_bit_exact"]
            or cli["identity"]["checkpoint_sha256"]!=sha(checkpoint)
            or cli["identity"]["original_verification_sha256"]!=sha(run/"verification.json")):
        raise ValueError("completed original artifacts differ")
    for name,digest in report["files_sha256"].items():
        if sha(run/"development"/name)!=digest:raise ValueError("original autonomous output changed")
    return dict(training_sha256=sha(run/"training/training_report.json"),checkpoint_sha256=sha(checkpoint),
        original_audit_sha256=sha(run/"development/report.json"),verification_sha256=sha(run/"verification.json"),
        standalone_verification_sha256=sha(run/"standalone_inference/verification.json"))


class Reader:
    def __init__(self,server,url,out,identity):
        self.server,self.url,self.out=server,url,out;self.session=requests.Session();self.session.trust_env=False
        self.request_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.calls=0;self.valid_responses=0;self.max_memory=0;self.started=time.perf_counter()

    def recognize(self,path):
        key=sha(path);destination=self.out/"responses"/(key+".json")
        if destination.exists():
            value=json.loads(destination.read_text())
            if value["image_sha256"]!=key or value["request_sha256"]!=self.request_sha:raise ValueError("cached response differs")
            return value
        if self.server.poll() is not None:raise RuntimeError("owned neural reader exited")
        if time.perf_counter()-self.started>4*3600:raise RuntimeError("neural diagnostic time ceiling")
        body=dict(model=MODEL,messages=[dict(role="user",content=PROMPT,images=[base64.b64encode(path.read_bytes()).decode()])],
            stream=False,think=False,format=SCHEMA,options=OPTIONS,keep_alive="10m")
        raw=None;error=None;begun=time.perf_counter()
        try:
            response=self.session.post(self.url+"/api/chat",json=body,timeout=(5,180))
            response.raise_for_status();raw=response.json()
            value=json.loads(raw["message"]["content"])
            if (not isinstance(value,dict) or set(value)!={"text","readable"} or not isinstance(value["text"],str)
                    or type(value["readable"]) is not bool or (not value["readable"] and value["text"])):
                raise ValueError("invalid or inconsistent transcription JSON")
            valid=True
        except (requests.RequestException,ValueError,KeyError,TypeError) as exception:
            value=dict(text=None,readable=None);valid=False;error=repr(exception)
        self.calls+=1;self.valid_responses+=int(valid)
        resident=gpu_memory();self.max_memory=max(self.max_memory,resident)
        result=dict(image_sha256=key,request_sha256=self.request_sha,valid_json=valid,**value,
            raw_response=raw,error=error,elapsed_seconds=time.perf_counter()-begun,gpu0_memory_mib=resident)
        write(destination,result)
        if resident>=20*1024:raise RuntimeError("owned neural reader exceeded its memory ceiling")
        return result


@contextmanager
def owned_reader(run,out,local,port,identity):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("explicit physical GPU0 required")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);completed=ready(run);host=check_host_memory()
        if gpu_memory()>256:raise RuntimeError("GPU0 occupied; do not stop another project")
        buses=subprocess.check_output(["nvidia-smi","--query-gpu=index,pci.bus_id","--format=csv,noheader"],text=True).splitlines()
        if sorted((r.split(",")[1].strip(),r.split(",")[0].strip()) for r in buses)[0][1]!="0":raise RuntimeError("physical GPU ordering differs")
        with socket.socket() as probe:probe.bind(("127.0.0.1",port))
        models=local/".local/models/ollama";binary=local/".local/ollama/bin/ollama"
        manifest=models/"manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M"
        if sha(manifest)!=MANIFEST_SHA:raise ValueError("pinned model manifest differs")
        spec=json.loads(manifest.read_text());assets={}
        for descriptor in [spec["config"],*spec["layers"]]:
            algorithm,digest=descriptor["digest"].split(":",1);path=models/"blobs"/(algorithm+"-"+digest)
            if algorithm!="sha256" or sha(path)!=digest or path.stat().st_size!=descriptor["size"]:raise ValueError("pinned evaluator asset differs")
            assets[descriptor["digest"]]=descriptor["size"]
        if "sha256:"+MODEL_SHA not in assets:raise ValueError("pinned evaluator body absent")
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",CUDA_DEVICE_ORDER="PCI_BUS_ID",OLLAMA_VULKAN="0",
            OLLAMA_HOST=f"127.0.0.1:{port}",OLLAMA_MODELS=str(models),OLLAMA_CONTEXT_LENGTH="8192",
            OLLAMA_NUM_PARALLEL="1",OLLAMA_MAX_LOADED_MODELS="1",OLLAMA_KEEP_ALIVE="10m",
            OLLAMA_NO_CLOUD="1",OLLAMA_NOPRUNE="1",OLLAMA_SCHED_SPREAD="0")
        log=(out/"owned_server.log").open("a")
        try:
            server=subprocess.Popen([str(binary),"serve"],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            log.close();raise
        reader=None;started=time.perf_counter()
        try:
            reader=Reader(server,f"http://127.0.0.1:{port}",out,identity)
            write(out/"owned_runtime.json",dict(server_pid=server.pid,physical_gpu=0,port=port,host=host,
                binary_sha256=sha(binary),assets=assets,completed_original_pipeline=completed,options=OPTIONS))
            while True:
                if server.poll() is not None:raise RuntimeError("owned evaluator failed to start")
                try:
                    response=reader.session.get(reader.url+"/api/version",timeout=1);response.raise_for_status()
                    write(out/"server_version.json",response.json());break
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
            for pid,birth in owned.items():
                if pid!=server.pid and birth is not None and _starttime(pid)==birth:
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            deadline=time.perf_counter()+10
            while any(_starttime(pid)==birth for pid,birth in owned.items() if birth is not None) and time.perf_counter()<deadline:time.sleep(.2)
            for pid,birth in owned.items():
                if birth is not None and _starttime(pid)==birth:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            time.sleep(.5);remaining=[pid for pid,birth in owned.items() if birth is not None and _starttime(pid)==birth]
            if reader is not None:reader.session.close()
            log.close()
            with socket.socket() as probe:opened=probe.connect_ex(("127.0.0.1",port))==0
            write(out/"cleanup.json",dict(remaining_owned_pids=remaining,port_open_after_cleanup=opened,
                gpu0_memory_mib_after=gpu_memory(),recognition_request_attempts=reader.calls if reader else 0,
                valid_responses_this_invocation=reader.valid_responses if reader else 0,
                sampled_gpu0_memory_mib_max=reader.max_memory if reader else 0,elapsed_seconds=time.perf_counter()-started))
            if remaining:raise RuntimeError("owned reader survived cleanup")


def execute(args):
    run=args.run.resolve();out=(args.out or run/"neural_transcription_v80").resolve()
    if not out.is_relative_to(run) or out==run:raise ValueError("separate output inside run required")
    if (out/"report.json").exists():raise FileExistsError("completed diagnostic is immutable")
    if args.prepare_only:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("explicit CPU-only preparation required")
    else:ready(run)  # Real completion guard precedes preparation, service startup and model calls.
    identity,calibration=prepare(run,out)
    if args.prepare_only:
        print((out/"preparation.json").read_text());return
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda a,b:(_ for _ in ()).throw(KeyboardInterrupt()))
    with owned_reader(run,out,args.local_root.resolve(),args.port,identity) as reader:
        recognized={}
        for i,key in enumerate(sorted({r["image_sha256"] for r in calibration})):
            recognized[key]=reader.recognize(out/"calibration_images"/(key+".png"))
            print(json.dumps(dict(stage="calibration",completed=i+1)),flush=True)
        rows=[]
        for row in calibration:
            value=recognized[row["image_sha256"]]
            rows.append({**row,**value,"exact":bool(value["valid_json"] and normalize(value["text"])==normalize(row["expected_text"]))})
        write(out/"calibration_scores.json",rows)
        measured={kind:dict(cases=sum(r["kind"]==kind for r in rows),exact=sum(r["kind"]==kind and r["exact"] for r in rows)) for kind in ("cmrc","dolly","blank","random_ink")}
        controls=measured["blank"]["exact"]==16 and measured["random_ink"]["exact"]>=15 and all(r["valid_json"] for r in rows)
        qualified={source:controls and measured[source]["exact"]>=122 for source in ("cmrc","dolly")}
        result=dict(identity_sha256=sha(out/"identity.json"),calibration=measured,qualified_sources=qualified,
            actor_images_transcribed=0,primary_gates_unchanged=True,development_reused=True,
            shared_foundation_with_actor=True,ground_truth_semantic_accuracy=False,general_ilm_goal_complete=False)
        if any(qualified.values()):
            import torch
            from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
            from scripts.score_native_training_samples_v80 import check_answer
            data=CodecLanguageAuditData(ROOT/"data/general_instructions/codec_language_audit_v68_20261105")
            pending=[];images={}
            for i,row in enumerate(data.cases):
                if not qualified[row["source"]]:continue
                for route in ("normal","blank"):
                    folder=run/"development/predictions"/f"{i:03}_{route}"
                    header=json.loads((folder/"header.json").read_text())
                    if sha(folder/"answer.pt")!=header["output"]["raw_sha256"]:raise ValueError("raw actor answer changed")
                    value=torch.load(folder/"answer.pt",map_location="cpu",weights_only=True);pixels=check_answer(value)
                    key=None
                    if pixels.shape[1]:
                        if not np.array_equal(np.asarray(Image.open(folder/"answer.png")),pixels):raise ValueError("preview differs from actual pixels")
                        key,_=view(pixels,out/"actor_images");images[key]=out/"actor_images"/(key+".png")
                    pending.append(dict(source=row["source"],id=row["id"],article=row["article"],route=route,image_sha256=key))
            for i,key in enumerate(sorted(images)):
                recognized[key]=reader.recognize(images[key]);print(json.dumps(dict(stage="actor_images",completed=i+1,total=len(images))),flush=True)
            # Gold strings are first opened after all blind recognition responses are frozen.
            paths=dict(cmrc=ROOT/"data/visual_qa/cmrc2018_short/development.jsonl",dolly=ROOT/"data/general_instructions/dolly15k/development.jsonl")
            labels={}
            for source,path in paths.items():
                if sha(path)!=data.report["original_source_sha256"][source]:raise ValueError("original reference strings changed")
                labels[source]={r["id"]:r for r in map(json.loads,path.read_text().splitlines())}
            scored=[]
            for row in pending:
                value=recognized[row["image_sha256"]] if row["image_sha256"] else dict(valid_json=True,text="",readable=False)
                record=labels[row["source"]][row["id"]]
                refs=[r["text"] for r in record["references"]] if row["source"]=="cmrc" else [record["answer_flat"]]
                value_scores=scores(value["text"],refs) if value["valid_json"] else None
                intervals={m:[value_scores[m],value_scores[m]] if value_scores else [0.,1.] for m in ("text_exact","bounded_character_accuracy")}
                scored.append(dict(**row,text=value["text"],valid_json=value["valid_json"],intervals=intervals))
            write(out/"actor_transcripts.json",scored)
            result.update(actor_images_transcribed=len(images),actor_cases=len(scored),unknown_cases=sum(not r["valid_json"] for r in scored),
                measurements={source:{route:{m:np.mean([r["intervals"][m] for r in scored if r["source"]==source and r["route"]==route],axis=0).tolist()
                    for m in ("text_exact","bounded_character_accuracy")} for route in ("normal","blank")} for source in paths if qualified[source]})
            if qualified["cmrc"]:
                by={(r["id"],r["route"]):r for r in scored if r["source"]=="cmrc"};ids=[r["id"] for r in data.cases if r["source"]=="cmrc"]
                lower=[];upper=[];articles=[]
                for ident in ids:
                    a=by[ident,"normal"];b=by[ident,"blank"]
                    lower.append(a["intervals"]["bounded_character_accuracy"][0]-b["intervals"]["bounded_character_accuracy"][1])
                    upper.append(a["intervals"]["bounded_character_accuracy"][1]-b["intervals"]["bounded_character_accuracy"][0]);articles.append(a["article"])
                lo,hi=bootstrap(lower,articles),bootstrap(upper,articles)
                pairs=[[np.prod([by[x,"normal"]["intervals"]["text_exact"][k] for x in pair]) for k in (0,1)] for pair in data.report["chosen_pairs"]]
                result.update(chinese_both_correct_pair_interval=np.mean(pairs,axis=0).tolist(),
                    chinese_question_effect=dict(identification_interval=[lo["mean"],hi["mean"]],ci95_sensitivity_envelope=[lo["ci95"][0],hi["ci95"][1]],
                        bootstrap_seed=20266379,replicates=2000,reader_errors_not_covered=True))
        result["decision"]="auxiliary_transcription_complete" if any(qualified.values()) else "invalid_neural_instrument"
        result["unique_images_read_or_cached"]=len(recognized)
    result["cleanup_sha256"]=sha(out/"cleanup.json");write(out/"report.json",result);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--local-root",type=Path,default=ROOT.parent/"LocalLLM");parser.add_argument("--port",type=int,default=11437)
    args=parser.parse_args()
    try:execute(args)
    except PipelinePending:
        raise  # An expected sequencing refusal must not create a failure artifact.
    except BaseException as error:
        if args.out or (args.run/"neural_transcription_v80").exists():
            target=args.out or args.run/"neural_transcription_v80"
            if target.resolve().is_relative_to(args.run.resolve()) and target.resolve()!=args.run.resolve():write(target/"failure.json",dict(error=repr(error),completed_score=False))
        raise

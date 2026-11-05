#!/usr/bin/env python3
"""Offline native-source visual capability check, outside independent ILM inference."""

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.blind_vision_transcription import MODEL, gpu_memory, owned_reader
from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from scripts.verify_codec_language_v68 import independent_actor_prompt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/general_instructions/codec_language_audit_v68_20261105"
TASK=("Read the question or instruction in the supplied material. Answer it using "
      "any accompanying passage. Return only the requested answer, without an "
      "explanation or transcription of the full material. If there is no question "
      "or instruction, use an empty answer.")
SCHEMA=dict(type="object",properties=dict(answer=dict(type="string")),required=["answer"],additionalProperties=False)
OPTIONS=dict(temperature=0,top_k=1,top_p=1,repeat_penalty=1,seed=20267001,num_ctx=4096,num_predict=512)
ROUTES=("image","image_only","blank","text")
SOURCES=("scripts/native_visual_baseline_v70.py","scripts/verify_native_visual_baseline_v70.py",
    "references/native_visual_baseline_v70_protocol.md","ilm/visual_lm/blind_vision_transcription.py",
    "ilm/visual_lm/codec_language_audit.py","ilm/visual_lm/complete_writing_data.py",
    "scripts/verify_codec_language_v68.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(".partial")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def identity():
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    sources={name:sha(ROOT/name) for name in SOURCES}
    for name,digest in sources.items():
        raw=subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError("commit the fixed diagnostic before execution")
    return revision,sources


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("prepare and score with CUDA hidden")
    torch.set_num_threads(4)


def selected_cases():
    data=CodecLanguageAuditData(DATA)
    cache=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_development_20261105",
        purpose="development",expected_report_sha256=data.report["development_report_sha256"])
    original={}
    for source,path in cache.report["source_paths"].items():
        if sha(ROOT/path)!=cache.report["source_sha256"][source]:
            raise ValueError("original source strings changed")
        lines=[json.loads(x) for x in (ROOT/path).read_text().splitlines()]
        original[source]={row["id"]:row for row in lines}
        if len(lines)!=len(original[source]):
            raise ValueError("source identifiers are not unique")
    if cache.report["source_sha256"]!=data.report["original_source_sha256"]:
        raise ValueError("source strings differ from fixed audit provenance")
    indices=[i for i,r in enumerate(data.cases) if r["source"]=="cmrc"][:16]
    for category in sorted({r["category"] for r in data.cases if r["source"]=="dolly"}):
        indices.append(min((r["id"],i) for i,r in enumerate(data.cases)
            if r["source"]=="dolly" and r["category"]==category)[1])
    excluded={(r["id"],r["alternative"]) for r in data.report["excluded_alternative_references"]}
    result=[]
    for position,index in enumerate(indices):
        row=data.cases[index];text=original[row["source"]][row["id"]]
        references=([r["text"] for j,r in enumerate(text["references"]) if (row["id"],j) not in excluded]
            if row["source"]=="cmrc" else [text["answer_flat"]])
        if len(references)!=len(data.references(index)) or not all(isinstance(x,str) and x.strip() for x in references):
            raise ValueError("retained source reference identity differs")
        pages={}
        for route in ("normal","blank"):
            value=data.prompt(index,blank=route=="blank")
            independent=independent_actor_prompt(data.payload(index),blank=route=="blank")
            if not torch.equal(value,independent):
                raise ValueError("input page differs from independent physical layout")
            pages[route]=((value[0,0]+1)*127.5).to(torch.uint8).numpy()
        result.append(dict(position=position,audit_index=index,source=row["source"],id=row["id"],
            article=row["article"],category=row["category"],references=references,
            source_prompt=text["prompt"] if row["source"]=="cmrc" else text["prompt_flat"],pages=pages))
    if len(result)!=24 or len({r["article"] for r in result if r["source"]=="cmrc"})!=8:
        raise ValueError("fixed 24 cases and eight Chinese pairs required")
    return result


def prepare(out):
    cpu_only();revision,sources=identity();cases=selected_cases()
    out.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for row in cases:
        record={k:v for k,v in row.items() if k!="pages"};record["images"]={}
        for route,page in row["pages"].items():
            name=f"case_{row['position']:02}_{route}.png"
            Image.fromarray(page).convert("RGB").save(out/name)
            record["images"][route]=dict(file=name,sha256=sha(out/name),shape=list(page.shape),
                pixels_sha256=hashlib.sha256(page.tobytes()).hexdigest())
        manifest.append(record)
    atomic_json(out/"preparation.json",dict(source_commit=revision,sources=sources,cases=manifest,
        model_calls=0,gpu_allocations=0,sealed_examples=0,development_reused=True,
        task=TASK,options=OPTIONS,schema=SCHEMA,routes=list(ROUTES),source_data_sha256=sha(DATA/"report.json")))


def build_request(route,*,png=None,text=None):
    """No identifiers, labels or references are accepted at this boundary."""
    if route not in ROUTES:
        raise ValueError("unknown route")
    if route=="text":
        if png is not None or not isinstance(text,str):
            raise ValueError("text-only route requires source prompt alone")
        message=dict(role="user",content=TASK+"\n\n"+text)
    else:
        if not isinstance(png,bytes) or text is not None:
            raise ValueError("image route accepts PNG bytes only")
        message=dict(role="user",content="" if route=="image_only" else TASK,
            images=[base64.b64encode(png).decode("ascii")])
    return dict(model=MODEL,messages=[message],stream=False,think=False,
        format=copy.deepcopy(SCHEMA),options=dict(OPTIONS),keep_alive="1m")


def request_for(out,row,route):
    if route=="text":
        return build_request(route,text=row["source_prompt"])
    image=row["images"]["blank" if route=="blank" else "normal"]
    if sha(out/image["file"])!=image["sha256"]:
        raise ValueError("request image changed")
    return build_request(route,png=(out/image["file"]).read_bytes())


def query(out):
    revision,sources=identity();prep=json.loads((out/"preparation.json").read_text())
    if prep["sources"]!=sources:
        raise ValueError("fixed request/preparation implementation changed")
    if (out/"query_report.json").exists():
        raise FileExistsError("completed native replies are immutable")
    if os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("physical GPU 0 required for the isolated native source")
    lock=(out/"operation.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/"responses").mkdir(exist_ok=True);begun=time.perf_counter();files={};actual=0
    with owned_reader(ROOT/"artifacts/question_assignment_v63_20261105",out/"native_runtime",ROOT.parent/"LocalLLM",11437) as reader:
        for row in prep["cases"]:
            for route in ROUTES:
                payload=request_for(out,row,route)
                path=out/"responses"/f"case_{row['position']:02}_{route}.json"
                if path.exists():
                    saved=json.loads(path.read_text())
                    if saved["request"]!=payload or saved["request_sha256"]!=canonical_sha(payload):
                        raise ValueError("partial request identity changed")
                else:
                    started=time.perf_counter()
                    try:
                        response=reader.session.post(reader.base_url+"/api/chat",json=payload,timeout=(5,180))
                        response.raise_for_status();raw=response.json()
                    except Exception as error:
                        atomic_json(out/"transport_failure.json",dict(position=row["position"],route=route,
                            request_sha256=canonical_sha(payload),error=repr(error)))
                        raise
                    actual+=1;reader.actual_calls+=1
                    resident=gpu_memory()
                    reader.sampled_gpu0_memory_mib_max=max(reader.sampled_gpu0_memory_mib_max,resident)
                    saved=dict(position=row["position"],route=route,request=payload,
                        request_sha256=canonical_sha(payload),response=raw,
                        elapsed_seconds=time.perf_counter()-started,gpu0_resident_memory_mib=resident)
                    atomic_json(path,saved)
                    if resident>=20*1024:
                        raise RuntimeError("native source exceeded GPU 0 memory ceiling")
                files[str(path.relative_to(out))]=sha(path)
                atomic_json(out/"status.json",dict(stage="querying_native_source",pid=os.getpid(),completed=len(files),total=96))
                print(json.dumps(dict(case=row["position"],route=route,completed=len(files))),flush=True)
    cleanup=json.loads((out/"native_runtime/cleanup.json").read_text())
    if cleanup["remaining_owned_live_pids"] or cleanup["port_open_after_cleanup"]:
        raise RuntimeError("owned native service not cleaned up")
    atomic_json(out/"query_report.json",dict(source_commit=revision,sources=sources,
        preparation_sha256=sha(out/"preparation.json"),responses_sha256=files,
        cleanup_sha256=sha(out/"native_runtime/cleanup.json"),actual_requests=actual,total_responses=96,
        elapsed_seconds=time.perf_counter()-begun,actor_updates=0,actor_calls=0,source_native_text_decoder=True,
        language_quality_claim=False,general_ilm_goal_complete=False))
    atomic_json(out/"status.json",dict(stage="native_replies_complete",pid=None))


def parse(raw):
    try:
        content=raw["message"]["content"]
        value=json.loads(content)
        if not isinstance(value,dict) or set(value)!={"answer"} or not isinstance(value["answer"],str):
            raise ValueError("answer schema differs")
        return dict(valid_json=True,answer=value["answer"],completed=raw.get("done") is True and raw.get("done_reason")=="stop")
    except (KeyError,ValueError,TypeError):
        return dict(valid_json=False,answer=None,completed=False)


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,letter in enumerate(a,1):
        current=[i]
        for j,other in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(letter!=other)))
        previous=current
    return previous[-1]


def score_response(raw,references):
    value=parse(raw)
    if not value["valid_json"]:
        return dict(**value,exact=False,character_accuracy=0.)
    answer=value["answer"].strip();references=[x.strip() for x in references]
    agreement=value["completed"] and answer in references
    accuracy=max(max(0.,1-edit_distance(answer,ref)/max(len(answer),len(ref),1)) for ref in references)
    return dict(**value,exact=agreement,character_accuracy=accuracy)


def score(out):
    cpu_only();revision,sources=identity()
    if (out/"score_report.json").exists():
        raise FileExistsError("completed scores are immutable")
    query_report=json.loads((out/"query_report.json").read_text());prep=json.loads((out/"preparation.json").read_text())
    if query_report["sources"]!=sources or query_report["preparation_sha256"]!=sha(out/"preparation.json"):
        raise ValueError("request source identity changed")
    measured=[]
    for row in prep["cases"]:
        for route in ROUTES:
            relative=f"responses/case_{row['position']:02}_{route}.json"
            if sha(out/relative)!=query_report["responses_sha256"][relative]:
                raise ValueError("raw response changed")
            raw=json.loads((out/relative).read_text())
            measured.append(dict(position=row["position"],route=route,
                **score_response(raw["response"],row["references"])))
    summaries={};pairs={}
    for source in ("cmrc","dolly"):
        for route in ROUTES:
            chosen=[m for m in measured if m["route"]==route and prep["cases"][m["position"]]["source"]==source]
            summaries[source+"/"+route]=dict(count=len(chosen),valid_json=sum(m["valid_json"] for m in chosen),
                completed=sum(m["completed"] for m in chosen),exact=sum(m["exact"] for m in chosen),
                character_accuracy=float(np.mean([m["character_accuracy"] for m in chosen])))
    articles=sorted({r["article"] for r in prep["cases"] if r["source"]=="cmrc"})
    for route in ROUTES:
        pairs[route]=sum(all(m["exact"] for m in measured if m["route"]==route and prep["cases"][m["position"]]["article"]==article) for article in articles)
    atomic_json(out/"score_report.json",dict(source_commit=revision,sources=sources,
        query_report_sha256=sha(out/"query_report.json"),measurements=measured,summaries=summaries,
        chinese_both_exact_pairs=pairs,chinese_pairs=8,source_helper_only=True,
        native_tokenized_text_output=True,development_reused=True,sealed_examples=0,
        english_reference_agreement_only=True,language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps(dict(summaries=summaries,chinese_both_exact_pairs=pairs),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--stage",choices=("prepare","query","score"),required=True)
    args=parser.parse_args();{"prepare":prepare,"query":query,"score":score}[args.stage](args.out)

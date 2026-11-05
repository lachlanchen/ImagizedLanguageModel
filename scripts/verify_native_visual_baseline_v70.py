#!/usr/bin/env python3
"""Independently bind native source replies to pixels, prompts, scores and cleanup."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import socket
import time

import numpy as np
from PIL import Image

from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from scripts.native_visual_baseline_v70 import ROOT,DATA,TASK,SCHEMA,OPTIONS,ROUTES,MODEL,cpu_only,identity,sha,atomic_json


def distance(a,b):
    """Vectorized row recurrence, independent of the producer's scalar DP."""
    letters=np.array([ord(x) for x in b],dtype=np.int64)
    positions=np.arange(len(b)+1,dtype=np.int64);previous=positions.copy()
    for i,letter in enumerate(a,1):
        candidate=np.empty_like(previous);candidate[0]=i
        candidate[1:]=np.minimum(previous[1:]+1,previous[:-1]+(letters!=ord(letter)))
        previous=np.minimum.accumulate(candidate-positions)+positions
    return int(previous[-1])


def verify(out):
    cpu_only();begun=time.perf_counter();revision,sources=identity()
    prep=json.loads((out/"preparation.json").read_text())
    query=json.loads((out/"query_report.json").read_text())
    score=json.loads((out/"score_report.json").read_text())
    if any(d["sources"]!=sources for d in (prep,query,score)):
        raise ValueError("source implementation changed")
    if (query["preparation_sha256"]!=sha(out/"preparation.json")
            or score["query_report_sha256"]!=sha(out/"query_report.json")
            or query["total_responses"]!=96 or len(query["responses_sha256"])!=96):
        raise ValueError("complete source-bound requests required")
    data=CodecLanguageAuditData(DATA)
    cache=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_development_20261105",
        purpose="development",expected_report_sha256=data.report["development_report_sha256"])
    original={}
    for source,path in cache.report["source_paths"].items():
        if sha(ROOT/path)!=cache.report["source_sha256"][source]:
            raise ValueError("original source text changed")
        original[source]={r["id"]:r for r in map(json.loads,(ROOT/path).read_text().splitlines())}
    expected=[i for i,r in enumerate(data.cases) if r["source"]=="cmrc"][:16]
    for category in sorted({r["category"] for r in data.cases if r["source"]=="dolly"}):
        expected.append(sorted((r["id"],i) for i,r in enumerate(data.cases)
            if r["source"]=="dolly" and r["category"]==category)[0][1])
    if len(prep["cases"])!=24 or [r["audit_index"] for r in prep["cases"]]!=expected:
        raise ValueError("original deterministic case selection differs")
    excluded={(r["id"],r["alternative"]) for r in data.report["excluded_alternative_references"]}
    measurements=[];image_checks=0
    for position,row in enumerate(prep["cases"]):
        source_row=data.cases[expected[position]];text=original[row["source"]][row["id"]]
        references=([r["text"] for i,r in enumerate(text["references"]) if (row["id"],i) not in excluded]
            if row["source"]=="cmrc" else [text["answer_flat"]])
        source_prompt=text["prompt"] if row["source"]=="cmrc" else text["prompt_flat"]
        if (row["position"]!=position or row["source"]!=source_row["source"] or row["id"]!=source_row["id"]
                or row["references"]!=references or row["source_prompt"]!=source_prompt
                or row["article"]!=source_row["article"] or row["category"]!=source_row["category"]):
            raise ValueError("case metadata/source binding differs")
        for condition in ("normal","blank"):
            entry=row["images"][condition];path=out/entry["file"]
            actual=np.asarray(Image.open(path).convert("RGB"))
            reference=((data.prompt(expected[position],blank=condition=="blank")[0,0]+1)*127.5).numpy().astype(np.uint8)
            if (sha(path)!=entry["sha256"] or list(reference.shape)!=entry["shape"]
                    or hashlib.sha256(reference.tobytes()).hexdigest()!=entry["pixels_sha256"]
                    or actual.shape!=(*reference.shape,3) or not np.array_equal(actual,np.repeat(reference[:,:,None],3,axis=2))):
                raise ValueError("source PNG pixels differ")
            image_checks+=1
        for route in ROUTES:
            relative=f"responses/case_{position:02}_{route}.json"
            if sha(out/relative)!=query["responses_sha256"][relative]:
                raise ValueError("raw native response changed")
            raw=json.loads((out/relative).read_text())
            if route=="text":
                message=dict(role="user",content=TASK+"\n\n"+source_prompt)
            else:
                name=row["images"]["blank" if route=="blank" else "normal"]["file"]
                message=dict(role="user",content="" if route=="image_only" else TASK,
                    images=[base64.b64encode((out/name).read_bytes()).decode("ascii")])
            request=dict(model=MODEL,messages=[message],stream=False,think=False,
                format=SCHEMA,options=OPTIONS,keep_alive="1m")
            digest=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
            if raw["position"]!=position or raw["route"]!=route or raw["request"]!=request or raw["request_sha256"]!=digest:
                raise ValueError("request contains different inputs or hidden labels")
            response=raw["response"]
            if response.get("model")!=MODEL:
                raise ValueError("native response reports a different source model")
            try:
                parsed=json.loads(response["message"]["content"])
                valid=isinstance(parsed,dict) and set(parsed)=={"answer"} and isinstance(parsed["answer"],str)
            except (KeyError,ValueError,TypeError):
                valid=False
            answer=parsed["answer"] if valid else None
            completed=valid and response.get("done") is True and response.get("done_reason")=="stop"
            exact=bool(completed and answer.strip() in [r.strip() for r in references])
            accuracy=max(max(0.,1-distance(answer.strip(),ref.strip())/max(len(answer.strip()),len(ref.strip()),1))
                for ref in references) if valid else 0.
            measurements.append(dict(position=position,route=route,valid_json=valid,answer=answer,
                completed=completed,exact=exact,character_accuracy=accuracy))
    if len(score["measurements"])!=96:
        raise ValueError("all native measurements required")
    for actual,claimed in zip(measurements,score["measurements"]):
        if any(actual[k]!=claimed[k] for k in actual if k!="character_accuracy") or abs(actual["character_accuracy"]-claimed["character_accuracy"])>1e-12:
            raise ValueError("independent literal-reference score differs")
    for group,claimed in score["summaries"].items():
        source,route=group.split("/")
        values=[m for m in measurements if m["route"]==route and prep["cases"][m["position"]]["source"]==source]
        if any(sum(v[key] for v in values)!=claimed[key] for key in ("valid_json","completed","exact")):
            raise ValueError("summary counts differ")
        if claimed["count"]!=len(values) or abs(np.mean([v["character_accuracy"] for v in values])-claimed["character_accuracy"])>1e-12:
            raise ValueError("summary sample or character score differs")
    articles=sorted({r["article"] for r in prep["cases"] if r["source"]=="cmrc"})
    pairs={route:sum(all(v["exact"] for v in measurements if v["route"]==route
        and prep["cases"][v["position"]]["article"]==article) for article in articles) for route in ROUTES}
    if pairs!=score["chinese_both_exact_pairs"] or len(articles)!=8:
        raise ValueError("source-article pair scores differ")
    cleanup=json.loads((out/"native_runtime/cleanup.json").read_text())
    runtime=json.loads((out/"native_runtime/owned_runtime.json").read_text())
    if (sha(out/"native_runtime/cleanup.json")!=query["cleanup_sha256"]
            or cleanup["remaining_owned_live_pids"] or cleanup["port_open_after_cleanup"]
            or cleanup["sampled_gpu0_memory_mib_max"]>=20*1024 or runtime["physical_gpu"]!=0
            or cleanup["actual_recognition_calls"]!=query["actual_requests"]):
        raise ValueError("owned native runtime cleanup/resource check failed")
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1",11437))==0:
            raise ValueError("dedicated native source port remains open")
    result=dict(score_report_sha256=sha(out/"score_report.json"),query_report_sha256=sha(out/"query_report.json"),
        verifier_sha256=sha(Path(__file__)),source_model_sha256=runtime["model_weight_sha256"],
        source_manifest_sha256=runtime["model_manifest_sha256"],owned_binary_sha256=runtime["binary_sha256"],
        source_pngs_verified=image_checks,requests_and_responses_verified=96,measurements_recomputed=96,
        summaries=score["summaries"],chinese_both_exact_pairs=pairs,
        sampled_gpu0_resident_mib_max=cleanup["sampled_gpu0_memory_mib_max"],
        gpu0_resident_mib_after_cleanup=cleanup["gpu0_memory_mib_after"],owned_process_cleanup_verified=True,
        offline_checks_pass=True,source_helper_only=True,native_tokenized_text_output=True,
        actor_updates=0,actor_inference_calls=0,language_quality_claim=False,general_ilm_goal_complete=False,
        elapsed_verification_seconds=time.perf_counter()-begun)
    atomic_json(out/"verification.json",result)
    atomic_json(out/"status.json",dict(stage="completed_verified",pid=None,language_quality_claim=False))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    verify(parser.parse_args().out)

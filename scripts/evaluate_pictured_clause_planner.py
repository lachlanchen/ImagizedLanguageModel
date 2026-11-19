"""Measure learned plans, true pictured spans, and deliberately reversed plans."""
from collections import Counter
from dataclasses import replace
import fcntl
import json
import os

import numpy as np
import torch

from ilm.visual_lm.pictured_clause_planner import PlannedGlyphResponder,subquestions
from scripts.evaluate_grounded_propositions import assess,generate,metrics,pictured,release_notes,store_streams
from scripts.extend_grounded_propositions import pair_key,WITHHELD
from scripts.grounded_proposition_corpus import WORK,PUBLIC,Face,FONTS,materialize,LANG,label
from scripts.pointing_glyph_corpus import Row,load
from scripts.train_grounded_propositions import ROOT,boards,record,sha
from scripts.train_pictured_clause_planner import gold_spans


def streams_for(responder,rows,meta=None,compound=False):
    """Only the explicit oracle condition consults the annotated relation positions."""
    source_pairs={};plans={k:[] for k in (("predicted","oracle","reversed") if compound else ("predicted",))}
    plan_correct=counts_correct=0
    for start in range(0,len(rows),64):
        chunk=rows[start:start+64]
        doubles,spans=responder.plans([r.q for r in chunk])
        for j,(r,double,pos) in enumerate(zip(chunk,doubles,spans)):
            selected=subquestions(r.q,pos) if double else (r.q,)
            choices={"predicted":selected}
            counts_correct+=double==compound
            if compound:
                m=meta[start+j]
                oracle=subquestions(r.q,gold_spans(m["q"],m["facts"]))
                plan_correct+=selected==oracle
                choices.update(oracle=oracle,reversed=selected[::-1])
            for arm,qs in choices.items():
                calls=[source_pairs.setdefault((r.p,q),len(source_pairs)) for q in qs]
                plans[arm].append(calls)
    inputs=[Row("plan",str(i),p,q,p[:128],q) for i,(p,q) in enumerate(source_pairs)]
    written,ends=generate(responder.net,inputs)
    output={}
    for arm,calls in plans.items():
        streams=[];stops=[]
        for js in calls:
            y=b"".join(written[j] for j in js)
            stops.append(all(ends[j] for j in js) and len(y)<=48*128)
            streams.append(y[:48*128])
        output[arm]=streams,stops
    return output,dict(count_correct=counts_correct,exact_plan=plan_correct if compound else None,n=len(rows))


def inverse_wording(rows,meta):
    face=Face((FONTS[0],2))
    rr=[];mm=[]
    for row,m in zip(rows,meta):
        (s,r,_),(_,t,_)=m["facts"]
        q=f"为{label(s,r)}写两句。第二句说明{LANG[t][0]}，排在它前面的那句说明{LANG[r][0]}。"
        rr.append(replace(row,q=face(q)));mm.append(dict(m,q=q))
    return rr,mm


def combination_counts(cases,held):
    by={}
    for r in cases:
        k="withheld" if pair_key(held[r["board"]]) in WITHHELD else "practiced"
        by.setdefault(k,{}).setdefault(r["board"],[]).append(r)
    results={}
    for k,groups in by.items():
        cs=[v for rows in groups.values() for v in rows]
        results[k]=dict(n=len(cs),exact=sum(r["exact"] for r in cs),equivalent=sum(r["equivalent"] for r in cs),
            objects=sum(r["objects"] for r in cs),boards=len(groups),
            complete=sum(len(g)==4 and all(r["equivalent"] for r in g) for g in groups.values()))
    return results


def main():
    folder=WORK/"planned"
    fit=json.loads((PUBLIC/"planned_fit.json").read_text())
    assert all(sha(folder/k)==v for k,v in fit["weights"].items())
    assert sha(folder/"plan.safetensors")==json.loads((PUBLIC/"plan_head.json").read_text())["weights_sha256"]
    net=PlannedGlyphResponder(folder,"cuda")
    held=boards()["development"];face=Face((FONTS[0],2))
    compound=materialize(held,"development",True)
    suites=dict(single=materialize(held,"development"),compound=compound,inverted=inverse_wording(*compound),notes=release_notes())
    result={}
    for k,(rows,meta) in suites.items():
        is_two=k in ("compound","inverted")
        outputs,accuracy=streams_for(net,rows,meta,is_two)
        result[k]={"planner":accuracy}
        for arm,(y,end) in outputs.items():
            file=WORK/(f"planned_{k}.npz" if arm=="predicted" else f"planned_{arm}_{k}.npz")
            store_streams(file,y,end)
            stats,cases=assess(rows,meta,y,end,face)
            result[k][arm]=dict(scores=stats,stream_sha256=sha(file))
            if is_two:result[k][arm]["combinations"]=combination_counts(cases,held)
            record(WORK/f"planned_{arm}_{k}_cases.json",cases)
        print(k,result[k]["predicted"].get("combinations",result[k]["planner"]),flush=True)
    prior=load()[1]
    rr=[r for r in prior if r.task in ("absent_content_ink","single_span","distributed_ink","historical","historical_rephrased")]
    out,counts=streams_for(net,rr)
    y,end=out["predicted"]
    store_streams(WORK/"planned_regression.npz",y,end)
    result.update(regression=metrics(rr,y,end),single_invocations_regression=counts,regression_sha256=sha(WORK/"planned_regression.npz"))
    for arm in ("matched","ordered","unordered","varied","replay"):
        cs=json.loads((WORK/(arm+"_compound_cases.json")).read_text())
        result.setdefault("monolithic_compounds",{})[arm]=combination_counts(cs,held)
    record(PUBLIC/"planned_result.json",result)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);main()

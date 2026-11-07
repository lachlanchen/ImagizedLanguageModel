#!/usr/bin/env python3
"""Audit physical supervision and erasure geometry without invoking a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.semantic_image_memory import native_budget_image

ROOT = Path(__file__).absolute().parents[1]


def read(path): return json.loads(Path(path).read_text())
def pixels(path): return np.array(Image.open(path).convert("L"),copy=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trim(image):
    ink=np.flatnonzero((image==0).any(0))
    return image[:,ink[0]:ink[-1]+1] if len(ink) else image[:,:0]


def physical_occurrences(image,target):
    """Search exact column bytes, then enforce admissible physical boundaries."""
    occupied=(image==0).any(0);edges=np.diff(np.r_[False,occupied,False].astype(np.int8))
    left,right=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    starts={int(x):i for i,x in enumerate(left)};ends={int(x):i for i,x in enumerate(right)}
    target=trim(target)
    if not target.shape[1]:raise ValueError("blank reference is not a writing answer")
    stream=image.T.copy().tobytes();needle=target.T.copy().tobytes();position=0;found=[]
    while True:
        position=stream.find(needle,position)
        if position<0:break
        if position%image.shape[0]==0:
            column=position//image.shape[0];start=starts.get(column);end=ends.get(column+target.shape[1])
            if start is not None and end is not None and 0<=end-start<128:found.append((start,end-start+1))
        position+=1
    return found


def physical(image):
    return torch.from_numpy(image.copy()).float().div(127.5).sub(1)[None,None]


def same_retained_coordinates(a,b,fa,fb,columns=None):
    common=(a==0)&(b==0)
    if columns is not None:common&=columns[None]
    y,x=np.nonzero(common)
    def coordinates(f):
        ends=np.array([end for _,end in f.source_ranges])
        starts=np.array([start for start,_ in f.source_ranges])
        row=np.searchsorted(ends,x,side="right")
        return row*32+y,x-starts[row]
    ya,xa=coordinates(fa);yb,xb=coordinates(fb)
    return dict(common_ink_pixels=len(x),shifted_common_ink_pixels=int(((ya!=yb)|(xa!=xb)).sum()))


def audit(prepared,previous):
    report=read(prepared/"report.json")
    if sha(prepared/"inputs.json")!=report["inputs_sha256"] or sha(prepared/"references.json")!=report["references_sha256"]:
        raise ValueError("prepared inventories changed")
    rows=read(prepared/"inputs.json");refs={r["id"]:r for r in read(prepared/"references.json")}
    matches=Counter();source_files={}
    for row in rows:
        path=ROOT/row["input"]
        if sha(path)!=row["input_sha256"]:raise ValueError("training input changed")
        image=pixels(path)
        if image.shape[0]!=32 or not np.isin(image,[0,255]).all():raise ValueError("invalid physical training pixels")
        source_files[row["input"]]=row["input_sha256"]
        for ref in refs[row["id"]]["targets"]:
            path=ROOT/ref["file"]
            if sha(path)!=ref["sha256"]:raise ValueError("reference changed")
            occurrences=physical_occurrences(image,pixels(path))
            if not occurrences:raise ValueError("reference is absent from admissible source columns: "+row["id"])
            source_files[ref["file"]]=ref["sha256"]
            matches["reference_images"]+=1;matches["source_occurrences"]+=len(occurrences)
    old_report=read(previous/"report.json")
    if sha(previous/"inputs.json")!=old_report["inputs_sha256"] or sha(previous/"references.json")!=old_report["references_sha256"]:
        raise ValueError("original development inventories changed")
    old=read(previous/"inputs.json");old_refs={r["id"]:r for r in read(previous/"references.json")}
    normal=[r for r in old if r["split"]=="development" and r["task"]=="qa" and r["condition"]=="normal"]
    training=[r for r in rows if r["task"]=="qa"]
    if {r["article"] for r in training}&{r["article"] for r in normal} or {r["pair"] for r in training}&{r["pair"] for r in normal}:
        raise ValueError("training/development article or paragraph overlap")
    by_id={r["id"]:r for r in old};controls=[];observer_shapes={}
    def shape(folded):
        key=tuple(folded.pixels.shape[-2:])
        if key not in observer_shapes:
            observed=native_budget_image(torch.ones(1,1,*key))
            observer_shapes[key]=list(observed.shape[-2:])
        return observer_shapes[key]
    for row in normal:
        path=previous/row["input"]
        if sha(path)!=row["input_sha256"]:raise ValueError("normal development input changed")
        a=pixels(path);fa=fold_writing_prompt(physical(a));targets=[]
        for target in old_refs[row["id"]]["targets"]:
            path=previous/target["file"]
            if sha(path)!=target["sha256"]:raise ValueError("development reference changed")
            targets.append(pixels(path))
        edge=np.diff(np.r_[False,(a==0).any(0),False].astype(np.int8))
        left,right=np.flatnonzero(edge==1),np.flatnonzero(edge==-1)
        reference_columns=np.zeros(a.shape[1],dtype=bool)
        for target in targets:
            for start,length in physical_occurrences(a,target):
                reference_columns[left[start]:right[start+length-1]]=True
        for condition in ("question_erased","passage_erased"):
            other=by_id[row["id"]+"/"+condition];path=previous/other["input"]
            if sha(path)!=other["input_sha256"]:raise ValueError("erased development input changed")
            b=pixels(path)
            if a.shape!=b.shape or not np.all((b==a)|(b==255)):
                raise ValueError("control changes source ink beyond erasure")
            fb=fold_writing_prompt(physical(b))
            ref_motion=same_retained_coordinates(a,b,fa,fb,reference_columns)
            controls.append(dict(id=row["id"],condition=condition,raw_shape_equal=True,
                folded_shape_equal=fa.pixels.shape==fb.pixels.shape,
                fold_ranges_equal=fa.source_ranges==fb.source_ranges,
                observer_shape_equal=shape(fa)==shape(fb),
                reference_still_admissible=any(physical_occurrences(b,t) for t in targets),
                reference_common_ink_pixels=ref_motion["common_ink_pixels"],
                shifted_reference_ink_pixels=ref_motion["shifted_common_ink_pixels"],
                **same_retained_coordinates(a,b,fa,fb)))
    if len(rows)!=3257 or len(normal)!=64 or len(controls)!=128:raise ValueError("complete declared cohort required")
    summaries={}
    for condition in ("question_erased","passage_erased"):
        group=[r for r in controls if r["condition"]==condition]
        summaries[condition]=dict(cases=len(group),**{key:sum(bool(r[key]) for r in group) for key in
            ("raw_shape_equal","folded_shape_equal","fold_ranges_equal","observer_shape_equal","reference_still_admissible")},
            cases_with_shifted_retained_ink=sum(r["shifted_common_ink_pixels"]>0 for r in group),
            shifted_retained_ink_pixels=sum(r["shifted_common_ink_pixels"] for r in group),
            cases_with_shifted_reference_ink=sum(r["shifted_reference_ink_pixels"]>0 for r in group),
            reference_common_ink_pixels=sum(r["reference_common_ink_pixels"] for r in group),
            shifted_reference_ink_pixels=sum(r["shifted_reference_ink_pixels"] for r in group))
    return dict(preparation_sha256=sha(prepared/"report.json"),training_inputs=len(rows),
        reference_checks=dict(matches),authenticated_files=len(source_files),
        controls=summaries,control_cases=controls,source_files_sha256=source_files,
        training_development_articles_disjoint=True,training_development_paragraphs_disjoint=True,
        model_invocations=0,optimizer_updates=0,language_quality_claim=False,general_ilm_goal_complete=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,default=ROOT/"artifacts/contextual_pixel_learning_v1_attempt02/prepared")
    p.add_argument("--previous",type=Path,default=ROOT/"artifacts/contextual_pixel_spans_v1/prepared")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    result=audit(args.prepared,args.previous);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("training_inputs","reference_checks","authenticated_files","controls")},indent=2))

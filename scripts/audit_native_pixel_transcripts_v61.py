#!/usr/bin/env python3
"""Read generated answers with a blind CPU recognizer, outside the actor."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import unicodedata

import numpy as np
from PIL import Image, ImageOps
import torch

from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json


def normalize(text):
    return "".join(unicodedata.normalize("NFC",text).split())


def distance(left,right):
    row=list(range(len(right)+1))
    for i,x in enumerate(left,1):
        nxt=[i]
        for j,y in enumerate(right,1):
            nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def scores(text,references):
    text=normalize(text);refs=[normalize(r) for r in references]
    error=min(distance(text,r)/len(r) for r in refs)
    return dict(text_exact=float(text in refs),normalized_character_error=error,
                bounded_character_accuracy=max(0.,1-error))


def bootstrap(values,articles):
    values,articles=np.asarray(values,float),np.asarray(articles)
    groups=[np.flatnonzero(articles==a) for a in sorted(set(articles))]
    totals=np.asarray([values[g].sum() for g in groups]);counts=np.asarray([len(g) for g in groups])
    draws=np.random.default_rng(20266319).integers(0,len(groups),(2000,len(groups)))
    means=totals[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                articles=len(groups),examples=len(values),replicates=2000,seed=20266319)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",default="artifacts/native_pixel_instruction_v61_20261105")
    a=p.parse_args();run=Path(a.run);out=run/"transcript_diagnostic"
    if (out/"report.json").exists():raise FileExistsError("completed transcript diagnostic is immutable")
    out.mkdir(parents=True,exist_ok=True)
    primary=json.loads((run/"development/report.json").read_text())["primary_ids"]
    records={r["id"]:r for r in read_questions("data/visual_qa/cmrc2018_short/development.jsonl")}
    renderer=VisualQARenderer();images={};keys={}

    def add(route,identifier,pixels):
        if not len(pixels):keys[route,identifier]=None;return
        bitmap=np.concatenate([x[0].numpy() for x in pixels],axis=1)
        im=Image.fromarray(np.round(255*(1-bitmap)).astype(np.uint8))
        im=ImageOps.expand(im.resize((im.width*4,im.height*4),Image.Resampling.NEAREST),border=16,fill=255)
        key=hashlib.sha256(np.asarray(im).tobytes()+str(im.size).encode()).hexdigest()
        keys[route,identifier]=key;images[key]=im

    routes=["trained","trained_blank","base","base_blank"]
    for identifier in primary:
        add("gold_reference",identifier,renderer(records[identifier]["references"][0]["text"]))
        for route in routes:
            row=torch.load(run/"development/predictions"/route/(identifier+".pt"),map_location="cpu",weights_only=False)
            add(route,identifier,row["images"])
    version=subprocess.check_output(["tesseract","--version"],text=True,stderr=subprocess.STDOUT).splitlines()[0]
    tessdata=Path("/usr/share/tesseract-ocr/5/tessdata")
    identity=dict(image_audit_sha256=sha256(run/"development/report.json"),tesseract_version=version,
        language_sha256={lang:sha256(tessdata/(lang+".traineddata")) for lang in ("chi_sim","eng")},
        source_sha256={name:sha256(name) for name in [__file__,"references/native_pixel_transcript_audit_plan.md"]},
        preprocessing="nearest-neighbour 4x; 16px white margin",languages="chi_sim+eng",oem=1,psm=7,
        recognizer_sees_question=False,recognizer_sees_reference_text=False,recognizer_sees_model_name=False)
    path=out/"identity.json"
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError("partial OCR identity differs")
    _atomic_json(identity,path)
    folder=out/"anonymous_images";folder.mkdir(exist_ok=True)
    cache=out/"recognizer_outputs";cache.mkdir(exist_ok=True)

    def recognize(item):
        key,im=item;result=cache/(key+".json")
        if result.exists():return key,json.loads(result.read_text())
        image=folder/(key+".png");im.save(image)
        response=subprocess.run(["tesseract",str(image),"stdout","-l","chi_sim+eng","--oem","1","--psm","7","tsv"],
            check=True,capture_output=True,text=True,env=dict(os.environ,OMP_THREAD_LIMIT="1"),timeout=45)
        words=[r for r in csv.DictReader(io.StringIO(response.stdout),delimiter="\t")
               if r.get("text","").strip() and float(r["conf"])>=0]
        value=dict(text=" ".join(r["text"] for r in words),
            character_weighted_confidence=sum(float(r["conf"])*len(r["text"]) for r in words)/sum(len(r["text"]) for r in words) if words else None,
            image_sha256=sha256(image))
        _atomic_json(value,result);return key,value

    # Unique anonymous images are transcribed once; no reference or route is an OCR argument.
    with ThreadPoolExecutor(max_workers=2) as pool:
        recognized=dict(pool.map(recognize,sorted(images.items())))
    outputs={}
    for route in routes+["gold_reference"]:
        rows=[]
        for identifier in primary:
            record=records[identifier];key=keys[route,identifier]
            result=recognized[key] if key is not None else dict(text="",character_weighted_confidence=None)
            refs=[r["text"] for r in record["references"]]
            if route=="gold_reference":refs=refs[:1]
            rows.append(dict(id=identifier,article=record["article_title"],**result,**scores(result["text"],refs)))
        outputs[route]=rows;_atomic_json(rows,out/(route+"_transcripts.json"))
    measured={route:{key:float(np.mean([r[key] for r in rows])) for key in
        ("text_exact","normalized_character_error","bounded_character_accuracy")} for route,rows in outputs.items()}
    comparisons={}
    for other in ("base","trained_blank"):
        for key in ("text_exact","bounded_character_accuracy"):
            comparisons[f"trained_minus_{other}_{key}"]=bootstrap(
                [x[key]-y[key] for x,y in zip(outputs["trained"],outputs[other])],
                [r["article"] for r in outputs["trained"]])
    report=dict(**identity,measurements=measured,comparisons=comparisons,primary_ids=primary,
        anonymous_unique_images=len(images),neural_actor_calls=0,primary_decision_unchanged=True,
        qualification="auxiliary noisy transcription; no new language qualification or historical-script claim")
    _atomic_json(report,out/"report.json");print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Verify direct writing against original image data, RNG draws and raw outputs."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT=Path(__file__).resolve().parents[1]
TRAIN_SHA="0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"
DEV_SHA="14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369"


def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):value.update(block)
    return value.hexdigest()


def image(path):
    value=np.asarray(Image.open(path).convert("L"))
    if value.shape!=(32,64) or not set(np.unique(value))<={0,255}:
        raise ValueError("raw binary physical image required")
    return value<128


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/direct_writer_learning_v66_20261105")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();run=args.run
    report=json.loads((run/"report.json").read_text())
    if (report["update"]!=10000 or report["implementation"]!="one-pass direct conditional-mean writing"
            or report["general_ilm_goal_complete"] or report["language_quality_claim"]):
        raise ValueError("complete direct-renderer-only result required")
    if report["final_ema_sha256"]!=sha(run/"writer_final.pt"):
        raise ValueError("final fixed writer differs")
    for name,digest in report["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("frozen learning source changed: "+name)
    resume=json.loads((run/"resume_verification.json").read_text())
    if not resume["passed"] or not all(resume["checks"].values()) or resume["extra_optimizer_replays"]!=1:
        raise ValueError("exact optimizer/EMA/RNG replay required")
    metrics=[json.loads(row) for row in (run/"metrics.jsonl").read_text().splitlines()]
    journal=[json.loads(row) for row in (run/"sample_order.jsonl").read_text().splitlines()]
    for rows in (metrics,journal):
        if [row["update"] for row in rows]!=list(range(1,10001)):
            raise ValueError("missing or extra learning updates")
    if any(not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm_before_clip"]) for row in metrics):
        raise ValueError("nonfinite recorded optimization")
    training=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_train_20261105",
        purpose="train",expected_report_sha256=TRAIN_SHA)
    selected=set()
    for row in journal:
        if len(row["selections"])!=16:raise ValueError("batch size differs")
        for index,_ in row["selections"]:
            if type(index) is not int or not 0<=index<len(training):raise ValueError("invalid source index")
            selected.add(index)
    block_counts={}
    for index in sorted(selected):
        name=training.names[index];raw=(training.path/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=training.report["files_sha256"][name]:
            raise ValueError("consumed training source changed")
        metadata=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
        block_counts[index]=metadata["width"]//64-metadata["prompt_blocks"]-2
    pools=[[i for i,name in enumerate(training.names) if name.startswith(prefix)] for prefix in ("cmrc/","dolly/")]
    rng=torch.Generator().manual_seed(20266621)
    for row in journal:
        expected=[]
        for pool in pools:
            for _ in range(8):
                index=pool[int(torch.randint(len(pool),(),generator=rng))]
                block=int(torch.randint(block_counts[index],(),generator=rng))
                expected.append([index,block])
        if row["selections"]!=expected:
            raise ValueError("recorded draws differ from original corpus and fixed sampling seed")
    manifest=json.loads((run/"development/manifest.json").read_text())
    original=ROOT/"artifacts/grounded_writer_learning_v64_20261105/development/manifest.json"
    if (run/"development/manifest.json").read_bytes()!=original.read_bytes() or len(manifest)!=128:
        raise ValueError("development selection differs from fixed V64 cases")
    development=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_development_20261105",
        purpose="development",expected_report_sha256=DEV_SHA)
    references=[]
    for i,row in enumerate(manifest):
        if development.names[row["index"]]!=row["file"]:raise ValueError("development source index differs")
        target=development.target(row["index"])[...,:-128]
        reference=target[0,0,:,row["block"]*64:(row["block"]+1)*64].numpy()<0
        if not np.array_equal(image(run/f"development/reference/{i:03}.png"),reference):
            raise ValueError("reference raster differs from its original development pixels")
        references.append(reference)
    cases=json.loads((run/"development/cases.json").read_text())
    measured={};hashes={}
    for route in ("aligned","shuffled"):
        values=[]
        for i,reference in enumerate(references):
            path=run/f"development/{route}/{i:03}.png";prediction=image(path)
            denominator=int(prediction.sum()+reference.sum())
            actual=dict(foreground_f1=2*int((prediction&reference).sum())/denominator if denominator else 1.,
                exact=bool(np.array_equal(prediction,reference)),blank=not bool(prediction.any()))
            if actual!=cases[i]["scores"][route] or any(cases[i][key]!=v for key,v in manifest[i].items()):
                raise ValueError("individual generated-image score or ordering differs")
            hashes[str(path.relative_to(run))]=sha(path);values.append(actual)
        measured[route]={key:float(np.mean([row[key] for row in values])) for key in values[0]}
        for key,value in measured[route].items():
            if not math.isclose(value,report["component_result"]["measurements"][route][key],abs_tol=1e-12):
                raise ValueError("aggregate score differs from raw outputs")
    gain=measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
    gates=dict(foreground_f1=measured["aligned"]["foreground_f1"]>=.90,conditioning_gain=gain>=.20)
    if gates!=report["component_result"]["gates"] or all(gates.values())!=report["component_result"]["writing_component_pass"]:
        raise ValueError("registered component decision differs")
    if report["peak_allocated_vram_gib"]<max(row["peak_allocated_vram_gib"] for row in metrics):
        raise ValueError("final report drops earlier training memory peak")
    for name in ("report.json","writer_final.pt","development/report.json","development/cases.json",
            "development/manifest.json","metrics.jsonl","sample_order.jsonl","resume_verification.json"):
        hashes[name]=sha(run/name)
    result=dict(files_sha256=hashes,original_development_references_verified=128,
        raw_generated_image_scores_verified=256,training_sampling_draws_replayed=160000,
        distinct_training_examples_consumed=len(selected),available_training_examples=len(training),
        measurements=measured,conditioning_gain=gain,gates=gates,writing_component_pass=all(gates.values()),
        language_quality_claim=False,general_ilm_goal_complete=False,development_reused=True,
        verifier_sha256=sha(Path(__file__)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!="files_sha256"},indent=2))


if __name__=="__main__":main()

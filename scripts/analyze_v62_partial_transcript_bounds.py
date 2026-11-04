#!/usr/bin/env python3
"""Post-hoc full-case bounds without filling or dropping invalid transcripts."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps
import torch

from ilm.visual_lm.blind_vision_transcription import atomic_json,sha256
from ilm.visual_lm.visual_qa_data import read_questions
from scripts.audit_native_pixel_transcripts_v61 import scores


def png_bytes(cells):
    raster=np.concatenate([cell[0].numpy() for cell in cells],axis=1)
    image=Image.fromarray((255*(1-raster)).astype(np.uint8))
    image=ImageOps.expand(image.resize((image.width*4,image.height*4),Image.Resampling.NEAREST),border=16,fill=255)
    buffer=io.BytesIO();image.save(buffer,format="PNG");return buffer.getvalue()


def envelope(lower,upper,articles):
    lower,upper,articles=np.asarray(lower,float),np.asarray(upper,float),np.asarray(articles)
    groups=[np.flatnonzero(articles==a) for a in sorted(set(articles))]
    lo=np.array([lower[g].sum() for g in groups]);hi=np.array([upper[g].sum() for g in groups])
    counts=np.array([len(g) for g in groups])
    draws=np.random.default_rng(20266379).integers(0,len(groups),(2000,len(groups)))
    return dict(identification_interval=[float(lower.mean()),float(upper.mean())],
        article_bootstrap_sensitivity_envelope=[float(np.quantile(lo[draws].sum(1)/counts[draws].sum(1),.025)),
            float(np.quantile(hi[draws].sum(1)/counts[draws].sum(1),.975))],
        articles=len(groups),examples=len(lower),replicates=2000,seed=20266379)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/correlated_native_density_v62_20261105")
    run=Path(parser.parse_args().run);parent=run/"vision_transcript_diagnostic";out=parent/"partial_bounds"
    if (out/"report.json").exists():raise FileExistsError("completed sensitivity analysis is immutable")
    original=json.loads((parent/"report.json").read_text());cleanup=json.loads((parent/"cleanup.json").read_text())
    if original["decision"]!="invalid_actor_transcription" or not original["calibration"]["passed"]:
        raise ValueError("requires the completed original invalid-transcript diagnostic")
    if cleanup["remaining_owned_live_pids"] or cleanup["port_open_after_cleanup"]:
        raise ValueError("owned evaluator must clean up before CPU sensitivity analysis")
    primary=json.loads((run/"development/report.json").read_text())
    source=Path("data/visual_qa/cmrc2018_short/development.jsonl")
    if sha256(source)!=primary["data_sha256"]["development.jsonl"]:raise ValueError("development source changed")
    records={r["id"]:r for r in read_questions(source)}
    prediction_hashes=json.loads((parent/"actor_prediction_hashes.json").read_text())
    metrics=("text_exact","bounded_character_accuracy");routes=("k1","k1_blank","k8","k8_blank")
    outputs={};response_hashes={}
    for route in routes:
        rows=[]
        for identifier in primary["primary_ids"]:
            path=run/"development/predictions"/route/(identifier+".pt")
            if sha256(path)!=prediction_hashes[str(path.relative_to(run))]:raise ValueError("prediction pixels changed")
            payload=torch.load(path,map_location="cpu",weights_only=False);cells=payload["images"]
            if payload["id"]!=identifier:raise ValueError("prediction ID changed")
            valid=True;key=None;text=""
            if len(cells):
                data=png_bytes(cells);key=hashlib.sha256(data).hexdigest()
                if data!=(parent/"anonymous_actor_images"/(key+".png")).read_bytes():raise ValueError("anonymous image differs")
                response_path=parent/"recognizer_outputs"/(key+".json");response=json.loads(response_path.read_text())
                if response["image_sha256"]!=key or response["request_sha256"]!=original["calibration"]["recognition_request_sha256"]:
                    raise ValueError("recognition identity changed")
                response_hashes[key]=sha256(response_path);valid=response["valid_json"];text=response["text"]
            measured=scores(text,[r["text"] for r in records[identifier]["references"]]) if valid else None
            intervals={m:[measured[m],measured[m]] if valid else [0.,1.] for m in metrics}
            rows.append(dict(id=identifier,article=records[identifier]["article_title"],image_sha256=key,
                valid_transcript=valid,intervals=intervals))
        outputs[route]=rows
    measured={route:dict(valid_examples=sum(r["valid_transcript"] for r in rows),
        unknown_examples=sum(not r["valid_transcript"] for r in rows),
        metrics={m:envelope([r["intervals"][m][0] for r in rows],[r["intervals"][m][1] for r in rows],
            [r["article"] for r in rows]) for m in metrics}) for route,rows in outputs.items()}
    comparisons={}
    for left,right in (("k8","k1"),("k8","k8_blank"),("k1","k1_blank")):
        for m in metrics:
            a,b=outputs[left],outputs[right]
            comparisons[f"{left}_minus_{right}_{m}"]=envelope(
                [x["intervals"][m][0]-y["intervals"][m][1] for x,y in zip(a,b)],
                [x["intervals"][m][1]-y["intervals"][m][0] for x,y in zip(a,b)],
                [r["article"] for r in a])
    identity=dict(post_hoc=True,model_calls=0,primary_and_original_decisions_unchanged=True,
        primary_sha256=sha256(run/"development/report.json"),original_diagnostic_sha256=sha256(parent/"report.json"),
        evaluator_cleanup_sha256=sha256(parent/"cleanup.json"),development_sha256=sha256(source),
        source_sha256={p:sha256(p) for p in [__file__,"references/v62_partial_transcript_bounds_plan.md",
            "scripts/audit_native_pixel_transcripts_v61.py"]},response_sha256=response_hashes)
    for route,rows in outputs.items():atomic_json(rows,out/(route+"_case_intervals.json"))
    atomic_json(dict(**identity,calibration=original["calibration"],measurements=measured,comparisons=comparisons,
        qualification="Sensitivity bounds for missing transcripts; reader errors and seed/development selection are not covered"),out/"report.json")
    print(json.dumps(dict(measurements=measured,comparisons=comparisons),indent=2))


if __name__=="__main__":main()

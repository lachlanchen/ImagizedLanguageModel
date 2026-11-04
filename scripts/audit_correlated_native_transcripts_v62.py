#!/usr/bin/env python3
"""Calibrate a blinded local vision reader before auxiliary V62 transcription."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image,ImageOps
import torch

from ilm.visual_lm.blind_vision_transcription import (
    MODEL,MANIFEST_SHA,MODEL_SHA,OPTIONS,PROMPT,atomic_json,owned_reader,sha256,
)
from ilm.visual_lm.visual_qa_data import read_questions
from scripts.audit_native_pixel_transcripts_v61 import scores,normalize


def bootstrap(values,articles):
    values,articles=np.asarray(values,float),np.asarray(articles)
    groups=[np.flatnonzero(articles==a) for a in sorted(set(articles))]
    totals=np.asarray([values[g].sum() for g in groups]);counts=np.asarray([len(g) for g in groups])
    draws=np.random.default_rng(20266379).integers(0,len(groups),(2000,len(groups)))
    means=totals[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                articles=len(groups),examples=len(values),replicates=2000,seed=20266379)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/correlated_native_density_v62_20261105")
    parser.add_argument("--calibration",default="artifacts/v62_transcript_calibration_20261105")
    parser.add_argument("--local-root",default="../LocalLLM");parser.add_argument("--port",type=int,default=11437)
    args=parser.parse_args();run,cal=Path(args.run),Path(args.calibration)
    # This check precedes service startup and every evaluator model call.
    status=json.loads((run/"pipeline_status.json").read_text())
    if status["status"]!="audit_complete":raise RuntimeError("primary V62 pipeline is still active; do not start a second model")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("physical GPU 0 required")
    out=run/"vision_transcript_diagnostic"
    if (out/"report.json").exists():raise FileExistsError("completed vision diagnostic is immutable")
    primary=json.loads((run/"development/report.json").read_text())
    preparation=json.loads((cal/"report.json").read_text())
    protocol="references/v62_blind_vision_transcription_plan.md"
    if preparation["protocol_sha256"]!=sha256(protocol) or preparation["model_calls"]!=0:
        raise ValueError("calibration preparation or protocol changed")
    if preparation["manifest_sha256"]!=sha256(cal/"private_expected.jsonl"):
        raise ValueError("calibration labels changed")
    calibration=[json.loads(s) for s in (cal/"private_expected.jsonl").read_text().splitlines()]
    if len(calibration)!=160 or any(sha256(cal/r["image"])!=r["image_sha256"] for r in calibration):
        raise ValueError("calibration images differ")
    identity=dict(primary_audit_sha256=sha256(run/"development/report.json"),
        calibration_report_sha256=sha256(cal/"report.json"),model=MODEL,
        evaluator_manifest_sha256=MANIFEST_SHA,evaluator_weights_sha256=MODEL_SHA,
        options=OPTIONS,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        source_sha256={name:sha256(name) for name in [__file__,protocol,
            "ilm/visual_lm/blind_vision_transcription.py","scripts/audit_native_pixel_transcripts_v61.py"]},
        question_visible=False,reference_text_visible=False,actor_identity_visible=False,
        external_evaluator_not_actor=True,primary_decision_unchanged=True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"identity.json").exists() and json.loads((out/"identity.json").read_text())!=identity:
        raise ValueError("partial vision diagnostic identity changed")
    atomic_json(identity,out/"identity.json")
    torch.set_num_threads(2)
    begun=time.perf_counter()

    def stop(_signal,_frame):raise KeyboardInterrupt("stop requested; owned evaluator cleanup follows")

    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    with owned_reader(run,out,args.local_root,args.port) as reader:
        unique={row["image_sha256"]:cal/row["image"] for row in calibration}
        recognized={}
        for index,(key,path) in enumerate(sorted(unique.items())):
            if time.perf_counter()-begun>4*3600:raise RuntimeError("vision audit time ceiling")
            recognized[key]=reader.recognize(path)
            if index%16==0:print(json.dumps(dict(stage="calibration",completed=index+1,total=len(unique))),flush=True)
        rows=[]
        for row in calibration:
            value=recognized[row["image_sha256"]]
            exact=(value["valid_json"] and normalize(value["text"])==normalize(row["expected_text"]))
            rows.append(dict(kind=row["kind"],source_id=row["source_id"],image_sha256=row["image_sha256"],
                valid_json=value["valid_json"],text=value["text"],expected_text=row["expected_text"],exact=bool(exact)))
        measured={kind:dict(examples=sum(r["kind"]==kind for r in rows),
            exact_fraction=float(np.mean([r["exact"] for r in rows if r["kind"]==kind])))
            for kind in ("reference","blank","random_ink")}
        gates=dict(reference_transcription=measured["reference"]["exact_fraction"]>=.95,
            blanks_empty=measured["blank"]["exact_fraction"]==1,
            random_ink_empty=measured["random_ink"]["exact_fraction"]>=.9,
            valid_json=all(row["valid_json"] for row in rows))
        atomic_json(rows,out/"calibration_transcripts.json")
        calibration_result=dict(measurements=measured,gates=gates,passed=all(gates.values()),
            recognition_request_sha256=reader.request_sha256,source_image_count=len(unique),
            frozen_before_actor_recognition=True)
        atomic_json(calibration_result,out/"calibration_report.json")
        print(json.dumps(dict(calibration=calibration_result)),flush=True)
        if not all(gates.values()):
            atomic_json(dict(**identity,calibration=calibration_result,decision="invalid_vision_instrument",
                actor_images_transcribed=0,elapsed_seconds=time.perf_counter()-begun,
                qualification="No auxiliary language claim: reader failed registered calibration"),out/"report.json")
            return
        # The instrument is now frozen. Only after this point are actor images
        # assembled for recognition; question/reference data remain outside API requests.
        folder=out/"anonymous_actor_images";folder.mkdir(exist_ok=True)
        keys={};images={};prediction_hashes={}
        routes=("k1","k1_blank","k8","k8_blank")
        for route in routes:
            for identifier in primary["primary_ids"]:
                source=run/"development/predictions"/route/(identifier+".pt")
                prediction_hashes[str(source.relative_to(run))]=sha256(source)
                payload=torch.load(source,map_location="cpu",weights_only=False)
                if payload["id"]!=identifier:raise ValueError("actor prediction ID differs")
                pixels=payload["images"]
                if not len(pixels):keys[route,identifier]=None;continue
                if not bool(((pixels==0)|(pixels==1)).all()):raise ValueError("actor image is not the recorded binary raster")
                raster=np.concatenate([cell[0].numpy() for cell in pixels],axis=1)
                im=Image.fromarray((255*(1-raster)).astype(np.uint8))
                im=ImageOps.expand(im.resize((im.width*4,im.height*4),Image.Resampling.NEAREST),border=16,fill=255)
                buffer=io.BytesIO();im.save(buffer,format="PNG");data=buffer.getvalue()
                key=hashlib.sha256(data).hexdigest();path=folder/(key+".png")
                path.write_bytes(data);keys[route,identifier]=key;images[key]=path
        atomic_json(prediction_hashes,out/"actor_prediction_hashes.json")
        for index,(key,path) in enumerate(sorted(images.items())):
            if time.perf_counter()-begun>4*3600:raise RuntimeError("vision audit time ceiling")
            recognized[key]=reader.recognize(path)
            if index%16==0:print(json.dumps(dict(stage="actor_images",completed=index+1,total=len(images))),flush=True)
        if any(not recognized[key]["valid_json"] for key in images):
            atomic_json(dict(**identity,calibration=calibration_result,decision="invalid_actor_transcription",
                actor_images_transcribed=len(images),qualification="Invalid JSON prevents a complete fixed-case score"),out/"report.json")
            return
        records={r["id"]:r for r in read_questions("data/visual_qa/cmrc2018_short/development.jsonl")}
        outputs={}
        for route in routes:
            scored=[]
            for identifier in primary["primary_ids"]:
                record=records[identifier];key=keys[route,identifier]
                text=recognized[key]["text"] if key is not None else ""
                scored.append(dict(id=identifier,article=record["article_title"],image_sha256=key,text=text,
                    **scores(text,[r["text"] for r in record["references"]])))
            outputs[route]=scored;atomic_json(scored,out/(route+"_transcripts.json"))
        measurements={route:{metric:float(np.mean([r[metric] for r in rows])) for metric in
            ("text_exact","normalized_character_error","bounded_character_accuracy")} for route,rows in outputs.items()}
        comparisons={}
        for left,right in (("k8","k1"),("k8","k8_blank"),("k1","k1_blank")):
            for metric in ("text_exact","bounded_character_accuracy"):
                comparisons[f"{left}_minus_{right}_{metric}"]=bootstrap(
                    [a[metric]-b[metric] for a,b in zip(outputs[left],outputs[right])],
                    [r["article"] for r in outputs[left]])
        atomic_json(dict(**identity,calibration=calibration_result,measurements=measurements,
            comparisons=comparisons,actor_images_transcribed=len(images),primary_ids=primary["primary_ids"],
            elapsed_seconds=time.perf_counter()-begun,decision="auxiliary_transcription_complete",
            qualification="Calibrated answer-only transcription; no new primary model qualification"),out/"report.json")
        print(json.dumps(dict(measurements=measurements,comparisons=comparisons),indent=2),flush=True)


if __name__=="__main__":main()

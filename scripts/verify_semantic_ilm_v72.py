#!/usr/bin/env python3
"""CPU-only independent scoring of all raw V72 autonomous answer images."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_language_audit import AUDIT_DATA_SHA, CodecLanguageAuditData
from scripts.cache_semantic_images_v72 import ROOT, atomic_json, sha
from scripts.verify_codec_language_v68 import assert_close, independent_actor_prompt, physical_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("independent score verification must be CPU-only")
    torch.set_num_threads(4)
    begun = time.perf_counter()
    out = args.run/"development"
    report = json.loads((out/"report.json").read_text())
    identity = json.loads((out/"identity.json").read_text())
    if (identity!=report["identity"] or identity["prepared_data_sha256"]!=AUDIT_DATA_SHA
            or identity["maximum_blocks"]!=128 or identity["flow_steps"]!=32):
        raise ValueError("fixed full-answer audit identity differs")
    for name,digest in identity["sources"].items():
        original = subprocess.check_output(["git","show",identity["source_commit"]+":"+name],cwd=ROOT)
        if sha(ROOT/name)!=digest or hashlib.sha256(original).hexdigest()!=digest:
            raise ValueError("audit source provenance differs")
    training_file = args.run/"training/training_report.json"
    if sha(training_file)!=identity["training_report_sha256"]:
        raise ValueError("training report changed")
    training = json.loads(training_file.read_text())
    if (training["updates"]!=40000 or training["draws"]!=640000
            or training["checkpoint_sha256"]!=identity["checkpoint_sha256"]
            or sha(args.run/"training/checkpoint_latest.pt")!=identity["checkpoint_sha256"]):
        raise ValueError("production checkpoint identity differs")
    for name,digest in training["identity"]["sources"].items():
        raw = subprocess.check_output(["git","show",training["identity"]["source_commit"]+":"+name],cwd=ROOT)
        if sha(ROOT/name)!=digest or hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError("registered optimizer implementation differs")
    cache = args.run/"cache"
    cache_report = json.loads((cache/"report.json").read_text())
    if (sha(cache/"report.json")!=training["identity"]["cache_report_sha256"]
            or not cache_report["complete"] or cache_report["completed"]!=12976
            or sha(cache/"recipe.json")!=training["identity"]["recipe_sha256"]):
        raise ValueError("full admitted training cache differs")
    for name,digest in cache_report["files_sha256"].items():
        if sha(cache/name)!=digest:
            raise ValueError("completed training cache payload changed")
    if sha(out/"scores.json")!=report["scores_sha256"]:
        raise ValueError("retained score file differs")
    recorded_scores = json.loads((out/"scores.json").read_text())
    data = CodecLanguageAuditData(ROOT/"data/general_instructions/codec_language_audit_v68_20261105")
    scores = {}
    files = {}
    generated = 0
    for index,row in enumerate(data.cases):
        key = row["source"]+"/"+row["id"]
        scores[key] = {}
        for route in ("normal","blank"):
            folder = out/"predictions"/f"{index:03}_{route}"
            retained = json.loads((folder/"header.json").read_text())
            prompt = independent_actor_prompt(data.payload(index),blank=route=="blank")
            header = dict(identity_sha256=sha(out/"identity.json"),position=index,route=route,
                source=row["source"],id=row["id"],seed=identity["seed"]+index,
                prompt_sha256=hashlib.sha256(prompt.numpy().tobytes()).hexdigest())
            if {k:v for k,v in retained.items() if k!="output"}!=header:
                raise ValueError("independently reconstructed actor input differs")
            result = torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
            images = result["images"]
            if (images.dtype!=torch.float32 or images.ndim!=4 or images.shape[1:]!=(1,32,64)
                    or len(images)>128 or not bool(((images==-1)|(images==1)).all())):
                raise ValueError("unmodified generated physical image blocks required")
            stopped,capped = result["stopped"],result["capped"]
            probabilities = result["stop_probabilities"]
            if (type(stopped) is not bool or type(capped) is not bool or stopped==capped
                    or (capped and len(images)!=128) or len(probabilities)!=len(images)+1
                    or not all(np.isfinite(x) and 0<=x<=1 for x in probabilities)
                    or any(x>=.5 for x in probabilities[:-1]) or (probabilities[-1]>=.5)!=stopped):
                raise ValueError("learned structural stop events differ from registered policy")
            if (not result["feedback_from_emitted_pixels"] or not result["image_input_image_output"]
                    or len(result["feedback_sha256"])!=len(images)
                    or any(len(digest)!=64 for digest in result["feedback_sha256"])):
                raise ValueError("physical feedback trace is incomplete")
            raster = np.concatenate([image[0].numpy() for image in images],axis=1) if len(images) else np.empty((32,0),np.float32)
            prediction = raster<0
            png = folder/"answer.png"
            if len(images):
                expected = ((raster+1)*127.5).astype(np.uint8)
                if not np.array_equal(np.array(Image.open(png)),expected):
                    raise ValueError("answer PNG differs from raw emitted pixels")
            elif png.exists():
                raise ValueError("empty answer has a stale placeholder image")
            output = dict(raw_sha256=sha(folder/"answer.pt"),blocks=len(images),stopped=stopped,capped=capped,
                answer_png_sha256=sha(png) if len(images) else None,feedback_from_emitted_pixels=True)
            if output!=retained["output"]:
                raise ValueError("saved output receipt differs")
            scores[key][route] = physical_scores(prediction,data.references(index),stopped)
            for path in sorted(folder.iterdir()):
                files[str(path.relative_to(out))] = sha(path)
            generated += 1
    assert_close(scores,recorded_scores)
    if files!=report["files_sha256"] or generated!=384 or report["generations"]!=384:
        raise ValueError("raw generation inventory is incomplete")
    measurements = {}
    for source in ("cmrc","dolly"):
        selected = [scores[r["source"]+"/"+r["id"]] for r in data.cases if r["source"]==source]
        measurements[source] = {route:{metric:float(np.mean([r[route][metric] for r in selected]))
            for metric in ("foreground_f1","exact","blank","blocks","stopped")} for route in ("normal","blank")}
    chinese = [r for r in data.cases if r["source"]=="cmrc"]
    pairs = [dict(ids=pair,both_correct=all(scores["cmrc/"+identifier]["normal"]["exact"] for identifier in pair))
        for pair in data.report["chosen_pairs"]]
    paired = float(np.mean([r["both_correct"] for r in pairs]))
    articles = sorted({r["article"] for r in chinese})
    effects = np.array([np.mean([scores["cmrc/"+r["id"]]["normal"]["foreground_f1"]-
        scores["cmrc/"+r["id"]]["blank"]["foreground_f1"] for r in chinese if r["article"]==article]) for article in articles])
    bootstrap = np.random.default_rng(20266831).choice(effects,size=(2000,64),replace=True).mean(1)
    effect = dict(mean=float(effects.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        articles=64,replicates=2000,seed=20266831)
    normal = measurements["cmrc"]["normal"]
    gates = dict(chinese_exact=normal["exact"]>=.5,chinese_foreground_f1=normal["foreground_f1"]>=.8,
        both_correct_pairs=paired>=.35,question_gain=effect["mean"]>=.1,
        question_gain_ci_positive=effect["ci95"][0]>0,learned_stopping=normal["stopped"]>=.9)
    categories = {}
    for category in sorted({r["category"] for r in data.cases if r["source"]=="dolly"}):
        rows = [scores["dolly/"+r["id"]]["normal"] for r in data.cases if r["source"]=="dolly" and r["category"]==category]
        categories[category] = {metric:float(np.mean([r[metric] for r in rows])) for metric in ("foreground_f1","exact","blank","stopped")}
    summary = dict(measurements=measurements,paired_both_correct=paired,pairs=pairs,question_effect=effect,
        gates=gates,bounded_chinese_development_pass=all(gates.values()),english_categories=categories)
    for key,value in summary.items():
        assert_close(value,report[key])
    atomic_json(args.run/"verification.json",dict(report_sha256=sha(out/"report.json"),
        verifier_sha256=sha(Path(__file__)),all_384_outputs_independently_scored=True,
        full_training_cache_payloads_verified=12976,source_bound_inputs_verified=384,
        actual_generation_forward_replayed=False,physical_feedback_hashes_numerically_replayed=False,
        preflight_exact_feedback_check=True,offline_checks_pass=True,**summary,
        general_language_qualified=False,general_ilm_goal_complete=False,
        elapsed_seconds=time.perf_counter()-begun))


if __name__=="__main__":
    main()

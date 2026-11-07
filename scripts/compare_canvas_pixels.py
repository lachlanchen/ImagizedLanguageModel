#!/usr/bin/env python3
"""Rescore preserved predecessor and canvas images with the same pixel rule."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.cache_semantic_images_v72 import ROOT, sha, committed_identity, atomic_json
from scripts.run_parallel_canvas import scores


def main():
    identity=committed_identity(("scripts/compare_canvas_pixels.py","scripts/run_parallel_canvas.py"))
    prior=ROOT/"references/evidence/decoder_metric_v82_result";old=json.loads((prior/"result.json").read_text())
    current=ROOT/"references/evidence/parallel_canvas_result";new=json.loads((current/"result.json").read_text())
    cohort=[r for r in new["raw_cases"] if r["group"]=="training" and r["stage"]=="final"]
    if len(cohort)!=32: raise ValueError("all original training questions required")
    rows=[]
    for arm in ("control","metric"):
        headers=old["source_prediction_headers"][arm]["after"]
        if len(headers)!=32:raise ValueError("complete earlier cohort required")
        for row,header in zip(cohort,headers):
            i=row["index"];rel=f"raw/{arm}_after_{i:02}_answer.png";ref=f"raw/{arm}_after_{i:02}_reference.png"
            for name in (rel,ref):
                if sha(prior/name)!=old["raw_png_sha256"][name]:raise ValueError("earlier PNG changed")
            if sha(current/row["output"])!=new["files_sha256"][row["output"]]:raise ValueError("canvas PNG changed")
            target=np.asarray(Image.open(current/row["reference_images"][0]).convert("L"))
            if not np.array_equal(target,np.asarray(Image.open(prior/ref).convert("L"))):raise ValueError("physical reference differs")
            source=new["training_identity"] # Provenance is also checked through each ordered source header below.
            expected=json.loads((ROOT/"references/evidence/clean_learning_v74_result.json").read_text())["cases"][i]["source"]
            if header["position"]!=i or header["source"]!=expected:raise ValueError("question identity/order differs")
            image=np.asarray(Image.open(prior/rel).convert("L"))
            rows.append(dict(arm=arm,index=i,source=expected,output=str((prior/rel).relative_to(ROOT)),
                output_sha256=sha(prior/rel),**scores(image,[target])))
    groups={arm:dict(cases=32,exact=sum(r["exact"] for r in rows if r["arm"]==arm),
        foreground_f1=float(np.mean([r["foreground_f1"] for r in rows if r["arm"]==arm]))) for arm in ("control","metric")}
    result=dict(producer=identity,prior_result_sha256=sha(prior/"result.json"),canvas_result_sha256=sha(current/"result.json"),
        prior_under_common_origin_white_padding=groups,canvas_under_same_rule=new["groups"]["training/final"],cases=rows,
        all32_reference_pixels_identical=True,all_original_pngs_unchanged=True,
        no_model_calls_or_optimization=True,matched_training_intervention=False,
        changed_architecture_initialization_training_history_and_termination=True,
        causal_or_efficiency_advantage_established=False,general_ilm_goal_complete=False)
    atomic_json(ROOT/"references/evidence/parallel_canvas_prior_pixels.json",result)
    print(json.dumps(dict(prior=groups,canvas=new["groups"]["training/final"]),indent=2))


if __name__=="__main__": main()

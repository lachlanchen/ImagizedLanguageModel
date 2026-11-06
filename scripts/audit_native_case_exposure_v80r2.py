#!/usr/bin/env python3
"""Replay declared sample order and count actual exposure of saved diagnostic cases."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
from scripts.cache_semantic_images_v72 import ROOT, atomic_json, committed_identity, sha
from scripts.train_native_chinese_v80 import ChineseImages, SEED

SOURCES=("scripts/audit_native_case_exposure_v80r2.py", "scripts/audit_native_case_exposure_v80.py", "scripts/train_native_chinese_v80.py",
    "ilm/visual_lm/semantic_writing_data.py", "references/native_chinese_v80_protocol.md")


def run(run_dir):
    identity=json.loads((run_dir/"identity.json").read_text())
    if identity["purpose"]!="native_chinese_full_context" or identity["seed"]!=SEED:
        raise ValueError("the declared native Chinese run is required")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("original training implementation changed")
    data=ChineseImages()
    snapshots={}
    for folder in sorted((run_dir/"diagnostics").glob("update_*")):
        if not (folder/"pixel_scores.json").exists():
            continue
        report=json.loads((folder/"pixel_scores.json").read_text())
        if len(report["rows"])!=4 or not report["training_examples_only"]:
            raise ValueError("complete unselected training diagnostic required")
        snapshots[report["update"]]=(folder,report)
    if not snapshots:
        raise ValueError("no completed saved diagnostic exists")
    maximum=max(snapshots)
    # The live logger may be appending its next line. Only consume completed
    # newline-terminated records, and require all scheduled rows below maximum.
    lines=(run_dir/"training/metrics.jsonl").read_bytes().split(b"\n")[:-1]
    recorded={row["update"]:row for row in map(json.loads,lines) if row["update"]<=maximum}
    required={1,*range(10,maximum+1,10)}
    if set(recorded)!=required:
        raise ValueError("actual logged minibatch inventory differs")
    source=ExactEpochSampler(len(data),seed=SEED+1)
    exposures=Counter();prefilled=set();results=[];step=0;matched=0
    for pool_number in range((maximum+127)//128):
        owning=source.take(32);prefilled.update(owning)
        local=ExactEpochSampler(32,seed=SEED+100+pool_number)
        for pool_step in range(1,129):
            if step==maximum:
                break
            indices=local.take(4);step+=1
            exposures.update(owning[i] for i in indices)
            if step in recorded:
                row=recorded[step]
                if (row["indices"]!=indices or row["pool"]!=pool_number or row["draws"]!=step*4
                        or row["unique_training_examples"]!=len(prefilled)):
                    raise ValueError("declared sampler replay differs from the actual minibatch log")
                matched+=1
            if step in snapshots:
                folder,report=snapshots[step]
                expected=[data.rows[i][0] for i in owning[:4]]
                if [row["source"] for row in report["rows"]]!=expected:
                    raise ValueError("saved diagnostic selected different owning cases")
                results.append(dict(update=step,pool_number=pool_number,updates_in_current_pool=pool_step,
                    optimization_draws=step*4,unique_optimized_records=len(exposures),
                    unique_prefilled_records=len(prefilled),
                    diagnostic_report_sha256=sha(folder/"pixel_scores.json"),
                    cases=[dict(source=data.rows[i][0],total_optimization_exposures=exposures[i]) for i in owning[:4]],
                    exact=report["exact"],foreground_f1=report["mean_foreground_f1"],
                    cohorts_differ_between_snapshots=True,causal_learning_trend_claim=False))
    receipt=dict(**committed_identity(SOURCES),training_identity_sha256=sha(run_dir/"identity.json"),
        maximum_replayed_update=maximum,logged_minibatches_checked=matched,
        every_logged_minibatch_matches_declared_sampler=True,
        uses_original_sampler_implementation=True,independent_sampler_implementation=False,
        optimizer_or_model_replayed=False,model_updated=False,cuda_initialized=torch.cuda.is_initialized(),
        results=results,language_quality_claim=False,general_ilm_goal_complete=False)
    output=run_dir/"case_exposure";output.mkdir(exist_ok=True)
    destination=output/f"through_{maximum:05}_r2.json"
    if destination.exists():
        previous=json.loads(destination.read_text())
        if {k:v for k,v in previous.items() if k!="source_commit"}!={k:v for k,v in receipt.items() if k!="source_commit"}:
            raise ValueError("preserve the earlier exposure audit rather than overwrite a different receipt")
        # Unrelated documentation commits do not change these authenticated
        # inputs or computations. Preserve the original producing revision.
        receipt=previous
    atomic_json(destination,receipt)
    print(json.dumps(receipt,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    run(parser.parse_args().run.resolve())

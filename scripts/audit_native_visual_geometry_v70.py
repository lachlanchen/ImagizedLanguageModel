#!/usr/bin/env python3
"""Post-hoc native resize-budget reconstruction from V70 logs and token counts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("completed geometry audits are immutable")
    root=Path(__file__).resolve().parents[1]
    prep=json.loads((args.run/"preparation.json").read_text())
    query=json.loads((args.run/"query_report.json").read_text())
    verified=json.loads((args.run/"verification.json").read_text())
    if not verified["offline_checks_pass"] or sha(args.run/"query_report.json")!=verified["query_report_sha256"]:
        raise ValueError("verified native-source requests required")
    log=args.run/"native_runtime/owned_server.log";text=log.read_text()
    if "--image-min-tokens 1024" not in text or not re.search(r"image_min_pixels:\s+1048576",text):
        raise ValueError("observed native minimum differs")
    if not re.search(r"image_max_pixels:\s+4194304",text):
        raise ValueError("observed native maximum differs")
    records=[]
    for row in prep["cases"]:
        h,w=row["images"]["normal"]["shape"]
        # The pages already have dimensions divisible by 32. Reproduce the
        # native float32 minimum/maximum-area scaling and factor rounding.
        if h*w<1048576:
            scale=np.sqrt(np.float32(1048576)/(np.float32(h)*np.float32(w)))
            nh=int(np.ceil(np.float32(h)*scale/np.float32(32)))*32
            nw=int(np.ceil(np.float32(w)*scale/np.float32(32)))*32
        elif h*w>4194304:
            scale=np.sqrt(np.float32(h*w)/np.float32(4194304))
            nh=max(32,int(np.floor(np.float32(h)/scale/np.float32(32)))*32)
            nw=max(32,int(np.floor(np.float32(w)/scale/np.float32(32)))*32)
        else:
            nh,nw=h,w
        relative=f"responses/case_{row['position']:02}_image_only.json"
        if sha(args.run/relative)!=query["responses_sha256"][relative]:
            raise ValueError("observed native response changed")
        observed=json.loads((args.run/relative).read_text())["response"]["prompt_eval_count"]
        native_positions=(nh//32)*(nw//32);raw_positions=(h//32)*(w//32)
        if observed!=native_positions+10:
            raise ValueError("resize budget plus native chat overhead does not match observed count")
        records.append(dict(position=row["position"],source=row["source"],original_shape=[h,w],
            inferred_native_resized_shape=[nh,nw],custom_actor_visual_positions=raw_positions,
            reconstructed_native_visual_positions=native_positions,native_prompt_eval_count=observed,
            native_chat_overhead=10,visual_position_ratio=native_positions/raw_positions))
    tree=root.parent/"LocalLLM/.local/tools/llama.cpp"
    sources=[tree/"tools/mtmd/clip.cpp",tree/"tools/mtmd/mtmd-image.cpp"]
    report=dict(post_hoc=True,model_calls=0,optimizer_updates=0,source_verification_sha256=sha(args.run/"verification.json"),
        runtime_log_sha256=sha(log),executed_code_sha256=sha(Path(__file__)),
        inspected_source_tree_commit=subprocess.check_output(["git","-C",str(tree),"rev-parse","HEAD"],text=True).strip(),
        inspected_source_files_sha256={str(p.relative_to(tree)):sha(p) for p in sources},
        native_min_visual_positions=1024,native_min_pixels=1048576,native_max_pixels=4194304,
        records=records,all_24_native_prompt_counts_explained=True,
        exact_native_resized_pixels_verified=False,native_feature_equivalence_verified=False,
        causal_effect_of_resize_tested=False,language_quality_claim=False)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(cases=len(records),all_counts_explained=True,
        custom_positions_range=[min(r["custom_actor_visual_positions"] for r in records),max(r["custom_actor_visual_positions"] for r in records)],
        native_positions_range=[min(r["reconstructed_native_visual_positions"] for r in records),max(r["reconstructed_native_visual_positions"] for r in records)],
        first_case=records[0]),indent=2))


if __name__=="__main__":
    main()

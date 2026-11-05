#!/usr/bin/env python3
"""Source/pixel and temporary-projection verification for the V71 import check."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts.check_native_visual_transfer_v71 import (PREVIOUS,PREVIOUS_SHA,ROOT,
    archive_and_tokenizer,atomic_json,compile_controls,cpu_only,identity,sha)
from scripts.verify_native_visual_baseline_v70 import distance


def verify(out):
    cpu_only();begun=time.perf_counter();_,sources=identity()
    prep=json.loads((out/"preparation.json").read_text());report=json.loads((out/"generation_report.json").read_text())
    if prep["sources"]!=sources or report["sources"]!=sources or report["preparation_sha256"]!=sha(out/"preparation.json"):
        raise ValueError("fixed source implementation changed")
    if sha(PREVIOUS/"verification.json")!=PREVIOUS_SHA or prep["previous_preparation_sha256"]!=sha(PREVIOUS/"preparation.json"):
        raise ValueError("original source diagnostic changed")
    old=json.loads((PREVIOUS/"preparation.json").read_text())
    geometry=json.loads((PREVIOUS/"geometry_audit.json").read_text())
    if prep["geometry_sha256"]!=sha(PREVIOUS/"geometry_audit.json"):
        raise ValueError("source geometry changed")
    archive,tokenizer,markers=archive_and_tokenizer();controls,compiler=compile_controls(archive,tokenizer,markers)
    saved=torch.load(out/"structural_vectors.pt",map_location="cpu",weights_only=True)
    if (set(saved)!={"before","after"} or sha(out/"structural_vectors.pt")!=prep["structural_vectors_sha256"]
            or compiler!=json.loads((out/"compiler.json").read_text())):
        raise ValueError("structural numeric interface identity changed")
    if any(v.dtype!=torch.float32 or not torch.equal(v,controls[k]) for k,v in saved.items()):
        raise ValueError("source control vectors differ")
    eog=set(compiler["eog_ids"]);images=0;items=[];state_batches=[];selected=[];recorded_logits=[]
    for row,source_row in zip(prep["cases"],old["cases"][:8]):
        if any(row[k]!=source_row[k] for k in ("position","source","id","article","source_prompt","references")):
            raise ValueError("fixed case selection changed")
        for route,entry in row["images"].items():
            scale,condition=route.split("/")
            original=source_row["images"][condition];source=PREVIOUS/original["file"]
            if sha(source)!=entry["source_png_sha256"] or sha(out/entry["file"])!=entry["sha256"]:
                raise ValueError("source or transformed image changed")
            image=Image.open(source).convert("RGB")
            if scale=="native_budget":
                h,w=geometry["records"][row["position"]]["inferred_native_resized_shape"]
                image=image.resize((w,h),Image.Resampling.BILINEAR)
            if not np.array_equal(np.asarray(image),np.asarray(Image.open(out/entry["file"]).convert("RGB"))):
                raise ValueError("declared bilinear image transform differs")
            images+=1
        other=next(r for r in prep["cases"] if r["article"]==row["article"] and r["id"]!=row["id"])
        for route in ("raw/normal","raw/blank","native_budget/normal","native_budget/blank","text"):
            name=f"case_{row['position']:02}_"+route.replace("/","_")+".pt"
            if sha(out/name)!=report["raw_files_sha256"][name]:
                raise ValueError("raw diagnostic output changed")
            value=torch.load(out/name,map_location="cpu",weights_only=True)
            result=value["result"];ids=result["selected_ids"];states=result["hidden_states"]
            if (value["position"]!=row["position"] or value["route"]!=route or not 1<=len(ids)<=64
                    or states.shape!=(len(ids),4096) or not torch.isfinite(states).all()):
                raise ValueError("invalid retained helper states")
            stopped=ids[-1] in eog
            if result["stopped"]!=stopped or result["capped"]==stopped or (not stopped and len(ids)!=64):
                raise ValueError("source termination differs")
            answer=tokenizer.decode(ids[:-1] if stopped else ids,skip_special_tokens=False)
            if answer!=result["text"]:
                raise ValueError("helper text differs from retained choices")
            if route!="text":
                h,w=row["images"][route]["shape"];meta=result["metadata"]
                if (meta["visual_positions"]!=(h//32)*(w//32)
                        or meta["sequence_length"]!=len(saved["before"])+(h//32)*(w//32)+len(saved["after"])
                        or meta["rotary_offset"]!=len(saved["before"])+max(h//32,w//32)+len(saved["after"])):
                    raise ValueError("physical positions differ from native structural frame")
            references=[s.strip() for s in row["references"]];text=answer.strip()
            accuracy=max(max(0.,1-distance(text,s)/max(len(text),len(s),1)) for s in references)
            included=any(s and s in text for s in references)
            other_included=any(s.strip() and s.strip() in text for s in other["references"])
            items.append(dict(position=row["position"],route=route,answer=answer,stopped=stopped,
                exact=bool(stopped and text in references),character_accuracy=accuracy,
                reference_included=included,selective_reference_included=included and not other_included))
            state_batches.append(states);selected.extend(ids);recorded_logits.extend(result["selected_logits"])
    if len(items)!=40 or images!=32:
        raise ValueError("all 40 generations and 32 physical inputs required")
    # Recompute the helper's symbolic projection only; never initialize a model
    # or put a vocabulary table into the independent image interface here.
    source_head=archive.omitted["output.weight"]
    head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),copy=True)).to(torch.bfloat16).float()
    states=torch.cat(state_batches);tokens=torch.tensor(selected);recorded=torch.tensor(recorded_logits)
    agreements=0;largest_error=0.;largest_regret=0.;ties=0
    with torch.no_grad():
        for start in range(0,len(states),16):
            logits=F.linear(states[start:start+16],head)
            chosen=tokens[start:start+16];selected_value=logits.gather(1,chosen[:,None]).flatten()
            top=logits.max(1)
            error=float((selected_value-recorded[start:start+16]).abs().max())
            regret=float((top.values-selected_value).max())
            largest_error=max(largest_error,error);largest_regret=max(largest_regret,regret)
            agreements+=int((top.indices==chosen).sum());ties+=int((top.indices!=chosen).sum())
    if largest_error>1e-3 or largest_regret>1e-4:
        raise ValueError("temporary source projection does not replay within the fixed numerical tolerance")
    summaries={}
    for route in ("raw/normal","raw/blank","native_budget/normal","native_budget/blank","text"):
        subset=[r for r in items if r["route"]==route]
        summaries[route]=dict(count=len(subset),exact=sum(x["exact"] for x in subset),
            stopped=sum(x["stopped"] for x in subset),character_accuracy=float(np.mean([x["character_accuracy"] for x in subset])),
            reference_included=sum(x["reference_included"] for x in subset),
            selective_reference_included=sum(x["selective_reference_included"] for x in subset))
    result=dict(generation_report_sha256=sha(out/"generation_report.json"),verifier_sha256=sha(Path(__file__)),
        source_model_sha256=report["provenance"]["model_sha256"],source_images_verified=32,helper_generations_verified=40,
        helper_projection_positions=len(selected),helper_projection_argmax_agreements=agreements,
        numerical_near_tie_disagreements=ties,projection_selected_logit_error_max=largest_error,
        projection_choice_regret_max=largest_regret,measurements=items,summaries=summaries,
        offline_checks_pass=True,whole_vision_context_forward_replayed=False,helper_only_symbolic_decoding=True,
        image_actor_generated_images=0,language_quality_claim=False,general_ilm_goal_complete=False,
        elapsed_verification_seconds=time.perf_counter()-begun)
    atomic_json(out/"verification.json",result);atomic_json(out/"status.json",dict(stage="completed_verified",pid=None))
    print(json.dumps(dict(summaries=summaries,projection_positions=len(selected),argmax_agreements=agreements),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    verify(parser.parse_args().out)

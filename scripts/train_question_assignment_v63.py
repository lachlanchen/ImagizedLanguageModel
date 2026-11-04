#!/usr/bin/env python3
"""Full image-language adaptation with matched correct/swapped answer streams."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from ilm.visual_lm.correlated_native_runtime import load_correlated
from ilm.visual_lm.question_assignment import QuestionImagePairs,assignment_objective,image_assignment_energies
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json,_atomic_save,_scheduled_lr
from scripts.train_correlated_native_density_v62 import configure_gpu
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT

INITIAL="artifacts/correlated_native_density_v62_20261105/k8/model_final.pt"
INITIAL_SHA="9e22ebccbd769cc44e7de026ca89e433ae934ddeaf808af01020552f24d1b1b9"
DEFAULT_PAIRS="artifacts/question_assignment_v63_20261105/data"
SOURCES=["scripts/train_question_assignment_v63.py","ilm/visual_lm/question_assignment.py",
    "ilm/visual_lm/correlated_native_density.py","ilm/visual_lm/correlated_native_runtime.py",
    "scripts/train_correlated_native_density_v62.py","scripts/train_native_pixel_instruction_v61.py",
    "scripts/train_canonical_glyph_binding_v43.py","references/question_assignment_v63_protocol.md"]


def repair_journal(path,update):
    if not path.exists():return
    lines=path.read_text().splitlines();keep=[];tail=[]
    for line in lines:
        try:valid=json.loads(line)["update"]<=update
        except (ValueError,KeyError):valid=False
        (keep if valid else tail).append(line)
    if tail:
        archive=path.with_name(path.stem+f"_uncommitted_{time.time_ns()}.jsonl")
        archive.write_text("\n".join(tail)+"\n")
        path.write_text("\n".join(keep)+("\n" if keep else ""))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=("control","aligned"),required=True)
    parser.add_argument("--out",required=True);parser.add_argument("--pairs",default=DEFAULT_PAIRS)
    parser.add_argument("--snapshot",default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--resume");parser.add_argument("--stop-after",type=int)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--longest-pair",action="store_true",help="resource preflight, never production")
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/"model_final.pt").exists():raise FileExistsError("completed arm is immutable")
    if not args.resume and (out/"checkpoint_latest.pt").exists():raise FileExistsError("resume existing state")
    if args.longest_pair and not args.smoke:raise ValueError("longest pair is a preflight only")
    if sha256(INITIAL)!=INITIAL_SHA:raise ValueError("initial learned weights differ")
    device=configure_gpu();random.seed(20266320);np.random.seed(20266320)
    torch.manual_seed(20266320);torch.cuda.manual_seed(20266320)
    data=QuestionImagePairs(args.pairs);provenance={p:sha256(p) for p in SOURCES}
    data_sha=sha256(Path(args.pairs)/"report.json");steps,accumulation=(4,1) if args.smoke else (2000,4)
    weight=float(args.arm=="aligned");start=0;prior_elapsed=0.;prior_peak=0.
    loading=time.perf_counter();model=load_correlated(INITIAL,args.snapshot,device).requires_grad_(True)
    model.backbone.config.use_cache=False
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    ema=copy.deepcopy(model).requires_grad_(False).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.95),weight_decay=.01,fused=True)
    generator=torch.Generator().manual_seed(20266321)
    initial=torch.load(INITIAL,map_location="cpu",weights_only=False)
    metadata={k:initial[k] for k in ("config","external_revision","external_weight_sha256","external_code_sha256")}
    del initial
    loading=time.perf_counter()-loading
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=False)
        if any((saved["provenance"]!=provenance,saved["data_sha256"]!=data_sha,
                saved["arm"]!=args.arm,saved["smoke_only"]!=args.smoke,
                saved["longest_pair"]!=args.longest_pair,saved["initialization_sha256"]!=INITIAL_SHA)):
            raise ValueError("resume source/data/recipe mismatch")
        model.load_state_dict(saved["model"]);ema.load_state_dict(saved["ema_model"])
        optimizer.load_state_dict(saved["optimizer"]);generator.set_state(saved["data_rng"])
        random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["cpu_rng"]);torch.cuda.set_rng_state(saved["cuda_rng"],device)
        start,prior_elapsed,prior_peak=saved["update"],saved["elapsed_seconds"],saved["peak_allocated_vram_gib"]
        del saved
        for name in ("training_metrics.jsonl","sample_order.jsonl"):repair_journal(out/name,start)
    if start>=steps:raise ValueError("resume is complete")
    stopping=False
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    begun=time.perf_counter();checks=None
    longest=max(range(len(data.pairs)),key=lambda i:data.pairs[i]["maximum_joined_blocks"])
    try:
        for update in range(start+1,steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            rate=_scheduled_lr(update,base=2e-5,warmup=min(100,steps//2),total=steps,minimum_ratio=.1)
            optimizer.param_groups[0]["lr"]=rate;metrics=[];indices=[]
            for _ in range(accumulation):
                index=longest if args.longest_pair else data.sample(generator);indices.append(index)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    energies=image_assignment_energies(model,data.streams(index),device)
                    loss,values=assignment_objective(energies,weight)
                if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite image assignment loss")
                (loss/accumulation).backward()
                metrics.append({k:float(v.detach()) for k,v in values.items()})
            if update==start+1:
                names=("backbone.in_proj.weight","backbone.layers.0.self_attn.q_proj.weight",
                    "backbone.layers.23.self_attn.q_proj.weight","head.projection.weight","head.gate.weight")
                params=dict(model.named_parameters())
                checks={n:bool(params[n].grad is not None and torch.isfinite(params[n].grad).all() and params[n].grad.abs().sum()>0) for n in names}
                checks["all_parameters_have_finite_gradients"]=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not all(checks.values()):raise RuntimeError(checks)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.01)
            torch.cuda.synchronize(device)
            elapsed=prior_elapsed+time.perf_counter()-begun
            peak=max(prior_peak,torch.cuda.max_memory_allocated(device)/1024**3)
            row=dict(update=update,**{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                lr=rate,gradient_norm_before_clip=float(norm),elapsed_seconds=elapsed,peak_allocated_vram_gib=peak)
            if update==start+1 or update%10==0 or args.smoke:
                print(json.dumps(row),flush=True)
                with (out/"training_metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            with (out/"sample_order.jsonl").open("a") as f:f.write(json.dumps(dict(update=update,pair_indices=indices))+"\n")
            exceeded=peak>=20 or elapsed>8*3600
            stopping|=exceeded or (args.stop_after is not None and update>=args.stop_after)
            if update%250==0 or update==steps or stopping:
                saved=dict(architecture="correlated-native-v62",experiment="question-assignment-v63",arm=args.arm,
                    components=8,update=update,steps=steps,accumulation=accumulation,assignment_weight=weight,
                    model=model.state_dict(),ema_model=ema.state_dict(),optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),data_rng=generator.get_state(),**metadata,provenance=provenance,
                    data_sha256=data_sha,initialization_sha256=INITIAL_SHA,arguments=vars(args),
                    elapsed_seconds=elapsed,peak_allocated_vram_gib=peak,loading_seconds=loading,
                    smoke_only=args.smoke,longest_pair=args.longest_pair,backward_checks=checks,
                    trained_parameters=sum(p.numel() for p in model.parameters()),
                    boundary=dict(tokenizer=False,unicode_ids=False,ocr=False,glyph_inventory=False,external_model_call=False),
                    qualification="unqualified question-conditioned free image-language adaptation")
                _atomic_save(saved,out/"checkpoint_latest.pt")
                if update==steps and not exceeded:
                    final={k:v for k,v in saved.items() if k not in ("model","optimizer","cpu_rng","cuda_rng","python_rng","numpy_rng","data_rng")}
                    _atomic_save(final,out/"model_final.pt");_atomic_json({k:v for k,v in final.items() if k!="ema_model"},out/"training_report.json")
                if exceeded:raise RuntimeError("resource limit; state saved")
                if stopping:break
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=="__main__":main()

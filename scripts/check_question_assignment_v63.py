#!/usr/bin/env python3
"""Real memory, assignment-gradient, causality and exact-resume preflight."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch

from ilm.visual_lm.correlated_native_runtime import load_correlated
from ilm.visual_lm.question_assignment import QuestionImagePairs,assignment_objective,image_assignment_energies
from scripts.check_correlated_native_preflight_v62 import equal
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json
from scripts.train_correlated_native_density_v62 import configure_gpu
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT
from scripts.train_question_assignment_v63 import DEFAULT_PAIRS,INITIAL,SOURCES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="artifacts/question_assignment_v63_20261105/preflight")
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/"report.json").exists():raise FileExistsError("completed preflight is immutable")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("physical GPU 0 only")
    lock=(out/"preflight.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;stopping=False
    def record(stage,**values):
        _atomic_json(dict(stage=stage,pid=os.getpid(),child_pid=child.pid if child else None,**values),out/"status.json")
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    expected={p:sha256(p) for p in SOURCES}
    def execute(stage,arguments):
        nonlocal child
        if stopping:raise InterruptedError("preflight stopped")
        with (out/(stage+".log")).open("a") as log:
            child=subprocess.Popen([sys.executable,"scripts/train_question_assignment_v63.py",*arguments],stdout=log,stderr=subprocess.STDOUT)
            record(stage);result=child.wait();child=None
        if stopping or result:raise RuntimeError(f"{stage} did not finish: {result}")
        print(stage+" finished",flush=True)
    try:
        resource=out/"longest_pair"
        if not (resource/"checkpoint_latest.pt").exists():
            execute("longest_pair",["--arm","aligned","--smoke","--longest-pair","--stop-after","1","--out",str(resource)])
        saved=torch.load(resource/"checkpoint_latest.pt",map_location="cpu",weights_only=False)
        if saved["update"]!=1 or not saved["longest_pair"] or saved["provenance"]!=expected:raise ValueError("resource witness differs")
        resource_peak=saved["peak_allocated_vram_gib"];checks=dict(longest_pair_fits=resource_peak<20,full_backward=all(saved["backward_checks"].values()))
        del saved
        arms={}
        for arm in ("control","aligned"):
            full,resume=out/(arm+"_full"),out/(arm+"_resume")
            base=["--arm",arm,"--smoke"]
            if not (full/"model_final.pt").exists():execute(arm+"_full",base+["--out",str(full)])
            if not (resume/"checkpoint_latest.pt").exists():execute(arm+"_interrupted",base+["--out",str(resume),"--stop-after","2"])
            progress=torch.load(resume/"checkpoint_latest.pt",map_location="cpu",weights_only=False);update=progress["update"];del progress
            if update<4:execute(arm+"_resumed",base+["--out",str(resume),"--resume",str(resume/"checkpoint_latest.pt")])
            a=torch.load(full/"checkpoint_latest.pt",map_location="cpu",weights_only=False)
            b=torch.load(resume/"checkpoint_latest.pt",map_location="cpu",weights_only=False)
            if a["provenance"]!=expected or b["provenance"]!=expected or a["update"]!=4 or b["update"]!=4:
                raise ValueError("resume preflight identity differs")
            for key in ("model","ema_model","optimizer","cpu_rng","cuda_rng","python_rng","numpy_rng","data_rng"):
                checks[arm+"_exact_"+key]=equal(a[key],b[key])
            checks[arm+"_sample_order"]=(full/"sample_order.jsonl").read_bytes()==(resume/"sample_order.jsonl").read_bytes()
            checks[arm+"_finite_backward"]=all(a["backward_checks"].values()) and all(b["backward_checks"].values())
            arms[arm]=dict(parameters=a["trained_parameters"],peak_allocated_vram_gib=a["peak_allocated_vram_gib"],final_ema_sha256=sha256(full/"model_final.pt"))
            del a,b
        record("gradient_and_causality")
        data=QuestionImagePairs(DEFAULT_PAIRS);index=max(range(len(data.pairs)),key=lambda i:data.pairs[i]["maximum_joined_blocks"])
        streams=data.streams(index);p0,b0=streams[0];p1,b1=streams[3]
        checks["physical_stream_assembly"]=(torch.equal(streams[1][0][...,:b0*64],p0[...,:b0*64]) and
            torch.equal(streams[1][0][...,b0*64:],p1[...,b1*64:]) and
            torch.equal(streams[2][0][...,:b1*64],p1[...,:b1*64]) and
            torch.equal(streams[2][0][...,b1*64:],p0[...,b0*64:]))
        device=configure_gpu();model=load_correlated(INITIAL,DEFAULT_SNAPSHOT,device).requires_grad_(True).train()
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            energies=image_assignment_energies(model,streams,device);_,values=assignment_objective(energies,1.)
        margin=float(values["assignment_margin"].detach());values["assignment_loss"].backward()
        names=("backbone.in_proj.weight","backbone.layers.0.self_attn.q_proj.weight",
            "backbone.layers.23.self_attn.q_proj.weight","head.gate.weight","head.projection.weight")
        params=dict(model.named_parameters());gradient_norms={n:float(params[n].grad.float().norm()) for n in names}
        checks["assignment_only_gradient"]=all(v>0 and torch.isfinite(torch.tensor(v)) for v in gradient_norms.values())
        model.zero_grad(set_to_none=True);del energies,values
        pixels=p0.to(device);altered=pixels.clone();altered[...,b0*64:]*=-1
        mask=torch.ones(1,pixels.shape[-1]//64,device=device);model.eval()
        with torch.no_grad():
            a=model(pixels,mask);b=model(altered,mask)
            causal=max(float((x[:,:b0]-y[:,:b0]).abs().max()) for x,y in zip(a[:2],b[:2]))
            del a,b,altered
            torch.set_float32_matmul_precision("highest");torch.backends.cudnn.allow_tf32=False
            full=model(pixels[...,:b0*64],mask[:,:b0])
            prefix=model(pixels[...,:64*(b0-1)],mask[:,:b0-1],use_cache=True)
            last=model(pixels[...,64*(b0-1):b0*64],mask[:,:b0],use_cache=True,past_key_values=prefix[2])
            cache_error=max(float((x[:,-1]-y[:,-1]).abs().max()) for x,y in zip(full[:2],last[:2]))
        checks["future_pixel_causality"]=causal==0;checks["strict_fp32_cache"]=cache_error<2e-4
        del model,pixels,full,prefix,last;torch.cuda.empty_cache()
        receipt=dict(checks=checks,arms=arms,longest_pair_peak_allocated_vram_gib=resource_peak,
            assignment_only_gradient_norms=gradient_norms,initial_assignment_margin=margin,
            future_pixel_max_error=causal,cached_prefix_strict_fp32_max_error=cache_error,
            provenance=expected,data_sha256=sha256(Path(DEFAULT_PAIRS)/"report.json"),
            verifier_sha256=sha256(__file__),training_only=True,development_examples=0,language_qualified=False)
        if not all(checks.values()):
            _atomic_json(receipt,out/"failed_verification.json");raise AssertionError(checks)
        _atomic_json(receipt,out/"report.json");record("complete",checks_passed=True)
        print(json.dumps(receipt,indent=2),flush=True)
    except BaseException as error:
        record("interrupted" if stopping else "failed",error=repr(error));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=="__main__":main()

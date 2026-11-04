#!/usr/bin/env python3
"""Real full-backbone interrupted/resumed training checks for both V62 arms."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from ilm.visual_lm.correlated_native_runtime import load_correlated
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json
from scripts.train_correlated_native_density_v62 import SOURCES, configure_gpu
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT, DEFAULT_DATA, ImageData


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="artifacts/correlated_native_density_v62_20261105/preflight")
    parser.add_argument("--verify-only",action="store_true",help="recheck completed smoke checkpoints without retraining")
    args=parser.parse_args();out=Path(args.out)
    if (out/"report.json").exists():
        raise FileExistsError("completed preflight is immutable")
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("physical GPU 0 required")
    checks,arms={},{}
    for k in (1,8):
        full,resume=out/f"k{k}_full",out/f"k{k}_resume"
        base=[sys.executable,"scripts/train_correlated_native_density_v62.py","--components",str(k),"--smoke"]
        for name,command in (("full",base+["--out",str(full)]),
            ("interrupted",base+["--out",str(resume),"--stop-after","2"]),
            ("resumed",base+["--out",str(resume),"--resume",str(resume/"checkpoint_latest.pt")])):
            if not args.verify_only:
                with (out/f"k{k}_{name}.log").open("w") as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                print(f"K={k} {name} complete",flush=True)
        a=torch.load(full/"checkpoint_latest.pt",map_location="cpu",weights_only=False)
        b=torch.load(resume/"checkpoint_latest.pt",map_location="cpu",weights_only=False)
        expected={name:sha256(name) for name in SOURCES}
        if a["provenance"]!=expected or b["provenance"]!=expected:
            raise ValueError("completed smoke source identity changed")
        for key in ("model","ema_model","optimizer","cpu_rng","cuda_rng","python_rng","numpy_rng","data_rng"):
            checks[f"k{k}_exact_{key}"]=equal(a[key],b[key])
        checks[f"k{k}_full_backward"]=all(a["backward_checks"].values()) and all(b["backward_checks"].values())
        checks[f"k{k}_same_sample_order"]=(full/"sample_order.jsonl").read_bytes()==(resume/"sample_order.jsonl").read_bytes()
        arms[str(k)]=dict(parameters=a["trained_parameters"],peak_allocated_vram_gib=a["peak_allocated_vram_gib"],
            final_ema_sha256=sha256(full/"model_final.pt"))
        del a,b
        device=configure_gpu()
        model=load_correlated(full/"model_final.pt",DEFAULT_SNAPSHOT,device,allow_smoke=True)
        cache=ImageData(DEFAULT_DATA);pixels,boundary=cache.get(cache.training[0]);pixels=pixels.to(device)
        altered=pixels.clone();altered[...,boundary*64:]*=-1
        mask=torch.ones(1,pixels.shape[-1]//64,device=device)
        with torch.no_grad():
            original=model(pixels,mask);future=model(altered,mask)
            error=max(float((x[:,:boundary]-y[:,:boundary]).abs().max())
                      for x,y in zip(original[:2],future[:2]))
            first=model(pixels[...,:boundary*64],mask[:,:boundary])
            prefix=model(pixels[...,:64*(boundary-1)],mask[:,:boundary-1],use_cache=True)
            last=model(pixels[...,64*(boundary-1):64*boundary],mask[:,:boundary],
                       use_cache=True,past_key_values=prefix[2])
            cache_error=max(float((x[:,-1]-y[:,-1]).abs().max()) for x,y in zip(first[:2],last[:2]))
            mode_changes=int(((first[0][:,-1]>0)!=(last[0][:,-1]>0)).sum())
            # TF32 selects different matrix/conv kernels for long prefill and
            # one-block decoding. Verify cache semantics with strict FP32;
            # keep the fast-kernel discrepancy as separate numerical evidence.
            torch.set_float32_matmul_precision("highest");torch.backends.cudnn.allow_tf32=False
            first=model(pixels[...,:boundary*64],mask[:,:boundary])
            prefix=model(pixels[...,:64*(boundary-1)],mask[:,:boundary-1],use_cache=True)
            last=model(pixels[...,64*(boundary-1):64*boundary],mask[:,:boundary],
                       use_cache=True,past_key_values=prefix[2])
            strict_error=max(float((x[:,-1]-y[:,-1]).abs().max()) for x,y in zip(first[:2],last[:2]))
        checks[f"k{k}_future_pixel_causality"]=error==0
        checks[f"k{k}_cached_prefix_equivalence"]=strict_error<2e-4
        arms[str(k)].update(future_pixel_max_error=error,cached_prefix_strict_fp32_max_error=strict_error,
            tf32_cached_prefix_max_error=cache_error,tf32_mode_bit_changes=mode_changes)
        del model,original,future,first,prefix,last,pixels,altered
        torch.cuda.empty_cache()
    receipt=dict(checks=checks,arms=arms,provenance={name:sha256(name) for name in SOURCES},
        verifier_sha256=sha256(__file__),training_only=True,development_examples=0,
        numerical_note="Initial TF32 2e-4 cache check failed for both arms; strict FP32 verifies cache semantics separately. Training settings unchanged.")
    if not all(checks.values()):
        _atomic_json(receipt,out/"failed_verification.json")
        raise AssertionError(checks)
    _atomic_json(receipt,out/"report.json")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":main()

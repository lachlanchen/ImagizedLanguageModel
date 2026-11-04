#!/usr/bin/env python3
"""One real full-actor update to measure feasibility, without development use."""

import copy
import json
import os
from pathlib import Path
import time

import torch

from ilm.visual_lm.correlated_native_density import CorrelatedNativeLanguage
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT,DEFAULT_DATA,ImageData,load_base


def main():
    out=Path("artifacts/correlated_native_density_v62_preflight.json")
    if out.exists():raise FileExistsError("completed feasibility preflight exists")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("physical GPU 0 required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high");torch.manual_seed(20266210)
    device=torch.device("cuda:0");torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    author=load_base(DEFAULT_SNAPSHOT,device)
    source_parameters=sum(v.numel() for v in author.parameters())
    model=CorrelatedNativeLanguage.from_author(author,components=8)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.train();ema=copy.deepcopy(model).requires_grad_(False).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.95),weight_decay=.01,fused=True)
    data=ImageData(DEFAULT_DATA);pixels,boundary=data.get(data.training[0]);pixels=pixels.to(device)
    before=model.head.projection.weight.detach().clone()
    begun=time.perf_counter()
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):loss=model.answer_nll(pixels,boundary)
    if not torch.isfinite(loss):raise FloatingPointError("nonfinite mixture likelihood")
    loss.backward()
    names=("backbone.in_proj.weight","backbone.layers.0.self_attn.q_proj.weight",
           "backbone.layers.23.self_attn.q_proj.weight","head.projection.weight","head.gate.weight")
    params=dict(model.named_parameters())
    checks={name:bool(params[name].grad is not None and torch.isfinite(params[name].grad).all()
                      and params[name].grad.abs().sum()>0) for name in names}
    checks["all_parameter_gradients_finite"]=all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    checks["real_optimizer_update"]=not torch.equal(before,model.head.projection.weight)
    with torch.no_grad():
        for ep,value in zip(ema.parameters(),model.parameters()):ep.lerp_(value,.01)
    torch.cuda.synchronize(device)
    peak=torch.cuda.max_memory_allocated(device)/1024**3
    checks["memory_below_20_gib"]=peak<20
    if not all(checks.values()):raise AssertionError(checks)
    receipt=dict(checks=checks,parameters=sum(v.numel() for v in model.parameters()),
        source_parameters=source_parameters,components=8,peak_allocated_vram_gib=peak,
        one_update_seconds=time.perf_counter()-begun,answer_pixel_nll=float(loss.detach()),
        source_sha256={name:sha256(name) for name in [__file__,"ilm/visual_lm/correlated_native_density.py",
                       "references/correlated_native_density_v62_preflight_plan.md"]},
        training_only_example=True,development_examples=0,language_quality_claim=False,
        status="feasibility_only; matched training protocol still required")
    _atomic_json(receipt,out);print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":main()

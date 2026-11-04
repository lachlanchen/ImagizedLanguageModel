#!/usr/bin/env python3
"""Matched full-backbone K=1/K=8 whole-image likelihood adaptation, GPU 0."""

from __future__ import annotations

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

from ilm.visual_lm.correlated_native_density import CorrelatedNativeLanguage
from ilm.visual_lm.native_pixel_instruction import REVISION, WEIGHT_SHA256
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save, _scheduled_lr
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT, DEFAULT_DATA, ImageData, load_base

SOURCES = ["scripts/train_correlated_native_density_v62.py",
    "ilm/visual_lm/correlated_native_density.py", "ilm/visual_lm/correlated_native_runtime.py",
    "ilm/visual_lm/native_pixel_instruction.py", "scripts/train_native_pixel_instruction_v61.py",
    "references/correlated_native_density_v62_protocol.md"]


def configure_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("explicit physical CUDA_VISIBLE_DEVICES=0 required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, choices=(1, 8), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--resume"); parser.add_argument("--stop-after", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/"model_final.pt").exists():
        raise FileExistsError("completed arm is immutable")
    if not args.resume and (out/"checkpoint_latest.pt").exists():
        raise FileExistsError("use --resume for an interrupted arm")
    device = configure_gpu()
    random.seed(20266220); np.random.seed(20266220); torch.manual_seed(20266220)
    torch.cuda.manual_seed(20266220)
    provenance = {name:sha256(name) for name in SOURCES}
    code = {name:sha256(Path(args.snapshot)/name) for name in
            ("modeling_llama_pixar.py", "configuration_llama_pixar.py", "utils_llama_pixar.py", "config.json")}
    data = ImageData(args.data); data_sha = sha256(Path(args.data)/"report.json")
    steps, accumulation = (4, 2) if args.smoke else (1000, 8)
    t = time.perf_counter()
    author = load_base(args.snapshot, device)
    config = author.config.to_dict()
    model = CorrelatedNativeLanguage.from_author(author, args.components).requires_grad_(True)
    del author
    model.backbone.config.use_cache = False
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9,.95), weight_decay=.01, fused=True)
    generator = torch.Generator().manual_seed(20266221)
    loading = time.perf_counter()-t
    start, prior_elapsed, prior_peak = 0, 0., 0.
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if any((saved["provenance"] != provenance, saved["data_sha256"] != data_sha,
                saved["components"] != args.components, saved["smoke_only"] != args.smoke,
                saved["external_code_sha256"] != code)):
            raise ValueError("resume identity mismatch")
        model.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema_model"])
        optimizer.load_state_dict(saved["optimizer"]); generator.set_state(saved["data_rng"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["cpu_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"], device)
        start, prior_elapsed, prior_peak = saved["update"], saved["elapsed_seconds"], saved["peak_allocated_vram_gib"]
        del saved
    if start >= steps:
        raise ValueError("resume is already complete")
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    handlers = {sig:signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    begun = time.perf_counter()
    checks = None
    try:
        for update in range(start+1, steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            rate = _scheduled_lr(update, base=2e-5, warmup=min(50,steps//2), total=steps, minimum_ratio=.1)
            optimizer.param_groups[0]["lr"] = rate
            losses, ids = [], []
            for _ in range(accumulation):
                index = int(torch.randint(len(data.training),(1,),generator=generator))
                relative = data.training[index]; pixels, boundary = data.get(relative)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model.answer_nll(pixels.to(device), boundary)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite joint image NLL")
                (loss/accumulation).backward()
                losses.append(float(loss.detach())); ids.append(relative)
            if update == start+1:
                names = ["backbone.in_proj.weight", "backbone.layers.0.self_attn.q_proj.weight",
                    "backbone.layers.23.self_attn.q_proj.weight", "head.projection.weight"]
                if args.components > 1:
                    names.append("head.gate.weight")
                params = dict(model.named_parameters())
                checks = {name:bool(params[name].grad is not None and torch.isfinite(params[name].grad).all()
                    and params[name].grad.abs().sum()>0) for name in names}
                checks["all_parameters_have_finite_gradients"] = all(v.grad is not None and
                    bool(torch.isfinite(v.grad).all()) for v in model.parameters())
                if not all(checks.values()):
                    raise RuntimeError(checks)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep,value in zip(ema.parameters(),model.parameters()):
                    ep.lerp_(value,.01)
            torch.cuda.synchronize(device)
            elapsed = prior_elapsed+time.perf_counter()-begun
            peak = max(prior_peak,torch.cuda.max_memory_allocated(device)/1024**3)
            metrics = dict(update=update,answer_pixel_nll=float(np.mean(losses)),lr=rate,
                gradient_norm_before_clip=float(norm),elapsed_seconds=elapsed,peak_allocated_vram_gib=peak)
            if update == start+1 or update%10 == 0 or args.smoke:
                print(json.dumps(metrics),flush=True)
                with (out/"training_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics)+"\n")
            # Complete data order is evidence of matching, without answer text.
            with (out/"sample_order.jsonl").open("a") as f:
                f.write(json.dumps(dict(update=update,images=ids))+"\n")
            exceeded = peak>=20 or elapsed>4*3600
            stopping |= exceeded or (args.stop_after is not None and update>=args.stop_after)
            if update%250 == 0 or update == steps or stopping:
                saved = dict(architecture="correlated-native-v62",components=args.components,update=update,
                    model=model.state_dict(),ema_model=ema.state_dict(),optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),data_rng=generator.get_state(),
                    provenance=provenance,data_sha256=data_sha,arguments=vars(args),config=config,
                    external_revision=REVISION,external_weight_sha256=WEIGHT_SHA256,external_code_sha256=code,
                    elapsed_seconds=elapsed,peak_allocated_vram_gib=peak,loading_seconds=loading,
                    smoke_only=args.smoke,steps=steps,accumulation=accumulation,backward_checks=checks,
                    trained_parameters=sum(v.numel() for v in model.parameters()),
                    boundary=dict(tokenizer=False,unicode_ids=False,ocr=False,glyph_inventory=False,external_model_call=False),
                    qualification="unqualified free image-language adaptation")
                _atomic_save(saved,out/"checkpoint_latest.pt")
                if update == steps and not exceeded:
                    final = {k:v for k,v in saved.items() if k not in
                        ("model","optimizer","cpu_rng","cuda_rng","python_rng","numpy_rng","data_rng")}
                    _atomic_save(final,out/"model_final.pt")
                    _atomic_json({k:v for k,v in final.items() if k!="ema_model"},out/"training_report.json")
                if exceeded:
                    raise RuntimeError("resource limit; resumable state saved")
                if stopping:
                    break
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == "__main__":
    main()

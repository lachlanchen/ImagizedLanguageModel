#!/usr/bin/env python3
"""Fine-tune every native pixel-language parameter on answer images, GPU 0."""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from ilm.visual_lm.native_pixel_instruction import REVISION, WEIGHT_SHA256, answer_only_bce
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save, _scheduled_lr

DEFAULT_SNAPSHOT = (Path.home()/".cache/huggingface/hub" /
    "models--TYTTYTTYT--llama-pixar-477m-p32-stage2/snapshots"/REVISION)
DEFAULT_DATA = "artifacts/native_pixel_instruction_v61_20261105/data"
SOURCES = ["scripts/train_native_pixel_instruction_v61.py",
           "ilm/visual_lm/native_pixel_instruction.py",
           "scripts/prepare_native_pixel_qa_v61.py",
           "references/native_pixel_instruction_v61_protocol.md"]


class ImageData:
    def __init__(self, path):
        self.path = Path(path)
        self.report = json.loads((self.path/"report.json").read_text())
        if self.report["counts"] != {"train": 3211, "development": 1321}:
            raise ValueError("incomplete native image data")
        for name, digest in self.report["sources"].items():
            if sha256(name) != digest:
                raise ValueError("prepared native image source changed")
        self.training = sorted(name for name in self.report["files_sha256"] if name.startswith("train/"))

    @lru_cache(maxsize=128)
    def get(self, relative):
        path = self.path/relative
        if sha256(path) != self.report["files_sha256"][relative]:
            raise ValueError("packed image cache changed")
        row = torch.load(path, map_location="cpu", weights_only=False)
        bits = np.unpackbits(row["bits"].numpy(), count=32*row["width"])
        pixels = torch.from_numpy(bits.copy()).reshape(1, 1, 32, row["width"]).float().mul(2).sub(1)
        return pixels, row["prompt_blocks"]


def load_base(snapshot, device):
    from transformers import AutoModelForCausalLM
    snapshot = Path(snapshot)
    if snapshot.name != REVISION or sha256(snapshot/"model.safetensors") != WEIGHT_SHA256:
        raise ValueError("pinned source release mismatch")
    return AutoModelForCausalLM.from_pretrained(str(snapshot), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/"model_final.pt").exists():
        raise FileExistsError("completed native model exists")
    steps, accumulation = (4, 2) if a.smoke else (1000, 8)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("explicit CUDA_VISIBLE_DEVICES=0 required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(20266110); np.random.seed(20266110); torch.manual_seed(20266110)
    torch.cuda.manual_seed(20266110); torch.cuda.reset_peak_memory_stats(device)
    provenance = {name: sha256(name) for name in SOURCES}
    source_code = {name: sha256(Path(a.snapshot)/name) for name in
                   ("modeling_llama_pixar.py", "configuration_llama_pixar.py", "utils_llama_pixar.py", "config.json")}
    data = ImageData(a.data); data_sha = sha256(Path(a.data)/"report.json")
    loading = time.perf_counter()
    model = load_base(a.snapshot, device).requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95),
                                 weight_decay=.01, fused=True)
    generator = torch.Generator().manual_seed(20266111)
    loading = time.perf_counter()-loading
    prior_elapsed, prior_peak, start = 0., 0., 0
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance or saved["data_sha256"] != data_sha or saved["smoke_only"] != a.smoke:
            raise ValueError("resume identity mismatch")
        if saved["external_code_sha256"] != source_code:
            raise ValueError("author implementation changed")
        model.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema_model"])
        optimizer.load_state_dict(saved["optimizer"]); generator.set_state(saved["data_rng"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["cpu_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"], device)
        prior_elapsed, prior_peak, start = saved["elapsed_seconds"], saved["peak_allocated_vram_gib"], saved["update"]
        del saved
    if start >= steps:
        raise ValueError("resume is already complete")
    stopping = False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    begun = time.perf_counter()
    checks = None
    try:
        for update in range(start+1, steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            rate = _scheduled_lr(update, base=2e-5, warmup=min(50, steps//2), total=steps, minimum_ratio=.1)
            optimizer.param_groups[0]["lr"] = rate
            losses = []
            for _ in range(accumulation):
                index = int(torch.randint(len(data.training), (1,), generator=generator))
                pixels, boundary = data.get(data.training[index]); pixels = pixels.to(device)
                mask = torch.ones(1, pixels.shape[-1]//64, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(inputs_embeds=pixels, attention_mask=mask, use_cache=False)
                    loss = answer_only_bce(prediction.logits, pixels, boundary)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite native answer-image NLL")
                (loss/accumulation).backward(); losses.append(float(loss.detach()))
            if update == start+1:
                names = ("model.in_proj.weight", "model.layers.0.self_attn.q_proj.weight",
                         "model.layers.23.self_attn.q_proj.weight", "out_proj.weight")
                params = dict(model.named_parameters())
                checks = {name: bool(params[name].grad is not None and
                    torch.isfinite(params[name].grad).all() and params[name].grad.abs().sum() > 0)
                          for name in names}
                checks["all_parameters_have_finite_gradients"] = all(
                    v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters())
                if not all(checks.values()):
                    raise RuntimeError(f"full native backward failed: {checks}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, value in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(value, .01)
            elapsed = prior_elapsed+time.perf_counter()-begun
            peak = max(prior_peak, torch.cuda.max_memory_allocated(device)/1024**3)
            metrics = dict(update=update, answer_pixel_nll=float(np.mean(losses)),
                gradient_norm_before_clip=float(norm), lr=rate, elapsed_seconds=elapsed,
                peak_allocated_vram_gib=peak)
            if update == start+1 or update % 10 == 0 or a.smoke:
                print(json.dumps(metrics), flush=True)
                with (out/"training_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics)+"\n")
            exceeded = peak >= 20 or elapsed > 4*3600
            stopping |= exceeded or (a.stop_after is not None and update >= a.stop_after)
            if update % 250 == 0 or update == steps or stopping:
                saved = dict(architecture="native-pixel-instruction-v61", update=update,
                    model=model.state_dict(), ema_model=ema.state_dict(), optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), data_rng=generator.get_state(),
                    provenance=provenance, data_sha256=data_sha, arguments=vars(a),
                    external_revision=REVISION, external_weight_sha256=WEIGHT_SHA256,
                    external_code_sha256=source_code, config=model.config.to_dict(),
                    elapsed_seconds=elapsed, peak_allocated_vram_gib=peak, loading_seconds=loading,
                    smoke_only=a.smoke, accumulation=accumulation, steps=steps,
                    trained_parameters=sum(v.numel() for v in model.parameters()), backward_checks=checks,
                    boundary=dict(tokenizer=False, unicode_ids=False, ocr=False, glyph_inventory=False,
                                  external_model_call=False, input="binary images", output="free binary images"),
                    qualification="unqualified full native image-language adaptation")
                _atomic_save(saved, out/"checkpoint_latest.pt")
                if update == steps and not exceeded:
                    final = {k: v for k, v in saved.items() if k not in
                             ("model", "optimizer", "cpu_rng", "cuda_rng", "python_rng", "numpy_rng", "data_rng")}
                    _atomic_save(final, out/"model_final.pt")
                    _atomic_json({k: v for k, v in final.items() if k != "ema_model"}, out/"training_report.json")
                if exceeded:
                    raise RuntimeError("native training resource limit; checkpoint saved")
                if stopping:
                    break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the registered wider writer with the unchanged image-feedback protocol."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.full_rank_visual_span import MODEL_SOURCES, load_full_rank_checkpoint
from scripts.infer_compact_visual_span import generate_case, inference_gpu, input_cases


def run(args, cases, device, admission=None):
    identity = {
        "checkpoint_sha256": sha(args.checkpoint),
        "inputs": [{"id": key, "sha256": digest, "glyphs": count} for key, _, count, digest in cases],
        "seed": args.seed, "flow_steps": args.flow_steps, "maximum_glyphs": args.maximum_glyphs,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": {name: sha(ROOT / name) for name in (
            *MODEL_SOURCES, "scripts/infer_compact_visual_span.py", "scripts/infer_full_rank_visual_span.py")},
    }
    request = args.output / "request.json"
    if request.exists():
        if json.loads(request.read_text()) != identity:
            raise ValueError("Resume requires the original checkpoint, images and generation settings")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save_json(request, identity)
    begin = time.perf_counter()
    model, updates, provenance = load_full_rank_checkpoint(args.checkpoint, args.local, args.codec, device)
    seconds = time.perf_counter() - begin
    rows = []
    for case in cases:
        row = generate_case(model, device, args, case, args.output)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("id", "emitted_glyphs", "stopped", "elapsed_seconds")}), flush=True)
    save_json(args.output / "report.json", {
        "request_sha256": sha(request), "optimizer_updates": updates, "provenance": provenance,
        "checkpoint_load_seconds": seconds, "total_generation_seconds": sum(row["elapsed_seconds"] for row in rows),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "reference_inputs": False, "output_repaired": False, "records": rows, "resource_admission": admission,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--glyph-count", type=int)
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=ROOT.parent / "LocalLLM")
    parser.add_argument("--codec", type=Path, default=ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--maximum-glyphs", type=int, default=128)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202611153)
    args = parser.parse_args()
    if (args.manifest and args.glyph_count is not None) or (args.image and args.normal_only):
        parser.error("--glyph-count applies to one image; --normal-only applies to a manifest")
    if min(args.maximum_glyphs, args.flow_steps) < 1:
        parser.error("Generation budgets must be positive")
    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Hide CUDA for CPU inference")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cases = input_cases(args)
    torch.set_num_threads(2)
    runtime = inference_gpu() if args.device == "cuda" else nullcontext((torch.device("cpu"), None))
    with runtime as (device, admission):
        run(args, cases, device, admission)


if __name__ == "__main__":
    main()

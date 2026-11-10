#!/usr/bin/env python3
"""Generate raw glyph images from a standalone compact image-head checkpoint."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.visual_lm.compact_visual_span_weights import load_compact_visual_span_model
from ilm.visual_lm.visual_cell_data import pack_visual_cells


MODEL_SOURCES = (
    "ilm/visual_lm/compact_visual_span_model.py",
    "ilm/visual_lm/compact_visual_span_weights.py",
)


def read_image(path, glyph_count=None):
    """Read a gapless 32×32 tile grid in row-major order."""
    with Image.open(path) as image:
        width, height = image.size
        if image.format != "PNG" or image.mode not in {"1", "L", "RGB"} or width % 32 or height % 32:
            raise ValueError("Supply a grayscale or RGB PNG in a gapless 32×32 tile grid")
        columns, rows = width // 32, height // 32
        capacity = columns * rows
        count = capacity if glyph_count is None else glyph_count
        if not 0 < count <= capacity or capacity - count >= columns:
            raise ValueError("Glyph count must occupy every row, with optional trailing padding")
        ink = (np.asarray(image.convert("L")) < 128).astype(np.uint8)
    tiles = ink.reshape(rows, 32, columns, 32).transpose(0, 2, 1, 3).reshape(capacity, 1, 32, 32)
    if tiles[count:].any():
        raise ValueError("Unused tiles at the end of the grid must be blank")
    return torch.from_numpy(tiles[:count].copy())


def save_grid(ink, path):
    if len(ink):
        pack_visual_cells(ink, columns=min(32, len(ink)), gutter=0).save(path)
    else:
        Image.new("L", (32, 32), 255).save(path)


def input_cases(args):
    if args.image is not None:
        return [("answer", args.image, args.glyph_count, sha(args.image))]
    cases = json.loads(args.manifest.read_text())["inputs"]
    selected = []
    identifiers = set()
    for row in cases:
        key, name = row["id"], row["image"]
        if not re.fullmatch(r"[a-z0-9_]+", key) or key in identifiers or name != key + ".png":
            raise ValueError("Invalid or repeated inference case identifier")
        identifiers.add(key)
        if args.normal_only and row["condition"] != "normal":
            continue
        path = args.manifest.parent / name
        if sha(path) != row["image_sha256"]:
            raise ValueError(f"Input image changed: {name}")
        selected.append((key, path, row["glyphs"], row["image_sha256"]))
    if not selected:
        raise ValueError("No input images selected")
    return selected


def load_checkpoint(path, local, codec, device):
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if set(snapshot) != {"identity", "updates", "image_heads"} or snapshot["updates"] < 0:
        raise ValueError("Supply a deployable image_heads_step checkpoint")
    identity = snapshot["identity"]
    if sha(codec) != identity["codec_sha256"] or any(
        sha(ROOT / name) != identity["code_sha256"][name] for name in MODEL_SOURCES
    ):
        raise ValueError("Checkpoint model source or glyph codec differs")
    model, provenance = load_compact_visual_span_model(local, codec, device)
    model.encoder.load_state_dict(snapshot["image_heads"]["encoder"], strict=True)
    model.writer.load_state_dict(snapshot["image_heads"]["writer"], strict=True)
    provenance.pop("visual_span_heads_initialized_for_training")
    provenance["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return model.requires_grad_(False).eval(), snapshot["updates"], provenance


def check_output(result, prompt_length):
    if sum(result["prompt_span_lengths"]) != prompt_length or result["stopped"] == result["capped"]:
        raise ValueError("Generation returned inconsistent prompt or termination metadata")
    ink = result["ink"]
    if ink.dtype != torch.uint8 or ink.shape[1:] != (1, 32, 32) or bool((ink > 1).any()):
        raise ValueError("Generated ink must retain binary 32×32 tiles")
    position = 0
    for span in result["emitted_pixel_feedback"]:
        end = position + span["glyphs"]
        digest = hashlib.sha256(ink[position:end].numpy().tobytes()).hexdigest()
        if end > len(ink) or digest != span["ink_sha256"]:
            raise ValueError("Returned pixels differ from the generation feedback trace")
        position = end
    if position != len(ink):
        raise ValueError("An emitted span has no feedback trace")


def generate_case(model, device, args, case, output):
    key, path, glyph_count, digest = case
    final = output / key
    if final.exists():
        prior = json.loads((final / "report.json").read_text())
        if (prior["input_sha256"] != digest or prior["declared_glyph_count"] != glyph_count
                or sha(final / "ink.npz") != prior["raw_sha256"]
                or sha(final / "answer.png") != prior["image_sha256"]):
            raise ValueError(f"Saved inference output changed: {key}")
        return prior
    prompt = read_image(path, glyph_count)
    begin = time.perf_counter()
    result = model.generate(prompt.to(device), maximum_glyphs=args.maximum_glyphs,
                            flow_steps=args.flow_steps, seed=args.seed)
    elapsed = time.perf_counter() - begin
    check_output(result, len(prompt))
    ink = result.pop("ink")
    pending = output / f".{key}.{time.time_ns()}.partial"
    pending.mkdir()
    np.savez_compressed(pending / "ink.npz", ink=ink.numpy())
    save_grid(ink, pending / "answer.png")
    report = {"id": key, "input_sha256": digest, "input_glyphs": len(prompt), "declared_glyph_count": glyph_count,
              "emitted_glyphs": len(ink), "elapsed_seconds": elapsed,
              "raw_sha256": sha(pending / "ink.npz"), "image_sha256": sha(pending / "answer.png"), **result}
    save_json(pending / "report.json", report)
    pending.rename(final)
    return report


def run(args, cases, device):
    identity = {
        "checkpoint_sha256": sha(args.checkpoint),
        "inputs": [{"id": key, "glyphs": count, "sha256": digest} for key, _, count, digest in cases],
        "seed": args.seed, "flow_steps": args.flow_steps, "maximum_glyphs": args.maximum_glyphs,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": {name: sha(ROOT / name) for name in (*MODEL_SOURCES, "scripts/infer_compact_visual_span.py")},
    }
    output = args.output
    request = output / "request.json"
    if request.exists():
        if json.loads(request.read_text()) != identity:
            raise ValueError("Resume requires the original inference checkpoint, inputs and configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(request, identity)
    begin = time.perf_counter()
    model, updates, provenance = load_checkpoint(args.checkpoint, args.local, args.codec, device)
    load_seconds = time.perf_counter() - begin
    rows = []
    for case in cases:
        row = generate_case(model, device, args, case, output)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("id", "emitted_glyphs", "stopped", "elapsed_seconds")}), flush=True)
    report = {"request_sha256": sha(request), "optimizer_updates": updates, "provenance": provenance,
              "checkpoint_load_seconds": load_seconds, "total_generation_seconds": sum(r["elapsed_seconds"] for r in rows),
              "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
              "reference_inputs": False, "output_repaired": False, "records": rows}
    save_json(output / "report.json", report)


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
        parser.error("--glyph-count applies to one image, and --normal-only applies to a manifest")
    if min(args.maximum_glyphs, args.flow_steps) < 1:
        parser.error("Generation budgets must be positive")
    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for CPU inference")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cases = input_cases(args)
    torch.set_num_threads(2)
    runtime = gpu_zero() if args.device == "cuda" else nullcontext(torch.device("cpu"))
    with runtime as device:
        run(args, cases, device)


if __name__ == "__main__":
    main()

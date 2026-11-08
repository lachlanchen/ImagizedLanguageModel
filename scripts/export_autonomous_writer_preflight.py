#!/usr/bin/env python3
"""Authenticate full-trajectory execution without claiming writing quality."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import save_json, sha, tensor_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run, out = args.run, args.output
    report = json.loads((run / "report.json").read_text())
    updates = [json.loads(s) for s in (run / "updates.jsonl").read_text().splitlines() if s]
    if report["stage"] != "preflight" or report["updates"] != 2 or len(updates) != 2:
        raise ValueError("complete original two-update execution required")
    for row in updates:
        grads = row["trajectory_prediction_gradients"]
        if set(grads) != {"0", "24", "49"} or not all(g["finite_nonzero"] for g in grads.values()) or row["input_gradient_norm"] <= 0:
            raise ValueError("complete early/middle/late and input gradient proof required")
    if report["peak_allocated_vram_gib"] >= 22 or report["changed_adapter_tensors"] != 206:
        raise ValueError("bounded real optimization proof differs")
    out.mkdir(parents=True, exist_ok=False)
    proof, files = {}, {}
    for language in ("zh", "en"):
        original = json.loads((run / f"equivalence_{language}.json").read_text())
        if not all(original[k] for k in ("all_step_tensors_exact", "final_pixels_exact", "retained_native_pixels_exact")):
            raise ValueError("all original source comparisons must pass")
        trace_path = run / f"trace_{language}.pt"
        trace = torch.load(trace_path, map_location="cpu", weights_only=True)
        if len(trace["native"]) != 50 or len(trace["custom"]) != 50:
            raise ValueError("full 50-step trace required")
        steps = []
        for i, (native, actual) in enumerate(zip(trace["native"], trace["custom"])):
            if not torch.equal(native, actual):
                raise ValueError("actual native and differentiable trajectory differ")
            steps.append(dict(step=i, shape=list(native.shape), dtype=str(native.dtype),
                native_sha256=tensor_sha(native), differentiable_sha256=tensor_sha(actual)))
        proof[language] = dict(original, steps=steps, retained_trace_sha256=sha(trace_path))
        for phase in ("before", "after"):
            folder = run / (phase + "_" + language)
            png, pixels = folder / "answer.png", folder / "pixels.pt"
            value = torch.load(pixels, map_location="cpu", weights_only=True)
            raw = (value[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
            if not np.array_equal(raw, np.array(Image.open(png).convert("RGB"))):
                raise ValueError("preflight PNG differs from actual decoder pixels")
            name = phase + "_" + language + ".png"; shutil.copyfile(png, out / name)
            files[name] = dict(png_sha256=sha(png), retained_tensor_sha256=sha(pixels),
                png_equals_quantized_neural_tensor=True)
    for name in ("report.json", "updates.jsonl"):
        shutil.copyfile(run / name, out / ("source_" + name))
    # Check the actual autograd/checkpointed training forward before update 1,
    # in addition to the no-grad trajectory comparison above.
    original = torch.load(run / "before_zh/pixels.pt", map_location="cpu", weights_only=True)
    differentiable = torch.load(run / "update_001_before/pixels.pt", map_location="cpu", weights_only=True)
    if not torch.equal(original, differentiable):
        raise ValueError("actual differentiable training forward differs from native pixels")
    save_json(out / "result.json", dict(trajectory_proof=proof, actual_pages=files,
        source_report_sha256=sha(run / "report.json"), source_journal_sha256=sha(run / "updates.jsonl"),
        cpu_rescored_all_100_native_step_tensors=True, output_repair=False,
        first_actual_differentiable_training_forward_native_pixel_exact=True,
        first_actual_differentiable_training_pixels_sha256=sha(run / "update_001_before/pixels.pt"),
        independent_neural_replay=False, quality_qualification=False, general_ilm_goal_complete=False))
    print("Authenticated 100 native step tensors, both exact final images and four unrepaired execution pages.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read supplied binary writing PNGs and emit learned source-pixel answers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.contextual_pixel_encoder import ContextualPixelReader
from scripts.cache_semantic_images_v72 import ROOT, CONTROLS, CONTROL_SHA, sha, atomic_json
from scripts.probe_native_unified_source import gpu_slot
from scripts.train_clean_corpus_v75 import setup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.run / "training/report.json"
    report = json.loads(report_path.read_text())
    if report["updates"] != 3200:
        raise ValueError("completed training required")
    with gpu_slot(), torch.inference_mode():
        device = setup(); began = time.monotonic(); local = ROOT.parent / "LocalLLM/.local"
        reader = ContextualPixelReader(checkpoint=args.run / "training/checkpoint.pt",
            checkpoint_sha256=report["checkpoint_sha256"], controls=CONTROLS, controls_sha256=CONTROL_SHA,
            ollama_root=local / "models/ollama", gguf_source=local / "tools/llama.cpp/gguf-py", device=device)
        load_seconds = time.monotonic() - began
        results = []; began = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        for i, path in enumerate(args.images):
            image = np.array(Image.open(path).convert("L"), copy=True)
            answer, metadata = reader.generate(image)
            dest = args.output / f"{i:03d}.png"; Image.fromarray(answer).save(dest)
            results.append(dict(input=str(path), input_sha256=sha(path), output=dest.name,
                output_sha256=sha(dest), **metadata))
        atomic_json(args.output / "report.json", dict(outputs=results, setup_seconds=load_seconds,
            inference_seconds=time.monotonic() - began, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            full_runtime_parameters=reader.parameter_count(), training_report_sha256=sha(report_path),
            implementation={n: sha(ROOT / n) for n in ("scripts/infer_contextual_pixel_spans.py",
                "ilm/visual_lm/contextual_pixel_encoder.py", "ilm/visual_lm/contextual_pixel_spans.py")},
            quality_claim=False, free_generation_demonstrated=False, general_ilm_goal_complete=False))
        print("Saved", len(results), "unmodified answer images.", flush=True)


if __name__ == "__main__":
    main()

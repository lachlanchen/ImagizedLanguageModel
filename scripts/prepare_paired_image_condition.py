#!/usr/bin/env python3
"""Compile the fixed generic task condition without experimental predecessors.

This one-time offline step uses the existing verified shared Qwen3 encoder.
Every question subsequently receives the same numeric tensors. The deployed
image generator never loads the encoder or tokenizer.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import CONDITIONER, ROOT, gpu_zero, save_json, sha
from scripts.image_edit_language_baseline import prepare_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioner", type=Path, default=CONDITIONER,
                        help="verified shared Z-Image Qwen3-4B source, with text_encoder and tokenizer subfolders")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {name: sha(ROOT / name) for name in
        ("scripts/prepare_paired_image_condition.py", "scripts/image_edit_language_baseline.py",
         "ilm/image_edit/baseline.py")}
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    started = time.perf_counter()
    try:
        with gpu_zero() as device:
            import torch
            # This pure compiler verifies its actual encoder/tokenizer bytes.
            # The baseline CLI's predecessor gate is never invoked.
            result = prepare_condition(args.output, args.conditioner, device)
            if sources != {name: sha(ROOT / name) for name in sources}:
                raise ValueError("preserve the actual compiler sources while running")
            report = dict(stage="constant_condition", result=result, sources_sha256=sources,
                packages={name: version(name) for name in ("torch", "transformers", "diffusers")},
                elapsed_seconds=time.perf_counter() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                experimental_predecessors_required=False, question_specific_information=False,
                external_pretraining=True, language_quality_qualified=False,
                general_ilm_goal_complete=False)
            save_json(args.output / "report.json", report)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: report[k] for k in
                ("elapsed_seconds", "peak_allocated_vram_gib", "experimental_predecessors_required")}, indent=2))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None,
            exception=type(error).__name__, reason=str(error)))
        raise


if __name__ == "__main__":
    main()

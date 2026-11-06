#!/usr/bin/env python3
"""Generate a complete input-image-only cohort with one model load.

The JSON input index accepts only id, input, input_sha256 and seed. It cannot
carry targets, reference text or per-question text embeddings.
"""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import condition_cache, gpu_zero, save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--condition", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    args = p.parse_args()
    rows = json.loads(args.inputs.read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique nonempty input-image index required")
    for row in rows:
        if set(row) != {"id", "input", "input_sha256", "seed"}:
            raise ValueError("input-only index must not carry answers or varying text conditions")
        if sha(args.inputs.parent / row["input"]) != row["input_sha256"]:
            raise ValueError("input PNG changed: " + row["id"])
    import torch
    from ilm.image_edit.paired import (add_adapters, answer, implementation, load_pipeline,
        restore_adapters, save_answer, verify_source)
    source = verify_source(args.snapshot)
    fixed = condition_cache(args.condition)
    code = implementation()
    runner_sha = sha(__file__)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    tick = time.perf_counter()
    try:
        with gpu_zero() as device:
            pipe = load_pipeline(args.snapshot, device)
            if args.checkpoint:
                saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                metadata = saved["metadata"]
                if metadata["source"] != source or metadata["condition_sha256"] != sha(args.condition):
                    raise ValueError("checkpoint source/constant condition differs")
                add_adapters(pipe.transformer, metadata["rank"])
                restore_adapters(pipe.transformer, saved["adapters"])
                # No optimizer or training data is restored for inference.
                del saved
            pipe.transformer.eval()
            outputs = []
            for index, row in enumerate(rows):
                raw = answer(pipe, args.inputs.parent / row["input"], fixed,
                    seed=row["seed"], width=args.width, height=args.height,
                    steps=args.steps, guidance=args.guidance)
                name = f"{index:03}"
                result = save_answer(pipe, raw, args.output / name)
                outputs.append(dict(id=row["id"], path=name + "/answer.png",
                    sha256=result["png_sha256"], result=result, seed=row["seed"],
                    input_sha256=row["input_sha256"]))
                save_json(args.output / "status.json", dict(stage="generating", pid=os.getpid(),
                    completed=len(outputs), total=len(rows)))
                print(f"Generated {len(outputs)}/{len(rows)}: {row['id']}", flush=True)
            if implementation() != code or sha(__file__) != runner_sha:
                raise ValueError("preserve the actual runner while generation is live")
            save_json(args.output / "outputs.json", outputs)
            save_json(args.output / "report.json", dict(source=source, implementation=code,
                runner_sha256=runner_sha, inputs_sha256=sha(args.inputs), complete_generations=len(outputs),
                condition_sha256=sha(args.condition), checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,
                transformer_parameters=sum(p.numel() for p in pipe.transformer.parameters()),
                codec_parameters=sum(p.numel() for p in pipe.vae.parameters()),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                elapsed_seconds=time.perf_counter() - tick, steps=args.steps, guidance=args.guidance,
                output_canvas=[args.width, args.height], training_manifest_loaded=False,
                reference_answer_used=False, question_specific_text_condition=False, output_repair=False,
                language_quality_qualified=False, general_ilm_goal_complete=False))
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None,
            exception=type(error).__name__, reason=str(error)))
        raise


if __name__ == "__main__":
    main()

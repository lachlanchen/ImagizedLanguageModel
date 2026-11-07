#!/usr/bin/env python3
"""Measure the pinned unadapted native reader with the same English controls."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from ilm.image_edit.baseline import save_json
from ilm.native_unified.showo2_source import load, sha, image_latent, reading_prefix, MODEL_SHA256
from ilm.native_unified.reading import cached_decisions
from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--inputs", type=Path, default=Path("artifacts/native_english_transfer/prepared_v2/inputs.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", pid=os.getpid()))
    try:
        torch.set_num_threads(8)
        with gpu_slot(), torch.inference_mode():
            b = load(args.assets, dtype=torch.bfloat16)
            b.vae = None
            torch.cuda.empty_cache()
            b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
            rows = json.loads(args.inputs.read_text())
            if len(rows) != 80 or any(k in row for row in rows for k in ("expected", "answer", "references")):
                raise ValueError("complete input-only English cohort required")
            outputs = []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for row in rows:
                path = args.inputs.parent / row["input"]
                if sha(path) != row["input_sha256"]:
                    raise ValueError("input raster changed")
                z, _ = image_latent(b, path, row["seed"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                    ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
                result = {key: row[key] for key in ("id", "pair", "condition", "input_sha256", "seed", "readout_seed")}
                result.update(ids=ids, text=b.tokenizer.decode(ids, skip_special_tokens=True),
                    stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id), student_answer=False)
                outputs.append(result)
                save_json(args.output / "outputs.json", outputs)
                save_json(args.output / "status.json", dict(state="running", pid=os.getpid(), complete_calls=len(outputs), planned_calls=80))
                print(row["id"], repr(result["text"]), flush=True)
            save_json(args.output / "report.json", dict(cases=80, inputs_sha256=sha(args.inputs), source_model_sha256=MODEL_SHA256,
                source_counts=b.counts, optimizer_updates=0, adapters_added=0, body_dtype="bfloat16", codec_dtype="float32",
                cap=96, instruction=ANSWER_INSTRUCTION, elapsed_seconds=time.monotonic() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                implementation_sha256=sha(__file__), auxiliary_unicode_head=True, teacher_answers_loaded=False,
                language_quality_qualified=False, general_ilm_goal_complete=False))
        save_json(args.output / "status.json", dict(state="completed", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read a prompt PNG and save a complete unassisted image answer."""

import argparse
import fcntl
from pathlib import Path

import torch

from ilm.visual_lm.clean_image_actor import load_clean_image_actor
from ilm.visual_lm.codec_image_interface import prepare_image_prompt, save_prepared_prompt
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, atomic_json
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_semantic_ilm_v72 import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=("page", "strip"), default="page")
    parser.add_argument("--seed", type=int, default=20267420)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("choose a fresh output directory")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup()
        prepared = prepare_image_prompt(args.prompt, layout=args.layout)
        save_prepared_prompt(prepared, args.out/"input")
        local = ROOT.parent/"LocalLLM/.local"
        actor, provenance = load_clean_image_actor(checkpoint=args.checkpoint, controls=CONTROLS,
            codec=CODEC, ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py", device=device)
        result = actor.generate(prepared.pixels, seed=args.seed)
        saved = save_image_answer(result, args.out)
        atomic_json(args.out/"report.json", dict(provenance=provenance, seed=args.seed,
            **saved, peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False, general_ilm_goal_complete=False))


if __name__ == "__main__":
    main()

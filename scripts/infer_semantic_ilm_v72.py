#!/usr/bin/env python3
"""Generate an autonomous image answer to an image prompt, without text decoding."""

import argparse
import fcntl
from pathlib import Path

import torch

from ilm.visual_lm.codec_image_interface import prepare_image_prompt, save_prepared_prompt
from ilm.visual_lm.semantic_image_actor import load_semantic_image_actor
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, atomic_json, sha
from scripts.train_semantic_ilm_v72 import setup


def save_image_answer(result,out):
    from PIL import Image
    out.mkdir(parents=True,exist_ok=True)
    raw = out/"answer.pt"
    temporary = raw.with_suffix(".partial")
    torch.save(result,temporary)
    temporary.replace(raw)
    if len(result["images"]):
        strip = torch.cat(list(result["images"]),dim=-1)
        pixels = ((strip[0]+1)*127.5).round().to(torch.uint8).numpy()
        Image.fromarray(pixels).save(out/"answer.png")
    elif (out/"answer.png").exists():
        raise ValueError("stale output PNG exists for an empty answer")
    return dict(raw_sha256=sha(raw),blocks=len(result["images"]),stopped=result["stopped"],
        capped=result["capped"],answer_png_sha256=sha(out/"answer.png") if len(result["images"]) else None,
        feedback_from_emitted_pixels=result["feedback_from_emitted_pixels"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--layout",choices=("page","strip"),default="page")
    parser.add_argument("--seed",type=int,default=20267204)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("choose a fresh output directory to preserve previous answers")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device = setup()
        prepared = prepare_image_prompt(args.prompt,layout=args.layout)
        save_prepared_prompt(prepared,args.output/"input")
        local = ROOT.parent/"LocalLLM/.local"
        actor,provenance = load_semantic_image_actor(checkpoint=args.checkpoint,controls=CONTROLS,
            codec=CODEC,ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        result = actor.generate(prepared.pixels,seed=args.seed)
        output = save_image_answer(result,args.output)
        atomic_json(args.output/"report.json",dict(provenance=provenance,seed=args.seed,**output,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False))


if __name__=="__main__":
    main()

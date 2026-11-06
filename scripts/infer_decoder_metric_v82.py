#!/usr/bin/env python3
"""Generate raw answer images from a completed research checkpoint."""
import argparse
import fcntl
from pathlib import Path
import time

import torch

from ilm.visual_lm.codec_image_interface import prepare_image_prompt, save_prepared_prompt
from ilm.visual_lm.metric_image_actor import completed_metric_checkpoint, load_metric_image_actor
from ilm.visual_lm.native_prompt_interface import prepare_text_prompt, prepare_instruction_image, validate_native_answer
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, atomic_json, sha
from scripts.infer_native_chinese_v80 import ensure_host_renderer
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_clean_corpus_v75 import setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt",type=Path,help="one input image")
    source.add_argument("--text",help="optional host rendering of a typed prompt")
    source.add_argument("--text-file",type=Path,help="UTF-8 prompt with explicit line breaks")
    parser.add_argument("--instruction",help="visible instruction above an input image")
    parser.add_argument("--layout",choices=("page","strip"))
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20268250)
    parser.add_argument("--maximum-blocks",type=int,choices=range(1,129),default=128,metavar="1..128")
    parser.add_argument("--prepare-only",action="store_true",help="save actual input pixels without loading any checkpoint or CUDA")
    args=parser.parse_args()
    if (args.instruction is not None or args.layout is not None) and args.prompt is None:
        parser.error("--instruction and --layout apply only to input images")
    if args.output.exists():parser.error("use a fresh output directory to preserve prior inputs and answers")
    if not args.prepare_only:
        completed_metric_checkpoint(args.checkpoint)
    if args.text is not None or args.text_file is not None or args.instruction is not None:
        ensure_host_renderer()
    if args.prompt is not None:
        prepared=prepare_image_prompt(args.prompt,layout=args.layout or "page")
        if args.instruction is not None:prepared=prepare_instruction_image(args.instruction,prepared)
    else:
        text=args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        prepared=prepare_text_prompt(text)
        if args.text_file is not None:prepared.receipt["source_utf8_file_sha256"]=sha(args.text_file)
    args.output.mkdir(parents=True)
    inputs=save_prepared_prompt(prepared,args.output/"input")
    if args.prepare_only:
        atomic_json(args.output/"report.json",dict(input=inputs,preparation_only=True,
            generation_performed=False,checkpoint_loaded=False,cuda_initialized=torch.cuda.is_initialized(),language_quality_claim=False))
        print(f"Prepared input pixels: {args.output/'input/prompt.png'}",flush=True);return
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup();begun=time.perf_counter();local=ROOT.parent/"LocalLLM/.local"
        actor,provenance=load_metric_image_actor(checkpoint=args.checkpoint,controls=CONTROLS,codec=CODEC,
            ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        loaded=time.perf_counter()
        result=actor.generate(prepared.pixels,seed=args.seed,maximum_blocks=args.maximum_blocks)
        validate_native_answer(result,args.maximum_blocks)
        output=save_image_answer(result,args.output/"output")
        atomic_json(args.output/"report.json",dict(input=inputs,output=output,provenance=provenance,
            seed=args.seed,maximum_blocks=args.maximum_blocks,flow_steps=32,checkpoint_sha256=sha(args.checkpoint),
            model_load_seconds=loaded-begun,generation_seconds=time.perf_counter()-loaded,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            generation_performed=True,output_repair=False,reference_images_used=False,language_quality_claim=False))
        print(f"Generated {output['blocks']} raw blocks; stopped={output['stopped']}; output: {args.output/'output'}",flush=True)


if __name__=="__main__":main()

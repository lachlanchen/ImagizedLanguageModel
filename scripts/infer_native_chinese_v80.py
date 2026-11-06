#!/usr/bin/env python3
"""Run the trained native-context image actor on typed, image or mixed prompts."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

import torch

from ilm.visual_lm.codec_image_interface import prepare_image_prompt, save_prepared_prompt
from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
from ilm.visual_lm.native_prompt_interface import prepare_text_prompt, prepare_instruction_image, validate_native_answer
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, atomic_json, sha
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_clean_corpus_v75 import setup


def ready_checkpoint(path):
    # Metadata inspection is CPU-only. mmap avoids materializing the live run's
    # Adam tensors merely to reject an incomplete checkpoint before GPU setup.
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    identity = saved["identity"]
    if (identity.get("purpose")!="native_chinese_full_context" or identity.get("planned_updates")!=40000
            or saved.get("update")!=40000 or not saved.get("training_complete",False)):
        raise ValueError("native training is incomplete: a completed 40,000-update checkpoint is required")


def ensure_host_renderer():
    """Start the text adapter with its verified native libraries before loading it."""
    if "libglib-2.0.so" in os.environ.get("LD_PRELOAD", ""):
        return
    libraries=[Path("/usr/lib/x86_64-linux-gnu")/name for name in (
        "libglib-2.0.so.0", "libgobject-2.0.so.0", "libgio-2.0.so.0")]
    if not all(p.is_file() for p in libraries):
        raise RuntimeError("the host text adapter requires the documented system GLib/GObject/GIO libraries; image-only input does not")
    if torch.cuda.is_initialized():
        raise RuntimeError("initialize the optional host renderer before CUDA")
    preload=":".join(map(str,libraries))
    if os.environ.get("LD_PRELOAD"):
        preload+=":"+os.environ["LD_PRELOAD"]
    os.execve(sys.executable,[sys.executable,*sys.argv],dict(os.environ,LD_PRELOAD=preload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", type=Path, help="one explicit image file")
    source.add_argument("--text", help="typed prompt rendered by the optional host adapter")
    source.add_argument("--text-file", type=Path, help="UTF-8 typed prompt, retaining explicit line breaks")
    parser.add_argument("--instruction", help="visible instruction above --prompt image pixels")
    parser.add_argument("--layout", choices=("page","strip"), help="image layout; default page")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"artifacts/native_chinese_v80_20261106/training/checkpoint_latest.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20268005)
    parser.add_argument("--maximum-blocks", type=int, choices=range(1,129), default=128, metavar="1..128")
    parser.add_argument("--prepare-only", action="store_true", help="save input pixels without loading a model or allocating CUDA")
    args = parser.parse_args()
    if (args.instruction is not None or args.layout is not None) and args.prompt is None:
        parser.error("--instruction and --layout apply only to --prompt images")
    if args.output.exists():
        parser.error("choose a fresh output directory to preserve previous inputs and answers")
    if not args.prepare_only:
        if not args.checkpoint.is_file():
            parser.error("completed native checkpoint is missing; --prepare-only can prepare input pixels now")
        ready_checkpoint(args.checkpoint)
    if args.text is not None or args.text_file is not None or args.instruction is not None:
        ensure_host_renderer()
    if args.prompt:
        prepared = prepare_image_prompt(args.prompt, layout=args.layout or "page")
        if args.instruction is not None:
            prepared = prepare_instruction_image(args.instruction, prepared)
    else:
        source_bytes = args.text_file.read_bytes() if args.text_file is not None else None
        text = args.text if args.text is not None else source_bytes.decode("utf-8")
        prepared = prepare_text_prompt(text)
        if source_bytes is not None:
            import hashlib
            prepared.receipt["source_utf8_file_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.mkdir(parents=True)
    inputs = save_prepared_prompt(prepared, args.output/"input")
    if args.prepare_only:
        atomic_json(args.output/"report.json", dict(input=inputs, preparation_only=True,
            generation_performed=False, checkpoint_loaded=False, cuda_initialized=torch.cuda.is_initialized(),
            language_quality_claim=False))
        print(f"Prepared input pixels: {args.output/'input/prompt.png'}", flush=True)
        return
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device = setup(); begun = time.perf_counter()
        local = ROOT.parent/"LocalLLM/.local"
        actor, provenance = load_trained_native_image_actor(checkpoint=args.checkpoint, controls=CONTROLS,
            codec=CODEC, ollama_root=local/"models/ollama", gguf_source=local/"tools/llama.cpp/gguf-py", device=device)
        loaded = time.perf_counter()
        result = actor.generate(prepared.pixels, seed=args.seed, maximum_blocks=args.maximum_blocks)
        validate_native_answer(result, args.maximum_blocks)
        output = save_image_answer(result, args.output/"output")
        atomic_json(args.output/"report.json", dict(input=inputs, output=output, provenance=provenance,
            seed=args.seed, maximum_blocks=args.maximum_blocks, flow_steps=32,
            checkpoint_sha256=sha(args.checkpoint), generation_performed=True,
            model_load_seconds=loaded-begun, generation_seconds=time.perf_counter()-loaded,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            output_repair=False, reference_images_used=False, language_quality_claim=False))
        print(f"Generated {output['blocks']} blocks; stopped={output['stopped']}; output: {args.output/'output'}", flush=True)


if __name__=="__main__": main()

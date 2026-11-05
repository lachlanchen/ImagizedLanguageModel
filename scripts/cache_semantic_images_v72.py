#!/usr/bin/env python3
"""Cache prompt-only visual semantics and complete image-derived writing codes."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.codec_block_writer import CODEC_SHA, CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.grounded_image_weights import MODEL_SHA, load_grounded_image_actor
from ilm.visual_lm.semantic_image_memory import SemanticImageMemory, native_budget_image

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/general_instructions/complete_lines_train_20261105"
TRAIN_SHA = "0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"
CONTROLS = ROOT / "artifacts/native_visual_transfer_v71_20261105/attempt01/structural_vectors.pt"
CONTROL_SHA = "e53cb8d13892aa5de283d55138101ca45527356f73b40a9a965cee77c67167c1"
CODEC = ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
SOURCES = (
    "scripts/cache_semantic_images_v72.py", "ilm/visual_lm/semantic_image_memory.py",
    "ilm/visual_lm/image_context_interface.py", "ilm/visual_lm/grounded_image_actor.py",
    "ilm/visual_lm/grounded_image_weights.py", "ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/causal_visual_prefix.py",
    "references/semantic_ilm_v72_protocol.md",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def committed_identity(names):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {name: sha(ROOT/name) for name in names}
    for name, digest in sources.items():
        raw = subprocess.check_output(["git", "show", revision+":"+name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("commit implementation before execution: " + name)
    return dict(source_commit=revision, sources=sources)


def pixels_sha(image):
    return hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()


def recipe(data):
    if sha(CONTROLS) != CONTROL_SHA:
        raise ValueError("numeric structural vectors changed")
    names = list(data.names)
    random.Random(20267201).shuffle(names)
    return dict(**committed_identity(SOURCES), train_report_sha256=TRAIN_SHA,
        source_model_sha256=MODEL_SHA, codec_sha256=CODEC_SHA,
        structural_vectors_sha256=CONTROL_SHA, cache_seed=20267201,
        observer="factor32 area [1048576,4194304], RGB Pillow bilinear, then L",
        memory_dtype="bfloat16", code_dtype="float32", memory="all final task-frame states",
        names=names, total=len(names), prompt_only_memory=True,
        symbolic_helper_loaded=False, target_termination_blocks_removed=2)


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("physical GPU 0 must be explicitly selected")
    check_host_memory()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_memory() > 256:
            raise RuntimeError("GPU 0 is occupied; do not overlap model jobs")
        data = CompleteWritingImageData(TRAIN, purpose="train", expected_report_sha256=TRAIN_SHA)
        config = recipe(data)
        if (out/"recipe.json").exists():
            original = json.loads((out/"recipe.json").read_text())
            # Unrelated later commits must not invalidate a pinned cache.
            if {k:v for k,v in original.items() if k!="source_commit"} != {k:v for k,v in config.items() if k!="source_commit"}:
                raise ValueError("resume recipe changed")
            config = original
        else:
            atomic_json(out/"recipe.json", config)
        recipe_sha = sha(out/"recipe.json")
        records = out/"records"
        records.mkdir(exist_ok=True)
        previous = json.loads((out/"report.json").read_text()) if (out/"report.json").exists() else {}
        files = {}
        for position, name in enumerate(config["names"]):
            relative = f"records/{position:05}.pt"
            path = out/relative
            if path.exists():
                digest = sha(path)
                if relative in previous.get("files_sha256", {}) and previous["files_sha256"][relative] != digest:
                    raise ValueError("completed cache payload changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (saved["recipe_sha256"] != recipe_sha or saved["name"] != name
                        or saved["source_sha256"] != data.report["files_sha256"][name]):
                    raise ValueError("cache source identity changed")
                files[relative] = digest
        limit = len(data) if args.limit is None else min(args.limit, len(data))
        if limit < 1:
            raise ValueError("positive cache limit required")
        begun = time.perf_counter()
        accumulated = previous.get("active_elapsed_seconds", 0.)

        def report(stage):
            value = dict(recipe_sha256=recipe_sha, total=len(data), completed=len(files),
                complete=len(files)==len(data), files_sha256=files,
                active_elapsed_seconds=accumulated+time.perf_counter()-begun,
                stage=stage, pid=os.getpid() if stage=="caching" else None)
            if torch.cuda.is_initialized():
                value["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated()/1024**3
            atomic_json(out/"report.json", value)

        if all(f"records/{i:05}.pt" in files for i in range(limit)):
            report("complete" if len(files)==len(data) else "preflight_cache_ready")
            return
        torch.set_num_threads(4)
        torch.manual_seed(20267201)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        report("loading_reader")
        local = ROOT.parent/"LocalLLM/.local"
        actor, provenance = load_grounded_image_actor(ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py", device=torch.device("cuda:0"))
        actor.writer = torch.nn.Identity()
        actor.requires_grad_(False).eval()
        if any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith(".up")):
            raise ValueError("source contextual adapter output must be zero")
        controls = torch.load(CONTROLS, map_location="cpu", weights_only=True)
        observer = SemanticImageMemory(actor, **controls).requires_grad_(False).eval()
        writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cuda:0")))
        atomic_json(out/"provenance.json", provenance)
        indices = {name:i for i,name in enumerate(data.names)}
        report("caching")
        with torch.inference_mode():
            for position, name in enumerate(config["names"][:limit]):
                relative = f"records/{position:05}.pt"
                if relative in files:
                    continue
                index = indices[name]
                prompt = fold_writing_prompt(data.prompt(index)).pixels
                # Compute this before loading the answer. The reader sees only
                # physical prompt pixels; labels never share its forward pass.
                memory, metadata = observer(prompt)
                memory = memory[0].cpu().clone()
                target = data.target(index)[..., :-128]
                blocks = torch.cat(target.split(64, dim=-1))
                codes = writer.encode_targets(blocks).cpu().clone()
                if (memory.dtype != torch.bfloat16 or memory.ndim != 2
                        or memory.shape[-1] != 4096 or codes.shape != (len(blocks),2,768)
                        or not torch.isfinite(memory).all() or not torch.isfinite(codes).all()):
                    raise ValueError("invalid continuous cache values")
                payload = dict(recipe_sha256=recipe_sha, name=name,
                    source_sha256=data.report["files_sha256"][name], memory=memory, codes=codes,
                    prompt_sha256=pixels_sha(prompt), transformed_prompt_sha256=pixels_sha(native_budget_image(prompt)),
                    target_sha256=pixels_sha(target), prompt_shape=list(prompt.shape), metadata=metadata)
                path = out/relative
                temporary = path.with_suffix(".partial")
                torch.save(payload, temporary)
                temporary.replace(path)
                files[relative] = sha(path)
                report("caching")
                print(json.dumps(dict(completed=len(files), total=len(data), source=name,
                    answer_blocks=len(blocks), elapsed_seconds=accumulated+time.perf_counter()-begun)), flush=True)
        report("complete" if len(files)==len(data) else "preflight_cache_ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.output.exists():
            atomic_json(args.output/"failure.json", dict(error=repr(error), pid=os.getpid(), time=time.time()))
        raise

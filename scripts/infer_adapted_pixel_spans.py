#!/usr/bin/env python3
"""Deploy a final span model on supplied PNGs without training-data files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.adapted_pixel_runtime import AdaptedPixelRuntime
from scripts.cache_semantic_images_v72 import ROOT, CONTROLS, CONTROL_SHA, atomic_json, sha
from scripts.probe_native_unified_source import gpu_slot
from scripts.train_clean_corpus_v75 import setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--images",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    # Validate user image format before occupying the model card.
    images=[]
    for path in args.images:
        raw=path.read_bytes()
        image=np.array(Image.open(io.BytesIO(raw)).convert("L"),copy=True)
        if image.ndim!=2 or image.shape[0]!=32 or not np.isin(image,[0,255]).all():
            raise ValueError("supplied images must be complete binary 32-pixel-high writing strips")
        images.append((path,image,hashlib.sha256(raw).hexdigest()))
    with gpu_slot():
        device=setup();args.output.mkdir(parents=True,exist_ok=False)
        local=ROOT.parent/"LocalLLM/.local";began=time.monotonic()
        model=AdaptedPixelRuntime(checkpoint=args.checkpoint,checkpoint_sha256=args.checkpoint_sha256,
            controls=CONTROLS,controls_sha256=CONTROL_SHA,ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        setup_seconds=time.monotonic()-began;began=time.monotonic();torch.cuda.reset_peak_memory_stats()
        outputs=[]
        for index,(path,image,input_sha256) in enumerate(images):
            answer,event=model.generate(image)
            destination=args.output/f"{index:03d}.png";Image.fromarray(answer).save(destination)
            absolute=path.absolute()
            display=str(absolute.relative_to(ROOT)) if absolute.is_relative_to(ROOT) else path.name
            outputs.append(dict(input=display,input_sha256=input_sha256,output=destination.name,
                output_sha256=sha(destination),**event))
        atomic_json(args.output/"report.json",dict(outputs=outputs,setup_seconds=setup_seconds,
            inference_seconds=time.monotonic()-began,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            full_runtime_parameters=model.parameter_count(),checkpoint_sha256=args.checkpoint_sha256,
            implementation={n:sha(ROOT/n) for n in ("scripts/infer_adapted_pixel_spans.py",
                "ilm/visual_lm/adapted_pixel_runtime.py","ilm/visual_lm/contextual_pixel_learning.py")},
            training_manifests_opened=False,reference_images_opened=False,feature_cache_opened=False,
            original_head_checkpoint_opened=False,free_generation_demonstrated=False,general_ilm_goal_complete=False))
        print(json.dumps(dict(saved_images=len(outputs),output=str(args.output),source_copy_only=True)),flush=True)


if __name__=="__main__":main()

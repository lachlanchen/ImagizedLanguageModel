#!/usr/bin/env python3
"""Standalone deployment runner, copied beside learned weights and actor code."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from actor.correlated_native_runtime import generate_correlated, load_correlated


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-image",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("physical GPU 0 required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    device=torch.device("cuda:0");torch.cuda.set_device(device)
    folder=Path(__file__).resolve().parent
    model=load_correlated(folder/"weights.pt",folder/"architecture",device)
    pixels=np.asarray(Image.open(args.prompt_image).convert("L"))
    if pixels.shape[0]!=32 or pixels.shape[1]%32 or not 1<=pixels.shape[1]//32<=480:
        raise ValueError("input must be an aligned 32-pixel writing strip")
    ink=torch.from_numpy((pixels<128).astype(np.float32))
    prompt=ink.reshape(1,32,-1,32).permute(2,0,1,3)
    result=generate_correlated(model,prompt.to(device));images=result.pop("images").cpu()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(images=images),out.with_suffix(".pt"))
    if len(images):
        raster=(1-images).permute(1,2,0,3).reshape(32,-1).mul(255).byte().numpy()
    else:raster=np.full((32,32),255,dtype=np.uint8)
    Image.fromarray(raster,mode="L").save(out)
    result.update(generated_cells=len(images),output_placeholder_only=not len(images))
    out.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()

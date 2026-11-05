#!/usr/bin/env python3
"""Measure V34's native white-positive interface on fixed V65 training images."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec,ContinuousGlyphCodecConfig

ROOT=Path(__file__).resolve().parents[1]
EXPECTED="a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/writer_factorial_v65_20261105")
    args=parser.parse_args();run=args.run
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only reference correction required")
    if (run/"codec_native_report.json").exists():
        raise FileExistsError("native reference measurement is immutable")
    if not (run/"report.json").exists() or json.loads((run/"status.json").read_text())["stage"]!="complete":
        raise ValueError("finish the owned four-arm fitting process first")
    receipt=json.loads((run/"conditions_receipt.json").read_text())
    if sha(run/"conditions.pt")!=receipt["sha256"]:
        raise ValueError("fixed fitting images changed")
    checkpoint=ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
    if sha(checkpoint)!=EXPECTED:raise ValueError("qualified codec checkpoint changed")
    original_sha=sha(run/"codec_report.json")
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=ContinuousGlyphCodec(ContinuousGlyphCodecConfig(**saved["model_config"]))
    model.load_state_dict(saved["ema"]["shadow"]);model.requires_grad_(False).eval()
    images=torch.load(run/"conditions.pt",map_location="cpu",weights_only=True)["images"]
    native=(torch.cat((images[...,:32],images[...,32:]),dim=0)+1)/2
    logits=model.decode(model.encode(native))
    white=logits>=0
    pixels=torch.cat((white[:64],white[64:]),dim=-1).numpy()
    rows=[];hashes={}
    folder=run/"codec_native";folder.mkdir(exist_ok=True)
    for i,predicted in enumerate(pixels):
        a=~predicted[0];b=images[i,0].numpy()<0
        count=int(a.sum()+b.sum())
        rows.append(dict(index=i,foreground_f1=2*int((a&b).sum())/count if count else 1.,
            exact=bool(np.array_equal(a,b)),blank=not bool(a.any())))
        path=folder/f"{i:03}.png"
        Image.fromarray(predicted[0].astype(np.uint8)*255).save(path)
        hashes[path.name]=sha(path)
    result=dict(measurements={key:float(np.mean([r[key] for r in rows])) for key in ("foreground_f1","exact","blank")},
        cases=rows,checkpoint_sha256=EXPECTED,conditions_sha256=receipt["sha256"],
        input_polarity="white=1, black=0",output_polarity="nonnegative logits are white",
        original_invalid_polarity_report_sha256=original_sha,images_sha256=hashes,
        correction_plan_sha256=sha(ROOT/"references/writer_factorial_v65_codec_correction.md"),
        evaluator_sha256=sha(Path(__file__)),codec_source_sha256=sha(ROOT/"ilm/visual_lm/continuous_glyph_codec.py"),
        training_only=True,new_optimization=False,language_quality_claim=False)
    (run/"codec_native_report.json").write_text(json.dumps(result,indent=2)+"\n")
    if sha(run/"codec_report.json")!=original_sha:raise ValueError("original failure was overwritten")
    print(json.dumps(result["measurements"],indent=2))


if __name__=="__main__":main()

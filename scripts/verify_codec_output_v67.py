#!/usr/bin/env python3
"""Check every source image, continuous code and native decoder output in V67."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png(path):
    value=np.asarray(Image.open(path).convert("L"))
    if value.shape!=(32,64) or not set(np.unique(value))<={0,255}:
        raise ValueError("unmodified binary physical image required")
    return value<128


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/codec_output_v67_20261105")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();run=args.run
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only verification required")
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    report=json.loads((run/"report.json").read_text())
    if report["optimization_updates"]!=0 or report["trainable_codec_parameters"]!=0 or report["language_quality_claim"]:
        raise ValueError("fixed reconstruction-only scope required")
    for name,digest in report["sources"].items():
        original=subprocess.check_output(["git","show",report["source_commit"]+":"+name],cwd=ROOT)
        if hashlib.sha256(original).hexdigest()!=digest or sha(ROOT/name)!=digest:
            raise ValueError("audited codec implementation differs")
    if sha(run/"manifest.json")!=report["manifest_sha256"] or sha(run/"continuous_codes.pt")!=report["continuous_codes_sha256"]:
        raise ValueError("fixed source manifest or latent codes changed")
    manifest=json.loads((run/"manifest.json").read_text());cases=json.loads((run/"cases.json").read_text())
    if len(manifest)!=128 or len(cases)!=128:raise ValueError("retain every fixed case")
    data=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_development_20261105",
        purpose="development",expected_report_sha256=report["development_cache_sha256"])
    images=[];files={}
    for i,row in enumerate(manifest):
        if data.names[row["index"]]!=row["file"]:raise ValueError("source index differs")
        strip=data.target(row["index"])[...,:-128]
        target=strip[...,row["block"]*64:(row["block"]+1)*64]
        if not np.array_equal(png(run/f"reference/{i:03}.png"),target[0,0].numpy()<0):
            raise ValueError("reference pixels differ from the original source")
        images.append(target)
        files[f"reference/{i:03}.png"]=sha(run/f"reference/{i:03}.png")
    images=torch.cat(images)
    codec=load_qualified_writing_codec(ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt",torch.device("cpu"))
    # Build native white-positive tiles in a different order from the wrapper,
    # then reassemble exact physical left/right pairs.
    tiles=torch.cat((images[...,:32],images[...,32:]),dim=0).add(1).div(2)
    native=codec.encode(tiles)
    expected=torch.stack((native[:128],native[128:]),dim=1)
    actual=torch.load(run/"continuous_codes.pt",map_location="cpu",weights_only=True)
    if actual.shape!=(128,2,768) or not torch.allclose(actual,expected,atol=2e-5,rtol=0):
        raise ValueError("latent codes are not native encodings of the original images")
    measured={}
    for route in ("aligned","shuffled"):
        values=[]
        for i,row in enumerate(manifest):
            source=i if route=="aligned" else (i+1)%128
            logits=codec.decode(actual[source].reshape(2,768))
            white=logits>=0
            native_ink=~torch.cat((white[0:1],white[1:2]),dim=-1)[0,0].numpy()
            name=f"{route}/{i:03}.png";prediction=png(run/name)
            if not np.array_equal(prediction,native_ink):raise ValueError("wrapper decoder changes native writing")
            reference=images[i,0].numpy()<0;count=int(prediction.sum()+reference.sum())
            score=dict(foreground_f1=2*int((prediction&reference).sum())/count if count else 1.,
                exact=bool(np.array_equal(prediction,reference)),blank=not bool(prediction.any()))
            if score!=cases[i]["scores"][route] or any(cases[i][key]!=v for key,v in row.items()):
                raise ValueError("per-case score or fixed order differs")
            files[name]=sha(run/name);values.append(score)
        measured[route]={key:float(np.mean([value[key] for value in values])) for key in values[0]}
        for key,value in measured[route].items():
            if not math.isclose(value,report["measurements"][route][key],abs_tol=1e-12):
                raise ValueError("aggregate reconstruction metric differs")
    gain=measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
    gates=dict(foreground_f1=measured["aligned"]["foreground_f1"]>=.90,conditioning_gain=gain>=.20)
    if gates!=report["gates"] or all(gates.values())!=report["output_interface_pass"]:
        raise ValueError("fixed interface decision differs")
    for name in ("report.json","manifest.json","cases.json","continuous_codes.pt"):
        files[name]=sha(run/name)
    receipt=dict(files_sha256=files,source_references_verified=128,raw_native_decoder_outputs_verified=256,
        maximum_native_code_difference=float((actual-expected).abs().max()),measurements=measured,
        conditioning_gain=gain,gates=gates,output_interface_pass=all(gates.values()),
        target_images_supplied_to_encoder=True,target_images_supplied_to_decoder=False,
        language_quality_claim=False,general_ilm_goal_complete=False,verifier_sha256=sha(Path(__file__)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:value for key,value in receipt.items() if key!="files_sha256"},indent=2))


if __name__=="__main__":main()

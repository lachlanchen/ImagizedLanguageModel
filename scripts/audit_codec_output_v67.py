#!/usr/bin/env python3
"""Qualify an existing continuous output codec on the fixed reused image cases."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from ilm.visual_lm.codec_block_writer import CodecBlockWriter,load_qualified_writing_codec,CODEC_SHA
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT=Path(__file__).resolve().parents[1]
DEV_SHA="14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,pixels):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(((pixels.reshape(32,64).numpy()+1)*127.5).astype(np.uint8)).save(path)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/codec_output_v67_20261105")
    args=parser.parse_args();out=args.out
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only audit required")
    if (out/"report.json").exists():raise FileExistsError("completed codec audit is immutable")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);begun=time.perf_counter()
    names=("scripts/audit_codec_output_v67.py","references/codec_output_v67_protocol.md",
        "ilm/visual_lm/codec_block_writer.py","ilm/visual_lm/continuous_glyph_codec.py",
        "ilm/visual_lm/complete_writing_data.py")
    sources={name:sha(ROOT/name) for name in names}
    manifest_path=ROOT/"artifacts/grounded_writer_learning_v64_20261105/development/manifest.json"
    if sha(manifest_path)!="16646a5e8379a991d40345e9a2b844604feccf2bdf6edbc77d159c71f4ec3934":
        raise ValueError("registered development manifest changed")
    manifest=json.loads(manifest_path.read_text())
    if len(manifest)!=128:raise ValueError("complete fixed manifest required")
    data=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_development_20261105",
        purpose="development",expected_report_sha256=DEV_SHA)
    images=[]
    for i,row in enumerate(manifest):
        if data.names[row["index"]]!=row["file"]:raise ValueError("fixed source index differs")
        target=data.target(row["index"])[...,:-128]
        block=target[...,row["block"]*64:(row["block"]+1)*64].clone()
        images.append(block);save(out/"reference"/f"{i:03}.png",block)
    images=torch.cat(images)
    checkpoint=ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
    writer=CodecBlockWriter(load_qualified_writing_codec(checkpoint,torch.device("cpu")))
    codes=writer.encode_targets(images)
    torch.save(codes,out/"continuous_codes.pt")
    rows=[]
    for i,row in enumerate(manifest):
        result=dict(**row,scores={})
        reference=images[i:i+1]<0
        for route,index in (("aligned",i),("shuffled",(i+1)%128)):
            scores=writer.sample(codes[index:index+1])
            pixels=torch.where(scores>=0,1.,-1.);ink=pixels<0
            denominator=int(ink.sum()+reference.sum())
            result["scores"][route]=dict(foreground_f1=2*int((ink&reference).sum())/denominator if denominator else 1.,
                exact=bool(torch.equal(ink,reference)),blank=not bool(ink.any()))
            save(out/route/f"{i:03}.png",pixels)
        rows.append(result)
    measured={route:{key:float(np.mean([row["scores"][route][key] for row in rows]))
        for key in ("foreground_f1","exact","blank")} for route in ("aligned","shuffled")}
    gain=measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
    gates=dict(foreground_f1=measured["aligned"]["foreground_f1"]>=.90,conditioning_gain=gain>=.20)
    report=dict(measurements=measured,conditioning_gain=gain,gates=gates,output_interface_pass=all(gates.values()),
        sources=sources,source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        manifest_sha256=sha(manifest_path),development_cache_sha256=DEV_SHA,codec_sha256=CODEC_SHA,
        continuous_codes_sha256=sha(out/"continuous_codes.pt"),examples=128,development_reused=True,
        target_images_supplied_to_encoder=True,target_images_supplied_to_decoder=False,
        optimization_updates=0,codec_parameters=sum(p.numel() for p in writer.parameters()),
        trainable_codec_parameters=sum(p.numel() for p in writer.parameters() if p.requires_grad),
        language_quality_claim=False,general_ilm_goal_complete=False,elapsed_seconds=time.perf_counter()-begun)
    (out/"manifest.json").write_bytes(manifest_path.read_bytes())
    (out/"cases.json").write_text(json.dumps(rows,indent=2)+"\n")
    canvas=Image.new("RGB",(790,820),"white");draw=ImageDraw.Draw(canvas)
    for j,route in enumerate(("reference","aligned","shuffled")):
        draw.text((70+j*235,10),route,fill="black")
        for i in range(8):
            canvas.paste(Image.open(out/route/f"{i:03}.png").resize((192,96),Image.Resampling.NEAREST),(70+j*235,45+i*96))
            draw.text((5,65+i*96),f"{i:03}",fill="black")
    canvas.save(out/"first_eight.png")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Decode retained near-clean probes and quantify the declared time coverage."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F

from ilm.visual_lm.codec_block_writer import CodecBlockWriter,load_qualified_writing_codec
from scripts.cache_semantic_images_v72 import ROOT,TRAIN,TRAIN_SHA,CODEC,sha,atomic_json,committed_identity

SOURCES=("scripts/analyze_corpus_boundary_v75.py","ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/clean_training_objective.py","scripts/probe_corpus_fit_v75.py")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/corpus_boundary_v75_20261106")
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only retained-code analysis required")
    if args.out.exists():
        raise ValueError("choose a fresh immutable analysis directory")
    torch.set_num_threads(4)
    identity=committed_identity(SOURCES)
    fit=ROOT/"artifacts/corpus_fit_v75_20261106"
    run=ROOT/"artifacts/clean_corpus_v75_20261106"
    report=json.loads((fit/"report.json").read_text())
    verified=json.loads((fit/"verification.json").read_text())
    if (verified["report_sha256"]!=sha(fit/"report.json")
            or not verified["all_32_free_and_assisted_outputs_independently_scored"]
            or report["conditioning_sha256"]!=sha(fit/"conditioning.pt")):
        raise ValueError("verified retained conditioning predictions required")
    value=torch.load(fit/"conditioning.pt",map_location="cpu",weights_only=True)
    target,noise=value["target"],value["noise"]
    original=np.stack([np.array(Image.open(fit/"predictions"/f"{i:02}"/"reference.png"))[:,:64]
        for i in range(32)])
    writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    decoded={}
    def decode(name,codes):
        with torch.inference_mode():
            normalized=F.layer_norm(codes.reshape(-1,768),(768,),
                eps=writer.codec.config.layer_norm_epsilon).reshape(-1,2,768)
            pixels=torch.where(writer.predict(normalized)<0,0,255).to(torch.uint8)[:,0].numpy()
        decoded[name]=pixels
        pred,ref=pixels==0,original==0
        denominator=pred.sum((1,2))+ref.sum((1,2))
        f1=np.divide(2*(pred&ref).sum((1,2)),denominator,
            out=np.ones(32,dtype=np.float64),where=denominator!=0)
        return dict(mean_foreground_f1=float(f1.mean()),exact_original_blocks=int(np.all(pixels==original,axis=(1,2)).sum()),
            code_mse=float((codes.double()-target.double()).square().mean()),
            normalized_code_mse=float((normalized.flatten(1).double()-target.double()).square().mean()))
    scores={"supplied_code":decode("supplied_code",target)}
    for t in (.96875,1.):
        index=value["times"].index(t)
        name=str(t)
        scores[name]=dict(input=decode(name+"_input",(1-t)*noise+t*target),
            predicted=decode(name+"_predicted",value["own"][index]))
    training=json.loads((run/"training/training_report.json").read_text())
    checkpoint=run/"training/checkpoint_latest.pt"
    if sha(checkpoint)!=training["checkpoint_sha256"] or sha(TRAIN/"report.json")!=TRAIN_SHA:
        raise ValueError("production checkpoint or corpus inventory changed")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
    sampler=saved["sampler"]
    corpus=json.loads((TRAIN/"report.json").read_text())
    lengths={}
    for source in ("cmrc","dolly"):
        manifest=TRAIN/(source+"_accepted.jsonl")
        if sha(manifest)!=corpus["summaries"][source]["accepted_manifest_sha256"]:
            raise ValueError("block-length manifest changed")
        for line in manifest.read_text().splitlines():
            row=json.loads(line)
            lengths[row["file"]]=min(16,row["answer"]["blocks"])
    recipe=json.loads((run/"cache/recipe.json").read_text())
    if sha(run/"cache/recipe.json")!=training["identity"]["recipe_sha256"]:
        raise ValueError("training record order changed")
    per_record=torch.tensor([lengths[name] for name in recipe["names"]])
    if (sampler["draws"]!=640000 or sampler["size"]!=12976
            or sampler["epoch"]*sampler["size"]+sampler["cursor"]!=sampler["draws"]
            or not torch.equal(sampler["order"].sort().values,torch.arange(12976))):
        raise ValueError("complete production sampler state differs")
    blocks=int(sampler["epoch"]*per_record.sum()+per_record[sampler["order"][:sampler["cursor"]]].sum())
    distribution=training["identity"]["time_distribution"]
    t=.96875
    z=(math.log(t/(1-t))-distribution["mean"])/distribution["std"]
    tail=.5*math.erfc(z/math.sqrt(2))
    args.out.mkdir(parents=True)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    page=Image.new("RGB",(640,70+32*72),"white")
    draw=ImageDraw.Draw(page)
    for x,label in [(48,"Original block"),(194,"Input, t=.96875"),(340,"Prediction, t=.96875"),(486,"Prediction, t=1")]:
        draw.text((x,8),label,font=font,fill="black")
    for i in range(32):
        y=44+i*72
        draw.text((8,y+22),f"{i:02}",font=font,fill="black")
        for x,array in [(48,original[i]),(194,decoded["0.96875_input"][i]),
                (340,decoded["0.96875_predicted"][i]),(486,decoded["1.0_predicted"][i])]:
            page.paste(Image.fromarray(array).resize((128,64),Image.Resampling.NEAREST),(x,y))
    page.save(args.out/"first_block_boundary.png")
    result=dict(identity=identity,source_fit_report_sha256=sha(fit/"report.json"),
        source_fit_verification_sha256=sha(fit/"verification.json"),conditioning_sha256=sha(fit/"conditioning.pt"),
        codec_sha256=sha(CODEC),scores=scores,first_block_cases=32,
        actual_sampled_training_blocks=blocks,time_distribution=distribution,
        final_sampler_network_time=t,probability_time_at_or_above_final=tail,
        expected_training_samples_at_or_above_final=blocks*tail,actual_tail_draw_count_replayed=False,
        comparison_image_sha256=sha(args.out/"first_block_boundary.png"),
        cpu_decoder_reconstruction=True,constructed_noisy_reference_codes=True,
        actual_free_sampler_states_assessed=False,autonomous_image_generation_calls=0,
        language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(args.out/"report.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

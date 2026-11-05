#!/usr/bin/env python3
"""Independently check every fitting image and matched batch in writer V65."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT=Path(__file__).resolve().parents[1]
ARMS=("inherited_flow","inherited_direct","pixels_flow","pixels_direct")
WRITING_SHA="0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"


def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):value.update(chunk)
    return value.hexdigest()


def pixels(path):
    image=np.asarray(Image.open(path).convert("L"))
    if image.shape!=(32,64) or not set(np.unique(image))<={0,255}:
        raise ValueError("physical binary raster required: "+str(path))
    return image<128


def measures(prediction,reference):
    count=int(prediction.sum()+reference.sum())
    return dict(foreground_f1=2*int((prediction&reference).sum())/count if count else 1.,
        exact=bool(np.array_equal(prediction,reference)),blank=not bool(prediction.any()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/writer_factorial_v65_20261105")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();run=args.run
    report=json.loads((run/"report.json").read_text())
    if not report["all_arms_complete"] or report["language_quality_claim"] or report["general_ilm_goal_complete"]:
        raise ValueError("complete fitting-only result required")
    config=json.loads((run/"configuration.json").read_text())
    if report["sources"]!=config["sources"] or set(report["arms"])!=set(ARMS):
        raise ValueError("source or arm identity differs")
    if config["arms"]!=list(ARMS) or config["updates_per_arm"]!=2000 or not config["training_only"]:
        raise ValueError("registered exposure differs")
    for name,digest in report["sources"].items():
        raw=subprocess.check_output(["git","show",config["source_commit"]+":"+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError("committed implementation differs: "+name)
    files={}
    def bind(name):
        files[name]=sha(run/name)
        return files[name]
    for name in ("configuration.json","report.json","manifest.json","conditions.pt",
            "conditions_receipt.json","initial_writer.pt","initial_writer_receipt.json","codec_report.json"):
        bind(name)
    condition_receipt=json.loads((run/"conditions_receipt.json").read_text())
    if files["conditions.pt"]!=condition_receipt["sha256"]:
        raise ValueError("cached conditions changed")
    cache=torch.load(run/"conditions.pt",map_location="cpu",weights_only=True)
    if (cache["images"].shape!=(64,1,32,64) or cache["inherited"].shape!=(64,2,4096)
            or cache["pixels"].shape!=(64,2,4096)):
        raise ValueError("conditional geometry differs")
    expected_schedule=torch.randint(64,(2000,16),generator=torch.Generator().manual_seed(20266521))
    if not torch.equal(cache["schedule"],expected_schedule):
        raise ValueError("shared batch schedule differs from registered seed")
    data=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_train_20261105",
        purpose="train",expected_report_sha256=WRITING_SHA)
    generator=torch.Generator().manual_seed(20266510)
    pools=[[i for i,name in enumerate(data.names) if name.startswith(prefix)] for prefix in ("cmrc/","dolly/")]
    chosen=[[pool[i] for i in torch.randperm(len(pool),generator=generator)[:32].tolist()] for pool in pools]
    indices=[index for pair in zip(*chosen) for index in pair]
    manifest=json.loads((run/"manifest.json").read_text())
    references=[]
    for i,index in enumerate(indices):
        target=data.target(index)[...,:-128]
        block=int(torch.randint(target.shape[-1]//64,(),generator=generator))
        expected=target[...,block*64:(block+1)*64]
        row=manifest[i]
        if (row["index"]!=index or row["block"]!=block or row["file"]!=data.names[index]
                or row["source_sha256"]!=data.report["files_sha256"][data.names[index]]
                or not torch.equal(expected,cache["images"][i:i+1])):
            raise ValueError("fixed fitting case differs from original train data")
        reference=pixels(run/f"reference/{i:03}.png")
        if not np.array_equal(reference,expected[0,0].numpy()<0):
            raise ValueError("reference image polarity or pixels differ")
        references.append(reference);bind(f"reference/{i:03}.png")
    tile=torch.stack((cache["images"][...,:32],cache["images"][...,32:]),dim=1).flatten(2)
    expected_raw=torch.cat((tile,tile,torch.ones_like(tile),-torch.ones_like(tile)),dim=-1)
    if not torch.equal(cache["pixels"],expected_raw):
        raise ValueError("oracle pixel condition is not lossless")
    results={}
    for arm in ARMS:
        row=report["arms"][arm]
        if (row["sources"]!=config["sources"] or row["initial_sha256"]!=files["initial_writer.pt"]
                or row["conditions_sha256"]!=files["conditions.pt"] or row["update"]!=2000
                or row["final_sha256"]!=bind(arm+"/final_ema.pt")):
            raise ValueError("matched weights, exposure or conditions differ")
        journal=[json.loads(line) for line in (run/arm/"metrics.jsonl").read_text().splitlines()]
        if [item["update"] for item in journal]!=list(range(1,2001)):
            raise ValueError("missing/extra fitting updates")
        for i,item in enumerate(journal):
            if (item["indices"]!=expected_schedule[i].tolist() or not math.isfinite(item["loss"])
                    or not math.isfinite(item["gradient_norm"])):
                raise ValueError("batch exposure or numerical state differs")
        resume=json.loads((run/arm/"resume_verification.json").read_text())
        if not resume["passed"] or not all(resume["checks"].values()) or resume["extra_optimizer_replays"]!=1:
            raise ValueError("exact optimizer/EMA/RNG replay did not pass")
        cases=json.loads((run/arm/"cases.json").read_text())
        measured={}
        for route in ("aligned","shuffled"):
            rows=[]
            for i,reference in enumerate(references):
                name=f"{arm}/{route}/{i:03}.png";actual=measures(pixels(run/name),reference)
                if cases[i]["index"]!=i or cases[i]["scores"][route]!=actual:
                    raise ValueError("individual raw-image score differs: "+name)
                bind(name);rows.append(actual)
            measured[route]={key:float(np.mean([value[key] for value in rows])) for key in rows[0]}
            for key,value in measured[route].items():
                if not math.isclose(value,row["measurements"][route][key],abs_tol=1e-12):
                    raise ValueError("aggregate image metric differs")
        gain=measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
        gates=dict(foreground_f1=measured["aligned"]["foreground_f1"]>=.95,conditioning_gain=gain>=.50)
        if gates!=row["gates"] or row["fitting_criterion_pass"]!=all(gates.values()):
            raise ValueError("registered fitting criterion differs")
        results[arm]=dict(measurements=measured,conditioning_gain=gain,gates=gates,
            fitting_criterion_pass=all(gates.values()),all_fixed_batches_verified=True)
        for suffix in ("metrics.jsonl","cases.json","report.json","resume_verification.json"):
            bind(arm+"/"+suffix)
    codec=[]
    for i,reference in enumerate(references):
        name=f"codec/{i:03}.png";row=measures(pixels(run/name),reference)
        if row!=report["codec"]["cases"][i]:raise ValueError("codec reference score differs")
        bind(name);codec.append(row)
    codec_means={key:float(np.mean([row[key] for row in codec])) for key in codec[0]}
    for key,value in codec_means.items():
        if not math.isclose(value,report["codec"]["measurements"][key],abs_tol=1e-12):
            raise ValueError("codec aggregate differs")
    native=json.loads((run/"codec_native_report.json").read_text())
    bind("codec_native_report.json")
    if (native["original_invalid_polarity_report_sha256"]!=files["codec_report.json"]
            or native["conditions_sha256"]!=files["conditions.pt"]
            or native["input_polarity"]!="white=1, black=0" or native["new_optimization"]):
        raise ValueError("native reference correction changed its fixed input or scope")
    native_rows=[]
    for i,reference in enumerate(references):
        name=f"codec_native/{i:03}.png";row=measures(pixels(run/name),reference)
        if dict(index=i,**row)!=native["cases"][i] or bind(name)!=native["images_sha256"][f"{i:03}.png"]:
            raise ValueError("native reference raw image score differs")
        native_rows.append(row)
    native_means={key:float(np.mean([row[key] for row in native_rows])) for key in native_rows[0]}
    for key,value in native_means.items():
        if not math.isclose(value,native["measurements"][key],abs_tol=1e-12):
            raise ValueError("native codec aggregate differs")
    receipt=dict(source_commit=config["source_commit"],files_sha256=files,arms=results,
        original_codec_invalid_polarity=codec_means,codec_native=native_means,
        original_training_images_verified=64,raw_generated_image_scores_verified=512,codec_image_scores_verified=128,
        independent_development_or_language_claim=False,source_condition_control_has_target_pixels=True,
        verifier_sha256=sha(Path(__file__)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:value for key,value in receipt.items() if key!="files_sha256"},indent=2))


if __name__=="__main__":main()

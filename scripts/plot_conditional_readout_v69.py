#!/usr/bin/env python3
"""Show the first fixed diagnostic block directly from verified V69 tensors."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("plot verified CPU tensors with CUDA hidden")
    report=json.loads((args.run/"fit_report.json").read_text())
    verification=json.loads((args.run/"verification.json").read_text())
    capture=json.loads((args.run/"capture_report.json").read_text())
    preparation=json.loads((args.run/"preparation.json").read_text())
    if (not verification["offline_checks_pass"] or sha(args.run/"fit_report.json")!=verification["fit_report_sha256"]
            or sha(args.run/"predictions.pt")!=report["predictions_sha256"]
            or sha(args.run/"capture_report.json")!=report["capture_report_sha256"]
            or sha(args.run/"preparation.json")!=capture["preparation_sha256"]
            or sha(args.run/"prepared.pt")!=preparation["prepared_sha256"]):
        raise ValueError("source-bound verified predictions required")
    rows=torch.load(args.run/"prepared.pt",map_location="cpu",weights_only=True)
    predictions=torch.load(args.run/"predictions.pt",map_location="cpu",weights_only=True)
    reference=rows[512]["reference"][0,0].numpy()>=0
    if rows[512]["id"]!="DEV_477_QUERY_0":
        raise ValueError("show the first originally fixed Chinese case")
    selected={(p["route"],p["head"]):p["pixels"][0].numpy()>=0 for p in predictions if p["index"]==512}
    fig,axes=plt.subplots(2,5,figsize=(12,4.2))
    for j,(kind,title) in enumerate((("reference","Reference"),("original","Original head"),
            ("mean","Mean control"),("shared","Shared ridge"),("joint","Joint ridge"))):
        for i,route in enumerate(("normal","blank")):
            image=reference if kind=="reference" else selected[route,kind]
            axes[i,j].imshow(image,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            axes[i,j].axis("off")
            if i==0:axes[i,j].set_title(title,fontsize=12)
    for i,title in enumerate(("Normal","Blank question")):
        axes[i,0].text(-.12,.5,title,transform=axes[i,0].transAxes,ha="right",va="center",rotation=90)
    fig.suptitle("First fixed Chinese development case: first 32 × 64 block only",fontsize=14)
    fig.text(.5,.03,"Actual unmodified binary outputs; nearest-neighbour enlargement for viewing. Stopping is not evaluated.",ha="center",fontsize=10)
    fig.tight_layout(rect=(.02,.07,1,.92));args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160);plt.close(fig)


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Completed V61 native image-language evidence; preserve failed outputs."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_evidence import HERE,ROOT,save
from generate_v57_evidence import pad_answer

RUN=ROOT/"artifacts/native_pixel_instruction_v61_20261105"


def main():
    report=json.loads((RUN/"development/report.json").read_text())
    training=json.loads((RUN/"training/training_report.json").read_text())
    if training["update"]!=1000 or report["decision"]!="reject_native_free_reading":
        raise ValueError("completed native audit required")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,
        "image.composite_image":False})
    fig=plt.figure(figsize=(7.25,4.75),layout="constrained")
    grid=fig.add_gridspec(2,2,height_ratios=(1.4,1.3))
    a,b,c=fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])
    values=[report["likelihood"][key] for key in ("base","trained")]
    a.bar([0,1],values,color=["#939b9f","#28638b"],width=.55)
    for i,value in enumerate(values):a.text(i,value+.018,f"{value:.4f}",ha="center",fontsize=8)
    a.set(xticks=[0,1],xticklabels=["Released base","Adapted"],ylim=(0,.8),
          ylabel="Answer-pixel NLL (nats)",title="Held-out pixel fit")
    routes=("trained","trained_blank","base");x=np.arange(3)
    for offset,key,label,color in [(-.17,"raster_f1","Exact-raster F1","#ae593f"),
                                    (.17,"pixel_foreground_f1","Foreground F1","#28638b")]:
        scores=[100*report["measurements"][r][key] for r in routes]
        b.bar(x+offset,scores,.32,label=label,color=color)
        for i,value in zip(x+offset,scores):b.text(i,value+.35,f"{value:.2f}",ha="center",fontsize=7)
    b.set(xticks=x,xticklabels=["Adapted","Blank question","Base"],ylim=(0,19),
          ylabel="Image score (%)",title="Generated answers")
    b.legend(frameon=False,fontsize=7,loc="upper right")
    c.axis("off");c.set(xlim=(0,640),ylim=(220,0))
    from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
    renderer=VisualQARenderer();records={r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")}
    c.set_title("First two fixed audit rows",loc="left",weight="bold",fontsize=8)
    for i,identifier in enumerate(report["primary_ids"][:2]):
        reference=renderer(records[identifier]["references"][0]["text"])
        for j,(label,images) in enumerate([
            ("Reference",reference),
            ("Adapted",torch.load(RUN/"development/predictions/trained"/(identifier+".pt"),map_location="cpu",weights_only=False)["images"]),
            ("Blank question",torch.load(RUN/"development/predictions/trained_blank"/(identifier+".pt"),map_location="cpu",weights_only=False)["images"]),
        ]):
            y=5+i*106+j*32
            c.text(0,y+16,f"{i+1}: {label}",fontsize=7,va="center")
            c.imshow(pad_answer(images,16),cmap="gray_r",vmin=0,vmax=1,interpolation="none",
                     extent=(100,612,y+32,y),aspect="equal")
    c.text(0,218,"0/256 exact answers; 0/393 pairs both correct. White display padding only.",fontsize=7)
    for ax,label in zip((a,b,c),"abc"):
        ax.text(-.10 if ax is not c else -.04,1.05,label,transform=ax.transAxes,fontweight="bold")
    save(fig,"Figure_6_native_pixel_language")
    sources={"v61_development_report.json":RUN/"development/report.json",
        "v61_training_report.json":RUN/"training/training_report.json",
        "v61_standalone_verification.json":RUN/"standalone/verification.json",
        "v61_transcript_diagnostic.json":RUN/"transcript_diagnostic/report.json"}
    hashes={}
    for name,source in sources.items():
        (HERE/"evidence"/name).write_text(source.read_text().replace(str(ROOT)+"/",""))
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    (HERE/"evidence/native_pixel_source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print("Generated V61 native-pixel evidence from completed audit")


if __name__=="__main__":main()

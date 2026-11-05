#!/usr/bin/env python3
"""Completed conditional bitmap-density evidence, preserving raw answers."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
import numpy as np
import torch

from generate_evidence import HERE,ROOT,save
from generate_v57_evidence import pad_answer

RUN=ROOT/"artifacts/correlated_native_density_v62_20261105"


def completed_report():
    status=json.loads((RUN/"pipeline_status.json").read_text())
    if status["status"]!="audit_complete":raise ValueError("completed matched audit required")
    report=json.loads((RUN/"development/report.json").read_text())
    for k in (1,8):
        training=json.loads((RUN/f"k{k}/training_report.json").read_text())
        if training["update"]!=1000 or training["smoke_only"]:raise ValueError("incomplete training")
    return report


def main():
    report=completed_report()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,
        "image.composite_image":False})
    fig=plt.figure(figsize=(7.25,6.6),layout="constrained")
    grid=fig.add_gridspec(3,2,height_ratios=(.85,1.6,1.5))
    a,b,c,d=fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1]),fig.add_subplot(grid[2,:])
    a.axis("off");a.set(xlim=(0,1),ylim=(0,1))
    boxes=[(.015,.23,"Writing images"),(.255,.23,"Causal visual state"),(.515,.23,"K whole-image experts"),(.785,.23,"Generated pixels")]
    widths=[.18,.20,.22,.20]
    for (x,y,label),w in zip(boxes,widths):
        a.add_patch(Rectangle((x,y),w,.35,facecolor="#f2f5f6",edgecolor="#677b87",lw=.7))
        a.text(x+w/2,y+.175,label,ha="center",va="center",fontsize=7)
    for ((x,y,_),w),(next_box,_) in zip(zip(boxes,widths),zip(boxes[1:],widths[1:])):
        a.add_patch(FancyArrowPatch((x+w,.405),(next_box[0],.405),arrowstyle="->",mutation_scale=8,color="#536674",lw=.8))
    a.text(.5,.82,r"$p(y\mid h)=\sum_{k=1}^{K}\pi_k(h)\,\prod_i\,\mathrm{Bernoulli}(y_i;\sigma(a_{ki}(h)))$",ha="center",fontsize=10)
    a.text(.5,.04,"One expert choice couples the 2,048 pixels; no expert has a fixed character identity.",ha="center",fontsize=7)
    values=[report["likelihood"][r] for r in ("k1","k8")]
    b.bar([0,1],values,color=["#929da4","#28638b"],width=.55)
    for i,value in enumerate(values):b.text(i,value+max(values)*.04,f"{value:.4f}",ha="center",fontsize=8)
    b.set(xticks=[0,1],xticklabels=["K = 1","K = 8"],ylim=(0,max(values)*1.35),
          ylabel="Answer-pixel NLL (nats)",title="Held-out likelihood")
    routes=("k1","k1_blank","k8","k8_blank");x=np.arange(4)
    all_values=[]
    for offset,key,label,color in [(-.17,"raster_f1","Exact-raster F1","#ae593f"),(.17,"pixel_foreground_f1","Foreground F1","#28638b")]:
        scores=[100*report["measurements"][r][key] for r in routes];all_values.extend(scores)
        c.bar(x+offset,scores,.32,label=label,color=color)
        for i,value in zip(x+offset,scores):c.text(i,value+.35,f"{value:.2f}",ha="center",fontsize=6.5)
    c.set(xticks=x,xticklabels=["K1","K1 blank Q","K8","K8 blank Q"],ylim=(0,max(all_values)*1.4+1),
          ylabel="Image score (%)",title="Unrepaired generated answers")
    c.legend(frameon=False,fontsize=7,loc="upper right")
    d.axis("off");d.set(xlim=(0,640),ylim=(220,0))
    from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
    records={r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")};renderer=VisualQARenderer()
    d.set_title("First two fixed audit rows",loc="left",weight="bold",fontsize=8)
    for i,identifier in enumerate(report["primary_ids"][:2]):
        reference=renderer(records[identifier]["references"][0]["text"])
        for j,(label,images) in enumerate([("Reference",reference)]+[(f"K = {k}",torch.load(RUN/f"development/predictions/k{k}"/(identifier+".pt"),map_location="cpu",weights_only=False)["images"]) for k in (1,8)]):
            if len(images)>20:raise ValueError("display would crop generated ink")
            y=5+i*106+j*32
            d.text(0,y+16,f"{i+1}: {label}",fontsize=7,va="center")
            d.imshow(pad_answer(images,20),cmap="gray_r",vmin=0,vmax=1,interpolation="none",
                     extent=(100,612,y+25.6,y),aspect="equal")
    d.text(0,218,"White display padding only. Exact-raster metrics do not measure all readable variants.",fontsize=7)
    for ax,label in zip((a,b,c,d),"abcd"):
        ax.text(-.04 if ax in (a,d) else -.10,1.05,label,transform=ax.transAxes,fontweight="bold")
    save(fig,"Figure_7_correlated_density")
    sources={"v62_development_report.json":RUN/"development/report.json",
        "v62_preflight_report.json":RUN/"preflight/report.json"}
    for k in (1,8):
        sources[f"v62_k{k}_training_report.json"]=RUN/f"k{k}/training_report.json"
        sources[f"v62_k{k}_standalone_verification.json"]=RUN/f"standalone/k{k}/verification.json"
    auxiliary=RUN/"vision_transcript_diagnostic/report.json"
    if auxiliary.exists():sources["v62_vision_transcript_diagnostic.json"]=auxiliary
    hashes={}
    for name,path in sources.items():
        (HERE/"evidence"/name).write_text(path.read_text().replace(str(ROOT)+"/","").replace(str(Path.home())+"/","~/"))
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/"evidence/correlated_density_source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print("Generated completed V62 density evidence")


if __name__=="__main__":main()

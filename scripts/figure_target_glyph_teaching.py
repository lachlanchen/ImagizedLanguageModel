"""Plot target-side teaching and place original, replay-verified answer PNGs in the plate."""
from itertools import chain
from pathlib import Path
import numpy as np
from scripts.figure_book_breadth import FIG,plt
from scripts.figure_book_rehearsal import plate
from scripts.reserve_target_glyph_book_panel import PUB, RUN, read, dump, sha
from ilm.visual_lm.target_glyph_teaching import ARMS,MODELS


def report():
    result=read(PUB/"evaluation.json")["primary"]
    modes=["start","likelihood","guided","exchange","qwen"]
    primary=dict(title="a   Complete translations",unit="Whole meanings credited per 96 requests",
                 labels=["Starting writer","Human likelihood","Teacher + human","Exchanged source","Qwen 8B Q4"],
                 series={s:[result["directions"][lang].get(m,0) for m in modes]
                         for s,lang in [("To English","en"),("To Chinese","zh")]})
    trends={}
    for k,arm in zip(("Human likelihood","Teacher + human"),ARMS,strict=True):
        h=read(MODELS/arm/"fit.json")["history"]
        trends[k]=[round(float(np.average(r["deficit"],weights=r["boundaries"])),4) for r in h]
    fit=dict(title="b   Training on visible probability",unit="Mean one-sided divergence per aligned boundary",
             labels=["Pass 1","Pass 2","Pass 3","Pass 4"],series=trends)
    colors={"To English":"#126957","To Chinese":"#b17646","Human likelihood":"#757c80","Teacher + human":"#126957"}
    panels=[primary,fit]
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.45))
    for ax,p in zip(axes,panels,strict=True):
        x=np.arange(len(p["labels"]))
        for j,(series,values) in enumerate(p["series"].items()):
            bars=ax.barh(x+(j-.5)*.34,values,.31,color=colors[series],label=series)
            ax.bar_label(bars,fmt="%.0f" if p is primary else "%.3f",fontsize=8,padding=3)
        ax.invert_yaxis()
        ax.set_yticks(x,p["labels"],fontsize=8)
        ax.tick_params(length=0,labelsize=8)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis="x",linewidth=.5,color="#d2dedc")
        ax.set_title(p["title"],loc="left",fontsize=10,fontweight="bold")
        ax.set_xlabel(p["unit"],fontsize=8)
        ax.set_xlim(0,100 if p is primary else max(chain.from_iterable(p["series"].values()))*1.24)
        ax.legend(loc="upper center",bbox_to_anchor=(.48,-.19),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.125,right=.98,top=.90,bottom=.28,wspace=.65)
    for ext in ("png","pdf"): fig.savefig(FIG/("Figure_220_optical_teacher."+ext),dpi=190)
    plt.close(fig)
    dump(PUB/"chart.json",dict(colors=colors,panels=panels))


def answer_plate():
    evaluation=read(PUB/"evaluation.json")
    replay=read(PUB/"replay.json")
    differing={(d["i"],d["mode"]) for d in replay["differences"]}
    if replay["archive"]!=evaluation["archive"] or sum(replay["reproduced"].values())+len(differing)!=768:
        raise ValueError("Raw ink replay incomplete")
    records=evaluation["primary"]["records"]
    raw=read(RUN/"raw.json")["answers"]
    tests=[("First gained meaning",[r["guided"] and not r["likelihood"] for r in records]),
           ("First lost meaning",[r["likelihood"] and not r["guided"] for r in records]),
           ("First rejected guided reply",[not r["guided"] for r in records]),
           ("First credited guided reply",[r["guided"] for r in records]),
           ("First changed output image",[r["guided"]["png"]!=r["likelihood"]["png"] for r in raw])]
    picked=[];used=set()
    for title,mask in tests:
        first=next((i for i,v in enumerate(mask) if v),None)
        if first is not None and first not in used:
            used.add(first);picked.append((title,first))
        if len(picked)==3: break
    folder=FIG/"target_glyph_teaching";folder.mkdir(exist_ok=True)
    scenes=[]
    for j,(title,i) in enumerate(picked):
        src=RUN/"png"/f"{i:04d}"
        files=[]
        for k,label in [("input","Pictured source"),("likelihood","Human likelihood"),("guided","Teacher + human")]:
            f=src/(k+".png")
            if (i,k) in differing: raise ValueError("Illustrated original differs across CPU and GPU; label it explicitly")
            if k!="input" and sha(f)!=raw[i][k]["png"]: raise ValueError("Different illustrated image")
            files.append((label,f"{i:04d}-{k}",f))
        scenes.append((chr(97+j)+"    "+title,files))
    plate("scene","Figure_221_optical_teacher_answers",scenes,assets=folder,evidence=PUB)
    dump(PUB/"illustrated.json",dict(examples=[dict(index=i,selection=t) for t,i in picked],replay=sha(PUB/"replay.json")))


if __name__=="__main__":
    report()
    answer_plate()

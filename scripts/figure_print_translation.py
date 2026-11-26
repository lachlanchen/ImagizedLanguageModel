"""Build the submission figures directly from the frozen visual test and timing samples."""
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json

import fitz
import numpy as np
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from scripts.figure_book_breadth import plt, FIG
from scripts.reserve_print_translation import ROOT, PUB, RUN, read, sha, dump
from scripts.evaluate_print_translation import ARCHIVE, MODES

ASSETS=FIG/"print_translation"
NAMES={"the-count-of-monte-cristo":"The Count of Monte Cristo","resurrection":"Resurrection",
       "sanshiro":"Sanshiro","siddhartha":"Siddhartha"}
METHODS={"guided":"Canonical","serif":"Serif","adapted":"Normalized","qwen":"Text Qwen","reference":"Reference"}
GREEN="#156c62"
INK="#19332f"


def freeze_summary():
    assessment=read(PUB/"assessment.json")
    replay=read(PUB/"replay.json")
    assert assessment["archive"]==sha(ARCHIVE)==replay["archive"] and len(assessment["primary"]["records"])==192
    assert all(replay["matches"].get(m,0)+sum(d["mode"]==m for d in replay["differences"])==192 for m in MODES)
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("outcomes.json"))
        assert len(rows)==192 and len(z.namelist())==1+8*len(rows)
        for r in rows:
            for name,ans in r["answers"].items():
                from hashlib import sha256
                assert ans["png"]==sha256(z.read(f'{r["id"]:04d}/{name}.png')).hexdigest()
    grades=assessment["primary"]["records"]
    stats={}
    for mode in ("serif","adapted"):
        full=[]; same=[]; total=correct=cap=0
        for row,verdict in zip(rows,grades,strict=True):
            n=row["source_cells"]["n"]; c=row["source_cells"][mode]
            total+=n;correct+=c
            full.append(n==c)
            a,b=(row["answers"][k] for k in ("guided",mode))
            eq=(a["png"],a["stop"],a["cells"])==(b["png"],b["stop"],b["cells"])
            if eq and verdict["guided"]!=verdict[mode]: raise ValueError("Identical candidate grade differs")
            same.append(eq);cap+=int(verdict["guided"] or not eq)
        stats[mode]=dict(total_cells=total,exact_cells=correct,exact_sources=sum(full),equal_answers=sum(same),
                         sharp_credit_ceiling=cap)
    known={m:sum(a["unknown"]==0 and a["cells"]>0 for r in rows for a in [r["answers"][m]]) for m in MODES}
    finite={m:sum(r["answers"][m]["stop"] for r in rows) for m in MODES}
    assert len({r["id"] for r in rows})==192
    choices=dict(restore=next((i for i,r in enumerate(rows) if r["answers"]["serif"]["png"]!=r["answers"]["guided"]["png"] and r["answers"]["adapted"]["png"]==r["answers"]["guided"]["png"]),None),
                 credit=next((i for i,g in enumerate(grades) if g["guided"]),None),
                 rejection=next((i for i,g in enumerate(grades) if not g["guided"]),None))
    paired=Counter((g["serif"],g["adapted"]) for g in grades)
    out=dict(shapes=stats,legible_nonempty=known,stops=finite,selected=choices,
             serif_to_adapted={f"{int(a)}_{int(b)}":paired[a,b] for a in (False,True) for b in (False,True)},
             foreign_work=read(PUB/"panel.json")["titles"],assessment=sha(PUB/"assessment.json"),
             implementation=sha(Path(__file__)))
    dump(PUB/"summary.json",out)
    return rows,grades,out


def output_plate(rows,summary,grades):
    ASSETS.mkdir(exist_ok=True)
    groups=[]
    selected=summary["selected"]
    if selected["restore"] is not None:
        i=selected["restore"]
        groups.append((i,"One passage in two print faces",
                       [("Canonical source","canonical"),("Serif source","serif-input"),("From serif","serif"),("After normalization","adapted")]))
    rejected=selected["rejection"]
    if rejected is not None and rejected not in [g[0] for g in groups]:
        groups.append((rejected,"First rejected canonical answer",[("Pictured source","canonical"),("Generated answer","guided")]))
    draw_strips(rows,groups,"Print_translation_outputs","scene")
    credited = [(i, "English output" if r["to"]=="en" else "Chinese output",
                 [("Pictured source","canonical"),("Generated answer","guided")])
                for i,(r,g) in enumerate(zip(rows,grades,strict=True)) if g["guided"]]
    draw_strips(rows,credited,"Print_translation_credits","credits")


def draw_strips(rows,groups,figure,receipt):
    items=[]; y=3
    diff={(d["index"],d["mode"]) for d in read(PUB/"replay.json")["differences"]}
    with ZipFile(ARCHIVE) as z:
        for j,(i,title,parts) in enumerate(groups):
            items.append(dict(type="heading",x=0,y=y,text=f'{chr(97+j)}   {title}',meta=NAMES[rows[i]["work"]]+f'   /   {i:04d}'))
            y+=20
            for label,k in parts:
                assert (i,k) not in diff
                blob=z.read(f"{i:04d}/{k}.png")
                name=f"{i:04d}-{k}.png"
                (ASSETS/name).write_bytes(blob)
                with Image.open(BytesIO(blob)) as im:w,h=im.size
                scale=11.8/32
                items.append(dict(type="strip",name=name,label=label,x=111,y=y,w=w*scale,h=h*scale,sha256=sha(ASSETS/name)))
                y+=max(16,h*scale+4)
            y+=7
    width=504; height=y-4
    file=FIG/(figure+".pdf");canvas=Canvas(str(file),pagesize=(width,height))
    for item in items:
        top=height-item["y"]
        if item["type"]=="heading":
            canvas.setFont("Helvetica-Bold",11);canvas.setFillColor(HexColor(INK))
            canvas.drawString(0,top-10,item["text"])
            canvas.setFont("Helvetica",7.8);canvas.setFillColor(HexColor("#45605a"))
            canvas.drawRightString(width,top-10,item["meta"])
        else:
            canvas.setFont("Helvetica",8.5);canvas.setFillColor(HexColor("#3b504a"))
            canvas.drawString(8,top-9,item["label"])
            canvas.drawImage(str(ASSETS/item["name"]),item["x"],top-item["h"],item["w"],item["h"])
    canvas.showPage();canvas.save()
    with fitz.open(file) as doc: doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(file.with_suffix(".png"))
    dump(PUB/(receipt+".json"),dict(width=width,height=height,items=items))


def outcome_plot(rows,grades):
    from matplotlib.colors import ListedColormap
    blocks=[]
    fig,axes=plt.subplots(2,4,figsize=(10.3,3.5),sharex=True,sharey=True)
    for t,lang in enumerate(("en","zh")):
        for j,(work,title) in enumerate(NAMES.items()):
            ix=[i for i,r in enumerate(rows) if r["work"]==work and r["to"]==lang]
            assert len(ix)==24
            binary=np.array([[int(grades[i][m]) for i in ix] for m in METHODS])
            blocks.append(dict(work=title,language=lang,ids=ix,values=binary.tolist()))
            ax=axes[t,j]
            ax.imshow(binary,cmap=ListedColormap(["#e3e8e6",GREEN]),vmin=0,vmax=1,aspect="auto",interpolation="nearest")
            ax.set_xticks(np.arange(-.5,24,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
            ax.grid(which="minor",color="white",linewidth=.55)
            ax.tick_params(which="both",length=0,labelsize=8)
            ax.set_xticks([0,11,23],["1","12","24"]);ax.set_yticks(range(5),list(METHODS.values()))
            for edge in ax.spines.values():edge.set_visible(False)
            ax.set_title(title if t==0 else "",fontsize=9,loc="left",weight="bold",color=INK)
    fig.text(.01,.73,"To English",rotation=90,va="center",fontsize=9,weight="bold")
    fig.text(.01,.32,"To Chinese",rotation=90,va="center",fontsize=9,weight="bold")
    fig.text(.53,.025,"Passage within each book",ha="center",fontsize=9)
    fig.subplots_adjust(left=.115,right=.993,top=.86,bottom=.16,wspace=.13,hspace=.35)
    for ext in ("png","pdf"): fig.savefig(FIG/("Print_translation_outcomes."+ext),dpi=190)
    plt.close(fig)
    dump(PUB/"blocks.json",dict(methods=METHODS,blocks=blocks))


def time_plot():
    data=read(PUB/"inference.json")
    fig,ax=plt.subplots(figsize=(5,2.05))
    samples=data["samples"]
    for i,row in enumerate(samples):
        yy=np.array([row["direct"],row["compiled"]])*1000
        ax.plot([0,1],yy,color="#8da49d",lw=1.2)
        ax.scatter([0,1],yy,color=["#596a6e",GREEN],s=21,zorder=3)
        # Label each pair at the midpoint; rounds one and two are close.
        x=[.20,.74,.22,.76,.48][i]
        ax.text(x,yy[0]*(1-x)+yy[1]*x+18,str(i+1),fontsize=8,ha="center",color=INK)
    ax.set_xlim(-.22,1.22);ax.set_ylim(0,1020)
    ax.set_xticks([0,1],["Direct motor","Compiled motor"])
    ax.set_ylabel("192 requests (ms)",fontsize=9)
    ax.tick_params(length=0,labelsize=8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="y",color="#e2e7e4",lw=.6)
    fig.subplots_adjust(left=.15,right=.98,bottom=.20,top=.97)
    for ext in ("png","pdf"):fig.savefig(FIG/("Print_translation_time."+ext),dpi=190)
    plt.close(fig)


def main():
    rows,grades,summary=freeze_summary()
    output_plate(rows,summary,grades);outcome_plot(rows,grades);time_plot()


if __name__=="__main__":
    main()

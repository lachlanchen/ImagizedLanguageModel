"""Publish the prespecified examples, every held grade and the two interface controls."""
from io import BytesIO
import json
from zipfile import ZipFile
import fitz
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from scripts.figure_book_breadth import FIG, plt
from scripts.reserve_optical_span_rollout import PUB, dump, read, sha
from scripts.evaluate_optical_span_rollout import ARCHIVE

GREEN = "#156c62"
WHITE = "#e3e8e6"
NAMES = {"sentimental-education":"Sentimental Education","walking":"Walking",
         "tagore-stray-birds":"Stray Birds","sound-and-fury":"The Sound and the Fury"}
METHODS = {"span":"Span writer","single":"Single glyph","exchange":"Exchanged","qwen":"Text Qwen","reference":"Reference"}
ASSETS = FIG/"span_rollout"


def plate(records, grades):
    ASSETS.mkdir(exist_ok=True)
    items=[]
    y=1
    with ZipFile(ARCHIVE) as stream:
        for accepted,heading in ((True,"First credited span translation"),(False,"First rejected span translation")):
            i=next(i for i,v in enumerate(grades) if v["span"]==accepted)
            row=records[i]
            items.append(dict(type="heading",text=heading,meta=NAMES[row["work"]]+f"   /   {i:04d}",y=y))
            y+=19
            for name,label in (("source","Input"),("single","Single glyph"),("span","Span writer")):
                raw=stream.read(f"{i:04d}/{name}.png")
                file=ASSETS/f"{i:04d}-{name}.png"
                file.write_bytes(raw)
                with Image.open(BytesIO(raw)) as img: w,h=img.size
                height=h*0.39
                items.append(dict(type="strip",name=file.name,x=84,y=y,w=w*0.39,h=height,label=label,sha256=sha(file)))
                y+=max(18,height+4)
            y+=6
    render_plate(items,y,"output","plate")


def render_plate(items,y,name,receipt):
    width=492
    filename=FIG/f"Span_translation_{name}.pdf"
    c=Canvas(str(filename),pagesize=(width,y))
    for item in items:
        top=y-item["y"]
        c.setFillColor(HexColor("#1d332e"))
        if item["type"]=="heading":
            c.setFont("Helvetica-Bold",11)
            c.drawString(0,top-10,item["text"])
            c.setFont("Helvetica",8)
            c.drawRightString(width,top-10,item["meta"])
        else:
            c.setFont("Helvetica",9)
            c.drawString(4,top-10,item["label"])
            c.drawImage(str(ASSETS/item["name"]),item["x"],top-item["h"],item["w"],item["h"])
    c.showPage()
    c.save()
    dump(PUB/(receipt+".json"),dict(width=width,height=y,items=items))


def origins_plate():
    archive=ARCHIVE.with_name("word_origin_probes.zip")
    items=[]
    y=1
    with ZipFile(archive) as z:
        records=json.loads(z.read("records.json"))
        for kind,title,parts in [
            ("English origin","Origin of an English word",[("input","Question"),("span","Span writer")]),
            ("historical forms","Historical forms of a Chinese headword",[("input","Prompt and forms"),("requested","Requested order"),("span","Span writer")])]:
            i=next(j for j,r in enumerate(records) if r["kind"]==kind)
            items.append(dict(type="heading",text=title,meta="Origin probe   /   "+str(i),y=y))
            y+=19
            for part,label in parts:
                raw=z.read(f"{i:03d}/{part}.png")
                file=ASSETS/f"probe-{i:03d}-{part}.png"
                file.write_bytes(raw)
                with Image.open(BytesIO(raw)) as im: w,h=im.size
                scale=.65 if part=="requested" else .39
                items.append(dict(type="strip",name=file.name,x=84,y=y,w=w*scale,h=h*scale,label=label,sha256=sha(file)))
                y+=max(18,h*scale+4)
            y+=6
    render_plate(items,y,"origins","origin-plate")


def matrices(records, grades):
    blocks=[]
    fig,axes=plt.subplots(2,4,figsize=(10.2,3.35),sharex=True,sharey=True)
    for t,lang in enumerate(("en","zh")):
        for j,(work,title) in enumerate(NAMES.items()):
            ids=[r["case"] for r in records if r["direction"]==lang and r["work"]==work]
            matrix=np.array([[int(grades[i][m]) for i in ids] for m in METHODS])
            assert matrix.shape==(5,24)
            ax=axes[t,j]
            ax.imshow(matrix,aspect="auto",cmap=ListedColormap([WHITE,GREEN]),vmin=0,vmax=1,interpolation="nearest")
            ax.set_xticks([0,11,23],["1","12","24"])
            ax.set_yticks(range(5),list(METHODS.values()))
            ax.tick_params(axis="both",length=0,labelsize=9,pad=3)
            ax.set_xticks(np.arange(23)+.5,minor=True)
            ax.set_yticks(np.arange(4)+.5,minor=True)
            ax.grid(which="minor",color="white",linewidth=.38)
            ax.tick_params(which="minor",length=0)
            for side in ax.spines.values(): side.set_visible(False)
            if t==0: ax.set_title(title,fontsize=10,loc="left",weight="bold")
            if j==0: ax.set_ylabel(("To English","To Chinese")[t],weight="bold",fontsize=10)
            blocks.append(dict(work=title,language=lang,ids=ids,values=matrix.tolist()))
    fig.subplots_adjust(left=.125,right=.995,bottom=.1,top=.89,hspace=.20,wspace=.12)
    fig.savefig(FIG/"Span_translation_cases.pdf")
    plt.close(fig)
    dump(PUB/"blocks.json",dict(methods=METHODS,blocks=blocks))


def interface_plot():
    data=read(PUB/"interface-votes.json")["records"]["directions"]
    keys=["native","pictured","span"]
    names=["Text input, text head","Image input, text head","Image input, span writer"]
    fig,ax=plt.subplots(figsize=(8.7,2.65))
    x=np.arange(3)
    for j,(lang,color) in enumerate((("en",GREEN),("zh","#956445"))):
        nums=[data[lang].get(k,0) for k in keys]
        bars=ax.barh(x+(j-.5)*.30,nums,height=.27,label=("To English","To Chinese")[j],color=color)
        ax.bar_label(bars,padding=4,fontsize=10)
    ax.invert_yaxis()
    ax.set_yticks(x,names)
    ax.set_xlim(0,98)
    ax.set_xlabel("Complete meanings credited among 96 requests")
    ax.legend(ncol=2,loc="lower right",frameon=False)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    fig.subplots_adjust(left=.28,right=.97,top=.98,bottom=.22)
    fig.savefig(FIG/"Span_translation_interfaces.pdf")
    plt.close(fig)
    dump(PUB/"interface-chart.json",dict(labels=names,keys=keys,series={lang:[data[lang].get(k,0) for k in keys] for lang in ("en","zh")}))


def main():
    with ZipFile(ARCHIVE) as z: records=json.loads(z.read("records.json"))
    grades=read(PUB/"assessment.json")["primary"]["records"]
    plate(records,grades)
    matrices(records,grades)
    interface_plot()
    origins_plate()
    for name in ("output","cases","interfaces","origins"):
        path=FIG/f"Span_translation_{name}.pdf"
        with fitz.open(path) as doc: doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(path.with_suffix(".png"))


if __name__=="__main__":
    main()

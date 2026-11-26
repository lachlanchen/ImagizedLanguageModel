"""Draw the matched reader measurements and the raw-pixel motor's held translations."""
from pathlib import Path
from shutil import copyfile
import numpy as np
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor

from scripts.figure_book_breadth import FIG, plt
from scripts.train_readout_objective import PUB, RUN, ROOT, read, dump, sha
from scripts.evaluate_neural_glyph_motor import EVID, OUT
from scripts.write_neural_glyphs import MODEL

ASSETS=FIG/"writer_provenance"


def chart():
    data=read(PUB/"evaluation.json")
    keys=["start","regression","prediction"]
    names=["Starting reader","Regression","Prediction loss"]
    kl=dict(title="a   Error in the next-piece distribution",unit="KL divergence (nats per decision)",
            labels=names,series={l:[round(data["dev"][k]["full_kl"][a],4) for a in keys] for k,l in (("en","To English"),("zh","To Chinese"))})
    methods=keys+["native","ilm","qwen"]
    lang=dict(title="b   Complete translations from new books",unit="Credited answers out of 96 per direction",
              labels=[*names,"Native teacher","Independent ILM","Quantized Qwen"],
              series={l:[int(data["source"]["directions"][k].get(a,0)) for a in methods] for k,l in (("en","To English"),("zh","To Chinese"))})
    colors={"To English":"#167263","To Chinese":"#b87844"}
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.6),gridspec_kw={"width_ratios":[.9,1.25]})
    for ax,panel in zip(axes,(kl,lang),strict=True):
        y=np.arange(len(panel["labels"]))
        for j,(l,values) in enumerate(panel["series"].items()):
            bars=ax.barh(y+(j-.5)*.37,values,.34,color=colors[l],label=l)
            ax.bar_label(bars,fmt="%.2f" if panel is kl else "%.0f",padding=2,fontsize=8)
        ax.set_title(panel["title"],loc="left",fontsize=10.5,weight="bold")
        ax.set_xlabel(panel["unit"],fontsize=8)
        ax.set_yticks(y,panel["labels"],fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0,1.24 if panel is kl else 100)
        ax.set_axisbelow(True)
        ax.grid(axis="x",linewidth=.45,color="#d7e2de")
        ax.tick_params(length=0)
        ax.spines[["top","right","left"]].set_visible(False)
    axes[1].legend(fontsize=8,loc="upper right",frameon=False)
    fig.subplots_adjust(left=.12,right=.985,top=.89,bottom=.16,wspace=.62)
    for ext in ("pdf","png"):
        fig.savefig(FIG/("Figure_216_prediction_sensitive_reading."+ext),dpi=180)
    plt.close(fig)
    dump(PUB/"chart.json",dict(panels=[kl,lang],colors=colors))


def motor():
    records=read(EVID/"evaluation.json")["records"]
    cases=[next(r for r in records if r["source_only_credit"] is True),
           next(r for r in records if r["source_only_credit"] is False)]
    width=512
    y=4
    items=[]
    for r,title in zip(cases,("a   First complete meaning retained","b   First rejected translation retained"),strict=True):
        items.append(dict(type="text",text=title,x=3,y=y,size=13,bold=True))
        y+=23
        k=f'{r["index"]:04d}'
        for label,path,ds in (("Pictured input",RUN/"pixels"/k/"input.png",r["input_png"]),
                              ("Neural pixel output",OUT/(k+".png"),r["output_png"])):
            if sha(path)!=ds: raise ValueError("Changed motor figure")
            name="motor-"+k+"-"+path.stem+".png"
            copyfile(path,ASSETS/name)
            w,h=Image.open(path).size
            sw=w/32*12.4
            sh=h/32*12.4
            items+=[dict(type="text",x=3,y=y+2,size=8,text=label,bold=True),
                    dict(type="image",x=108,y=y,w=sw,h=sh,name=name)]
            y+=max(22,sh+7)
        y+=7
    doc=FIG/"Figure_217_neural_pixel_output.pdf"
    c=Canvas(str(doc),pagesize=(width,y),pageCompression=1,invariant=1)
    for v in items:
        if v["type"]=="text":
            c.setFont("Helvetica-Bold" if v["bold"] else "Helvetica",v["size"])
            c.setFillColor(HexColor("#25413e"))
            c.drawString(v["x"],y-v["y"]-v["size"],v["text"])
        else: c.drawImage(str(ASSETS/v["name"]),v["x"],y-v["y"]-v["h"],v["w"],v["h"])
    c.showPage()
    c.save()
    dump(EVID/"scene.json",dict(width=width,height=y,items=items,examples=[r["index"] for r in cases],weights=sha(MODEL/"weights.safetensors")))


if __name__=="__main__":
    chart()
    motor()

"""Plot conserved optical attention, reader learning and held free-translation measurements."""

import numpy as np

from scripts.figure_book_breadth import FIG, plt
from scripts.prepare_teacher_translation import dump
from scripts.sample_visual_library import ROOT, read

PUB=ROOT/"references/evidence/optical_state_v1"
TPUB=ROOT/"references/evidence/library_teacher_v1"
COLORS={"Replicated":"#91a2a4","Mass corrected":"#087c69","Teacher":"#a97741","Visual writer":"#466785",
        "Chinese source":"#0c7769","English source":"#8e643f","English output":"#53748b","Chinese output":"#087969"}


def export(stem, panels):
    dump(PUB/(stem+".json"),dict(panels=panels,colors=COLORS))
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.3))
    for ax,p in zip(axes,panels,strict=True):
        x=np.arange(len(p["labels"]))
        for j,(name,values) in enumerate(p["series"].items()):
            if p["kind"]=="line":
                ax.plot(x,values,color=COLORS[name],label=name,linewidth=2,marker="o",markersize=3)
            else:
                bars=ax.bar(x+(j-.5)*.34,values,.31,color=COLORS[name],label=name)
                ax.bar_label(bars,fmt="%.0f",fontsize=8,padding=2)
        ax.set(xticks=x,xticklabels=p["labels"],ylabel=p["unit"],ylim=(0,p["upper"]))
        ax.set_title(p["title"],loc="left",fontsize=10,weight="bold")
        ax.tick_params(length=0,labelsize=8)
        ax.spines[["left","top","right"]].set_visible(False)
        ax.grid(axis="y",linewidth=.6,color="#e2e8e5")
        ax.set_axisbelow(True)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.13),ncol=2,fontsize=8,frameon=False)
    fig.subplots_adjust(left=.065,right=.992,top=.9,bottom=.25,wspace=.25)
    for ext in ("pdf","png"):
        fig.savefig(FIG/(stem+"."+ext),dpi=230)
    plt.close(fig)


def main():
    conserve=read(TPUB/"glyph-measure.json")["results"]
    assign=read(TPUB/"assignments.json")["directions"]
    panel1=dict(title="a    Reproducing the native translation",labels=["English output","Chinese output"],
                unit="Identical complete sequences out of 240",upper=270,kind="bar",series={})
    for a,name in (("copies","Replicated"),("corrected","Mass corrected")):
        panel1["series"][name]=[conserve[l]["conditions"][a]["equal_free"] for l in ("en","zh")]
    panel2=dict(title="b    Detecting a subject–object exchange",labels=["English output","Chinese output"],
                unit="Correct complete assignments out of 472",upper=530,kind="bar",series={})
    for k,n in (("teacher_correct","Teacher"),("student_correct","Visual writer")):
        panel2["series"][n]=[assign[l]["result"]["exchange"][k] for l in ("en","zh")]
    export("Figure_213_glyph_attention_measure",[panel1,panel2])
    ev=read(PUB/"evaluation.json")
    p=dict(title="a    Predicting contextual source states",labels=[str(i+1) for i in range(12)],
           unit="Held normalized mean squared error",upper=1,kind="line",series={})
    for l,n in (("en","Chinese source"),("zh","English source")):
        hist=read(ROOT/"models/optical_state"/l/"fit.json")["history"]
        p["series"][n]=[round(h["dev_state_mse"],5) for h in hist]
    q=dict(title="b    Reading the source through a fixed decoder",labels=["Exchanged","Learned","Native","ILM","Qwen"],
           unit="Faithful translations out of 48 per direction",upper=50,kind="bar",series={})
    for l,n in (("en","English output"),("zh","Chinese output")):
        q["series"][n]=[ev["languages"][l].get(m,0) for m in ("exchanged","probed","native","pure","qwen")]
    export("Figure_214_optical_state_reader",[p,q])


if __name__=="__main__":
    main()

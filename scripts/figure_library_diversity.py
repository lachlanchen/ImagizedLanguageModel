"""Typeset raw emitted PNGs from three distinct held-book comparisons."""

import numpy as np
import torch
from scripts.figure_book_breadth import FIG,plt
from scripts.figure_book_rehearsal import plate
from scripts.evaluate_visual_library import RUN,EVIDENCE,read,dump,sha,WRITERS
from scripts.build_library_memory import PUB as MPUB,RUN as MEMORY
from scripts.write_ordered_translation import infer,restore
from ilm.visual_lm.portable_writer import read_image,save_grid

ASSETS=FIG/"library_diversity"
COLORS={"Starting writer":"#919d9a","Repeated passages":"#647b92","Distinct passages":"#067b69",
        "Control":"#647b92","Intervention":"#087969","Qwen Q4":"#9a612d"}


def main():
    torch.set_num_threads(2)
    ASSETS.mkdir(exist_ok=True)
    raw=read(RUN/"generated.json")
    contrast=read(RUN/"contrast-generated.json")
    mem=read(MEMORY/"generated.json")
    base=read(EVIDENCE/"evaluation.json")
    adjust=read(EVIDENCE/"contrast.json")
    recall=read(MPUB/"evaluation.json")
    restored={m:restore(n,"cpu") for m,n in WRITERS.items()}
    checks=[]
    def display(stem,title,cases):
        groups=[]
        for i,heading,methods in cases:
            root=RUN/"pixels"/f"{i:04d}"
            r=raw["rows"][i]
            files=[("Pictured source",f"request-{i}-input",root/"input.png")]
            inp=read_image(root/"input.png",len(r["input"]))
            for m in methods:
                source=raw if m in WRITERS else contrast if m=="contrast" else mem
                name={"start":"Starting writer","repeat":"Repeated passages","broad":"Ordinary writer","contrast":"Source contrast","associative":"Associative recall"}[m]
                p=(root if m!="associative" else MEMORY/"pixels"/f"{i:04d}")/(m+".png")
                out=source["answers"][i][m]
                assert sha(p)==out["png"]
                mode={"associative":"associative","contrast":"background-half"}.get(m,"greedy")
                with torch.inference_mode():
                    pixels,done=infer(restored.get(m,restored["broad"]),inp,r["t"],"cpu",mode)
                tmp=RUN/f"paper-{i}-{m}.png"
                save_grid(pixels,tmp)
                checks.append(dict(request=i,method=m,cpu_matches=(sha(tmp)==sha(p) and done==out["stop"]),
                                   original=sha(p),cpu=sha(tmp),cpu_stopped=done,original_stopped=out["stop"]))
                files.append((name,f"request-{i}-{m}",p))
            groups.append((heading,files))
        plate(stem,title,groups,assets=ASSETS,evidence=EVIDENCE)
    display("natural-rasters","Figure_207_writing_from_more_books",[
        (740,"a    A question about a person",("start","repeat","broad")),
        (0,"b    The first reserved literary sentence",("broad",)),
    ])
    display("contrast-rasters","Figure_209_writing_with_source_contrast",[
        (278,"a    An English article changes",("broad","contrast")),
        (adjust["panel"][0],"b    First request in the contrast assessment",("broad","contrast")),
    ])
    panel=recall["panel"]
    records=recall["source_only"]["records"]
    improved=[i for i,r in zip(panel,records,strict=True) if r["associative"] and not r["ordinary"]]
    changed=[i for i in panel if mem["answers"][i]["associative"]["cells"]!=mem["answers"][i]["ordinary"]["cells"]]
    first=(improved or changed or panel)[0]
    second=next(i for i in panel if i!=first)
    display("memory-rasters","Figure_210_writing_from_nearby_states",[
        (first,"a    "+("First gained semantic credit" if improved else "First changed answer"),("broad","associative")),
        (second,"b    A further reserved request",("broad","associative")),
    ])
    dump(EVIDENCE/"illustrated.json",dict(checks=checks,memory_requests=[first,second]))
    a=dict(title="a    Grammatical control and natural writing",labels=["Role families (118)","Whole meanings (96)"],
           unit="Number credited",series={})
    for m,n in (("start","Starting writer"),("repeat","Repeated passages"),("broad","Distinct passages")):
        a["series"][n]=[base["optical"][m]["role_families"],base["source_only"]["total"][m]]
    b=dict(title="b    Complete meaning on new passages",labels=["More passages","Source contrast","Associative recall"],
           unit="Source-only credits per 96 requests",series={})
    results=(base,adjust,recall)
    for name,methods in (("Control",("repeat","ordinary","ordinary")),("Intervention",("broad","contrast","associative")),
                         ("Qwen Q4",("qwen","qwen","qwen"))):
        b["series"][name]=[r["source_only"]["total"].get(m,0) for r,m in zip(results,methods,strict=True)]
    dump(EVIDENCE/"charts.json",dict(panels=[a,b],colors=COLORS))
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.5))
    for ax,p,lim in zip(axes,(a,b),(120,100),strict=True):
        x=np.arange(len(p["labels"]))
        for j,(name,values) in enumerate(p["series"].items()):
            bar=ax.bar(x+(j-1)*.26,values,.23,label=name,color=COLORS[name])
            ax.bar_label(bar,fontsize=8,padding=2)
        ax.set(ylim=(0,lim),xticks=x,xticklabels=p["labels"],ylabel=p["unit"])
        ax.set_title(p["title"],loc="left",weight="bold",fontsize=11)
        ax.tick_params(labelsize=8,length=0)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="y",color="#dde5e2",linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.10),ncol=2,fontsize=8,frameon=False)
    fig.subplots_adjust(left=.074,right=.985,bottom=.27,top=.88,wspace=.32)
    for ext in ("pdf","png"):
        fig.savefig(FIG/("Figure_208_training_and_free_meaning."+ext),dpi=230)
    plt.close(fig)


if __name__=="__main__":
    main()

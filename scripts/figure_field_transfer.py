"""Plot the optical-path interventions and replay both writers from the actual pictured inputs."""
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.field_to_glyph import restore
from scripts.figure_book_breadth import FIG,plt
from scripts.figure_book_rehearsal import plate
from scripts.reserve_field_transfer import PUB, RUN, read, dump, sha
from scripts.evaluate_field_to_glyph import requests
from scripts.write_neural_glyphs import grid_bits,save_ink

ASSETS=FIG/"field_transfer"


def charts(primary):
    methods=["parent","regression","prediction","exchange","qwen"]
    names=["Original ILM","Regression field","Prediction field","Exchanged input","Qwen 8B Q4"]
    panel=dict(title="a   Complete literary translations",unit="Credited answers per 96 requests",
               labels=names,series={l:[primary["source"]["directions"][k].get(m,0) for m in methods]
                                     for l,k in (("To English","en"),("To Chinese","zh"))})
    audit=read(PUB/"paths.json")["conditions"]["prediction"]
    components=dict(title="b   Marginal contribution of each source path",unit="Gain in log probability (nats per glyph or stop)",
                    labels=["Original image path","Learned field"],series={l:[round(audit[k]["per_position"][x],4)
                          for x in ("base_gain","field_gain")] for l,k in (("To English","en"),("To Chinese","zh"))})
    colors={"To English":"#087260","To Chinese":"#b37744"}
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.45))
    for ax,p in zip(axes,(panel,components),strict=True):
        ix=np.arange(len(p["labels"]))
        for j,(name,values) in enumerate(p["series"].items()):
            bars=ax.barh(ix+(j-.5)*.37,values,.34,color=colors[name],label=name)
            ax.bar_label(bars,fmt="%.0f" if p is panel else "%.3f",fontsize=8,padding=3)
        ax.invert_yaxis()
        ax.set_yticks(ix,p["labels"],fontsize=8)
        ax.set_title(p["title"],loc="left",fontsize=10,weight="bold")
        ax.set_xlabel(p["unit"],fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="x",linewidth=.5,color="#d2dedc")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(length=0,labelsize=8)
        ax.set_xlim(0,100 if p is panel else 1.87)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=8,frameon=False)
    fig.subplots_adjust(left=.135,right=.975,top=.88,bottom=.22,wspace=.66)
    for ext in ("png","pdf"): fig.savefig(FIG/("Figure_218_field_transfer."+ext),dpi=180)
    plt.close(fig)
    dump(PUB/"chart.json",dict(colors=colors,panels=[panel,components]))


def examples(primary):
    rows=requests()
    decisions=primary["source"]["records"]
    data=read(RUN/"raw.json")["answers"]
    changed=next((i for i,r in enumerate(decisions) if r["regression"]!=r["prediction"]),None)
    good=next((i for i,r in enumerate(decisions) if r["prediction"]),None)
    fail=next(i for i,r in enumerate(decisions) if not r["prediction"])
    ids=list(dict.fromkeys(i for i in (good,changed,fail) if i is not None))
    nets={m:restore(m,"cpu")[0] for m in ("regression","prediction")}
    scenes=[]
    replay=[]
    ASSETS.mkdir(exist_ok=True)
    for j,i in enumerate(ids):
        folder=RUN/"png"/f"{i:04d}"
        x=torch.tensor(grid_bits(folder/"input.png",len(rows[i]["input"])))[None]
        files=[("Pictured source",f"{i:04d}-input",folder/"input.png")]
        for mode,net in nets.items():
            with torch.inference_mode():
                ink,n,stop=net.write(x,torch.tensor([x.shape[1]]),rows[i]["t"])
            v=data[i][mode]
            with TemporaryDirectory() as temp:
                p=Path(temp)/"answer.png"
                save_ink(ink[0,:int(n)].cpu().numpy(),p)
                if sha(p)!=v["png"] or sha(folder/(mode+".png"))!=v["png"] or bool(stop)!=v["stop"]:
                    raise ValueError(f"CPU replay mismatch {i} {mode}")
            files.append((mode.capitalize()+" field",f"{i:04d}-"+mode,folder/(mode+".png")))
            replay.append(dict(index=i,mode=mode,source_sha256=sha(folder/"input.png"),png=v["png"],cells=int(n),stopped=bool(stop)))
        title="First credited prediction-field answer" if i==good else "First change in whole-meaning credit" if i==changed else "First rejected prediction-field answer"
        scenes.append((chr(97+j)+"    "+title,files))
    plate("scene","Figure_219_field_answers",scenes,assets=ASSETS,evidence=PUB)
    dump(PUB/"illustrated.json",dict(ids=ids,first_credit=good,first_differing_credit=changed,first_failure=fail,replays=replay))


if __name__=="__main__":
    torch.set_num_threads(2)
    result=read(PUB/"evaluation.json")
    charts(result)
    examples(result)

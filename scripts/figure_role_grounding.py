"""Plot source-role attention and show actual, independently reproduced book answers."""

import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.evaluate_role_grounding import PUB, RUN, WRITERS, read, sha
from scripts.figure_book_breadth import FIG, plt
from scripts.figure_book_rehearsal import plate
from scripts.prepare_teacher_translation import dump
from scripts.write_ordered_translation import infer, restore

NAMES = {"start":"Starting writer", "likelihood":"Likelihood", "shuffled":"Shuffled roles", "grounded":"True roles"}
ASSETS = FIG / "library_roles"


def draw_chart(report):
    colors = {"Observed":"#07796c","Permuted expectation":"#a8b9b7",
              "Source only":"#07796c","With reference":"#ad7842"}
    names = list(NAMES.values())
    dev = report["development"]
    att = dict(title="a    Attention to the matching role", unit="Mean probability (%)", labels=names,
               series={label:[round(100*dev[m]["total"][k], 2) for m in NAMES]
                       for k,label in (("observed","Observed"),("expected","Permuted expectation"))})
    sem = dict(title="b    Faithful translations", unit="Complete meanings out of 384", labels=names,
               series={label:[report[group]["total"][m] for m in NAMES]
                       for group,label in (("source_only","Source only"),("reference_informed","With reference"))})
    fig, axes = plt.subplots(1,2,figsize=(10.4,3.3))
    for ax, panel in zip(axes, (att, sem), strict=True):
        x = np.arange(len(names))
        for j, (name, vals) in enumerate(panel["series"].items()):
            bars = ax.bar(x+(j-.5)*.36, vals, .33, label=name, color=colors[name])
            ax.bar_label(bars, fmt="%.1f" if panel is att else "%.0f", fontsize=8, padding=2)
        upper = 1.28*max(1, max(max(a) for a in panel["series"].values()))
        panel["upper"] = float(np.ceil(upper))
        ax.set(xticks=x, xticklabels=names, ylim=(0,panel["upper"]), ylabel=panel["unit"])
        ax.set_title(panel["title"], loc="left",fontsize=11,weight="bold")
        ax.tick_params(labelsize=8,length=0)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e8e6", linewidth=.6)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.1),frameon=False,ncol=2,fontsize=8)
    fig.subplots_adjust(left=.067,right=.993,bottom=.23,top=.89,wspace=.24)
    for ext in ("pdf","png"):
        fig.savefig(FIG/("Figure_211_role_learning."+ext),dpi=230)
    plt.close(fig)
    dump(PUB/"charts.json",dict(panels=[att,sem],colors=colors))


def raw_examples(report):
    gen = read(RUN/"generated.json")
    answers, rows = gen["answers"], gen["rows"]
    src = report["source_only"]["records"]
    ref = report["reference_informed"]["records"]
    successes = [i for i in range(len(src)) if src[i]["grounded"] and ref[i]["grounded"]]
    exact = [i for i in successes if answers[i]["grounded"]["exact"]]
    losses = [i for i in range(len(src)) if not src[i]["grounded"] and not ref[i]["grounded"]]
    good = (exact or successes or [0])[0]
    bad = next(i for i in losses if i!=good)
    checks, scenes = [], []
    models = {m:restore(WRITERS[m],"cpu") for m in ("likelihood","grounded")}
    first_title = "An exact translation" if exact else "First answer credited in both readings" if successes else "First reserved request"
    for i, title in ((good,"a    "+first_title), (bad,"b    First rejected answer")):
        prefix = f"request-{i}"
        parent = RUN/"pixels"/f"{i:04d}"
        files = [("Pictured source",prefix+"-input",parent/"input.png")]
        raster = read_image(parent/"input.png",len(rows[i]["input"]))
        for m, model in models.items():
            out = answers[i][m]
            p = parent/(m+".png")
            with torch.inference_mode():
                bits, stop = infer(model,raster,rows[i]["t"],"cpu")
            verified = RUN/(prefix+"-"+m+"-cpu.png")
            save_grid(bits, verified)
            if sha(p)!=out["png"] or sha(verified)!=out["png"] or stop!=out["stop"]:
                raise ValueError("CPU independently produced another answer")
            checks.append(dict(request=i, method=m, png=out["png"], cpu_png=sha(verified), stopped=stop))
            files.append((NAMES[m],prefix+"-"+m,p))
        scenes.append((title,files))
    ASSETS.mkdir(exist_ok=True)
    plate("rasters","Figure_212_words_and_roles",scenes,assets=ASSETS,evidence=PUB)
    dump(PUB/"illustrated.json",dict(first=good,failure=bad,exact=bool(exact),both_credited=len(successes),checks=checks))


if __name__ == "__main__":
    torch.set_num_threads(2)
    evaluation = read(PUB/"evaluation.json")
    draw_chart(evaluation)
    raw_examples(evaluation)

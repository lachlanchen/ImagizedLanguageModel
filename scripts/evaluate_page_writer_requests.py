"""Test whether varied wording preserves a generated fact when years or questions change."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import generate,score,tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.grounded_proposition_corpus import noun
from scripts.prepare_visual_facts import Face,FONTS
from scripts.train_causal_glyph_writer import in_cells,record,sha
from scripts.train_page_writer_requests import DEFAULT,VARIED,NEW,load_records
from scripts.train_span_choice import gpu_owner


def evaluate(parent,folder):
    torch.set_num_threads(2)
    dest=folder/"evaluation"
    dest.mkdir(exist_ok=False)
    face=Face((FONTS[0],2))
    fresh=load_records(folder,"fresh")
    held=load_records(folder,"held")
    metrics={}
    for arm,run in (("one_instruction",parent),("varied",folder)):
        weights=run/"page"
        if any(sha(weights/n)!=h for n,h in json.loads((run/"page_fit.json").read_text())["weights"].items()):
            raise ValueError("Writer weights changed")
        if arm=="varied":
            outputs=generate(weights,held,"page",face)
            store_streams(dest/"previous_held.npz",[y for y,_ in outputs],[s for _,s in outputs])
            metrics["previous_held"]=score(held,outputs,face)
        passed=defaultdict(list)
        all_trials=[]
        for k,pattern in enumerate(NEW):
            cases=[dict(v,q=pattern.format(noun(v["s"],v["r"]))) for v in fresh]
            outputs=generate(weights,cases,"page",face)
            path=dest/f"{arm}_{k}.npz"
            store_streams(path,[y for y,_ in outputs],[s for _,s in outputs])
            groups=score(cases,outputs,face)
            all_trials.append(dict(groups=groups,sha256=sha(path)))
            for v,(y,s) in zip(cases,outputs,strict=True):
                passed[v["task"],v["board"]].append(s and y==face(v["y"]))
            if arm=="varied" and k==0:
                save_examples(dest,cases,outputs,face)
        counts={}
        for task,expected in (("single",6),("dual",12)):
            boards=[xs for (t,_),xs in passed.items() if t==task]
            if not all(len(v)==expected for v in boards):
                raise ValueError("Missing counterfactual output")
            counts[task]=dict(n=len(boards),all_wordings=sum(all(v) for v in boards),
                              minimum=min(r["groups"][task]["complete"] for r in all_trials))
        metrics[arm]=dict(wordings=all_trials,boards=counts)
        print(arm,counts,flush=True)
    record(dest/"results.json",metrics)


def save_examples(dest,cases,outputs,face):
    exact=[s and y==face(v["y"]) for v,(y,s) in zip(cases,outputs,strict=True)]
    first=next((i for i,v in enumerate(cases[:-1]) if v["task"]=="single" and v["edit"]==0
                and exact[i] and exact[i+1]),0)
    failure=next((i for i,v in enumerate(cases) if v["task"]=="dual" and not exact[i]),None)
    samples={"original":first,"edit":first+1}
    if failure is not None:
        samples["failure"]=failure
    legend={face.cell(c):c for v in cases for txt in (v["p"],v["q"],v["y"]) for c in txt}
    rows={}
    for key,i in samples.items():
        v=cases[i]
        raw,stop=outputs[i]
        rows[key]=dict(index=i,board=v["board"],question=v["q"],reference=v["y"],
                       reading="".join(legend.get(c,"□") for c in in_cells(raw)),stopped=stop,exact=exact[i])
        for field,ink in (("page",face(v["p"])),("question",face(v["q"])),("output",raw),
                          ("reference",face(v["y"]))):
            write_strip(tensor(ink).numpy(),dest/(key+"_"+field+".png"))
    record(dest/"examples.json",rows)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run",type=Path,default=DEFAULT)
    ap.add_argument("--run-dir",type=Path,default=VARIED)
    args=ap.parse_args()
    with gpu_owner():
        evaluate(args.base_run,args.run_dir)

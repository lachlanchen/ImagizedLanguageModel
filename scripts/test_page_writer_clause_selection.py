"""Separate selective quotation from repeating an entire one-sentence passage."""
import argparse
import json
from pathlib import Path
from collections import Counter
import torch

from scripts.evaluate_page_evidence_writer import generate
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.page_evidence_corpus import fits
from scripts.prepare_visual_facts import Face,FONTS
from scripts.span_reservation import reservation
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_page_writer_requests import DEFAULT,VARIED,load_records
from scripts.train_span_choice import gpu_owner


def paired_clauses(held):
    remaining=[x for x in held if x["task"]=="clause"]
    boards=[]
    while remaining:
        a=remaining.pop(0)
        for i,b in enumerate(remaining):
            if (a["s"] in b["p"] or b["s"] in a["p"] or a["o"] in b["p"] or b["o"] in a["p"]
                    or a["y"]==b["y"] or a["q"]==b["q"]):
                continue
            take=[dict(v,p=p) for p in (a["p"]+"　"+b["p"],b["p"]+"　"+a["p"]) for v in (a,b)]
            if all(fits(x) and not reservation().overlaps(x["p"]) for x in take):
                boards.extend(take)
                remaining.pop(i)
                break
    return boards


def run(parent,folder):
    torch.set_num_threads(2)
    face=Face((FONTS[0],2))
    data=paired_clauses(load_records(folder,"held"))
    dest=folder/"clause_selection"
    dest.mkdir(exist_ok=False)
    result={}
    for arm,run in (("one",parent),("varied",folder)):
        if any(sha(run/"page"/n)!=h for n,h in json.loads((run/"page_fit.json").read_text())["weights"].items()):
            raise ValueError("Writer weights changed")
        ys=generate(run/"page",data,"page",face)
        stats=Counter()
        for i in range(0,len(data),4):
            exact=[]
            for j in range(i,i+4):
                y,stop=ys[j]
                ok=y==face(data[j]["y"]) and stop
                exact.append(ok)
                stats["requested_first" if j%4 in (0,3) else "requested_second"]+=int(ok)
                stats["rival_clause"]+=int(stop and y==face(data[j^1]["y"]))
                stats["stopped"]+=int(stop)
            stats["all_four"]+=int(all(exact))
        path=dest/(arm+".npz")
        store_streams(path,[y for y,_ in ys],[s for _,s in ys])
        result[arm]=dict(stats,boards=len(data)//4,questions=len(data),streams=sha(path))
    record(dest/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run",type=Path,default=DEFAULT)
    ap.add_argument("--run-dir",type=Path,default=VARIED)
    args=ap.parse_args()
    with gpu_owner():
        run(args.base_run,args.run_dir)

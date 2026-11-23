"""Recount stopped, source-sensitive answers directly from saved physical streams."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.page_evidence_corpus import DEFAULT
from scripts.prepare_visual_facts import Face,FONTS
from scripts.span_reservation import normalized
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_page_writer_requests import VARIED,load_records


def clause_baseline(held,path,face):
    raw,ends=load_streams(path)
    stats=Counter()
    if len(raw)!=len(held):
        raise ValueError("Physical output count differs")
    for v,y,stopped in zip(held,raw,ends,strict=True):
        if v["task"]!="clause":
            continue
        stats["requests"]+=1
        page=v["p"].strip(" ，,：:。；;！？\n")+"。"
        echo=page==v["y"]
        exact=stopped and y==face(v["y"])
        stats["whole_page_with_terminal_period"]+=int(echo)
        stats["exact"]+=int(exact)
        stats["exact_on_strict_subclause"]+=int(exact and not echo)
        stats["strict_subclause"]+=int(not echo)
    return dict(stats)


def audit(parent,folder):
    face=Face((FONTS[0],2))
    held=load_records(folder,"held")
    fresh=load_records(folder,"fresh")
    training=load_records(folder,"training")
    groups={"subjects":len({normalized(v["s"]) for v in fresh} & {normalized(v["s"]) for v in training}),
            "passages":len({normalized(v["p"]) for v in fresh} & {normalized(v["p"]) for v in training})}
    if any(groups.values()):
        raise ValueError("New evaluation overlaps writer training")
    result=json.loads((folder/"evaluation/results.json").read_text())
    recount={}
    for arm in ("one_instruction","varied"):
        counts=defaultdict(list)
        for k,expected in enumerate(result[arm]["wordings"]):
            file=folder/"evaluation"/f"{arm}_{k}.npz"
            if sha(file)!=expected["sha256"]:
                raise ValueError("Generated pixels changed")
            images,stop=load_streams(file)
            good=Counter()
            for v,im,s in zip(fresh,images,stop,strict=True):
                exact=s and im==face(v["y"])
                counts[v["task"],v["board"]].append(exact)
                good[v["task"]]+=int(exact)
            if any(n!=expected["groups"][g]["exact"] for g,n in good.items()):
                raise ValueError("Question-level score differs")
        rec={task:sum(all(v) for (g,_),v in counts.items() if g==task) for task in ("single","dual")}
        if any(n!=result[arm]["boards"][g]["all_wordings"] for g,n in rec.items()):
            raise ValueError("Whole-board score differs")
        recount[arm]=rec
    record(folder/"audit.json",dict(new_vs_training_overlap=groups,all_wording_boards=recount,
        clauses_one=clause_baseline(held,parent/"evaluation/page.npz",face),
        clauses_varied=clause_baseline(held,folder/"evaluation/previous_held.npz",face)))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run",type=Path,default=DEFAULT)
    ap.add_argument("--run-dir",type=Path,default=VARIED)
    args=ap.parse_args()
    audit(args.base_run,args.run_dir)

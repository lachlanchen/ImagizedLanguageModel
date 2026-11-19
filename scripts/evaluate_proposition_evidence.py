"""Compare endogenous, reference-assisted and mismatched evidence for visual statement writing."""
from collections import Counter
from dataclasses import replace
import fcntl
import json
import os
from opencc import OpenCC
import torch

from scripts.evaluate_human_proposition_questions import cases,alphabet
from scripts.evaluate_grounded_propositions import release_notes,assess,pictured,generate,forms,metrics,store_streams
from scripts.evaluate_pointing_causal_writer import reader_control
from scripts.grounded_proposition_corpus import Face,FONTS,WORK,PUBLIC,ROOT
from scripts.train_grounded_propositions import PointingCausalWriter,sha,record
from scripts.train_proposition_evidence import condition
from scripts.pointing_glyph_corpus import load
from scripts.train_causal_glyph_writer import in_cells


def main():
    human,hm,_,_=cases()
    notes,nm=release_notes()
    reg=[r for r in load()[1] if r.task in ("historical","historical_rephrased","single_span","distributed_ink","absent_content_ink")]
    face=Face((FONTS[0],2))
    real={}
    for name,rows in (("human",human),("notes",notes),("regression",reg)):
        real[name]=reader_control(rows)[0]
    norm=OpenCC("t2s")
    result={}
    for arm in ("evidence","wrongcue"):
        assert all(sha(WORK/arm/k)==v for k,v in json.loads((PUBLIC/(arm+"_fit.json")).read_text())["weights"].items())
        actor=PointingCausalWriter.restore(WORK/arm,"cuda");arm_stats={}
        for name,rows,meta in (("human",human,hm),("notes",notes,nm),("regression",reg,None)):
            opts={"predicted":real[name]}
            if meta is not None:
                oracle=[r.y for r in rows] if name=="human" else [face(m["facts"][0][2]) for m in meta]
                opts["oracle"]=oracle
            for cue,spans in opts.items():
                inputs=[replace(r,p=condition(r.p,e)) for r,e in zip(pictured(rows),spans)]
                y,ends=generate(actor,inputs)
                file=WORK/(arm+"_"+name+"_"+cue+".npz")
                store_streams(file,y,ends)
                stats={}
                if name=="human":
                    known=alphabet(actor,rows,meta)
                    cases_out=[]
                    for m,r,e,img,stop in zip(meta,rows,spans,y,ends):
                        text="".join(known.get(c,"□") for c in in_cells(img))
                        t=norm.convert(text).rstrip("。!?！")
                        good=stop and t in {norm.convert(v) for v in forms((m["s"],m["r"],m["o"]))}
                        rec=dict(key=m["key"],group=m["group"],quote_exact=e==r.y,reference=norm.convert(m["o"]) in t,statement=good,
                                 output=text)
                        cases_out.append(rec)
                        c=stats.setdefault(m["group"],Counter());c["n"]+=1
                        for k in ("quote_exact","reference","statement"):c[k]+=rec[k]
                    record(WORK/f"{arm}_human_{cue}_readings.json",cases_out)
                elif name=="notes":
                    stats=assess(rows,meta,y,ends,face)[0]
                    stats["quote_exact"]=sum(e==face(m["facts"][0][2]) for m,e in zip(meta,spans))
                else:stats=metrics(rows,y,ends)
                arm_stats[name+"_"+cue]=dict(scores=stats,sha256=sha(file))
                print(arm,name,cue,stats if name=="human" else {k:(v.get("exact") if isinstance(v,dict) else v) for k,v in stats.items()},flush=True)
        result[arm]=arm_stats
        del actor
    record(PUBLIC/"evidence_result.json",result)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);main()

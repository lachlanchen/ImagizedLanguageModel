"""Independently compare archived image streams and export fixed-order examples."""

import json
import numpy as np

from scripts.audit_interval_teacher import compare, save_image
from scripts.evaluate_interval_teacher import trials
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_typeface_writer import CONTROLS
from scripts.train_causal_glyph_writer import sha,record
from scripts.train_typeface_writer import PUBLIC, WORK
from scripts.typeface_writer_cases import sets

ARMS=("parent","map_only","replay","mapped")


def audit():
    cohorts,sources=sets()
    controls=trials()
    summary={}
    boards={}
    outcomes={}
    out=PUBLIC/"examples"
    out.mkdir(exist_ok=True,parents=True)
    for arm in ARMS:
        folder=WORK/"evaluation"/arm
        receipt=json.loads((folder/"streams.json").read_text())
        scores={}
        for key,digest in receipt["files"].items():
            path=folder/(key+".npz")
            assert sha(path)==digest
            images,stops=load_streams(path)
            if key in cohorts:
                cases=cohorts[key]
                marks=np.array([c.grade(y,end) for c,y,end in zip(cases,images,stops,strict=True)],bool)
                right=marks[:,2]
                score=dict(incoming=int(marks[:,0].sum()),canonical=int(marks[:,1].sum()))
            else:
                cases=controls[key]
                right=np.array([end and y in c.refs for c,y,end in zip(cases,images,stops,strict=True)])
                score={}
            by_board={}
            for c,ok in zip(cases,right,strict=True):
                by_board.setdefault(c.board,[]).append(bool(ok))
            both=np.array([all(v) for v in by_board.values()])
            boards[arm,key]=both
            outcomes[arm,key]=right
            scores[key]=dict(**score, n=len(cases),correct=int(right.sum()),complete=int(both.sum()),boards=len(both),stops=sum(stops),sha256=digest)
        summary[arm]=scores
        print(arm,{k:(v["correct"],v["complete"]) for k,v in scores.items()},flush=True)
    deltas={k:compare(boards["mapped",k],boards["replay",k]) for k in cohorts}
    exported={}
    for tag,by in [("gain",outcomes["mapped","trial_hana"]&~outcomes["replay","trial_hana"]),
                    ("success",outcomes["mapped","trial_hana"]),("error",~outcomes["mapped","trial_hana"])]:
        idx=np.flatnonzero(by)
        if not len(idx):
            continue
        i=int(idx[0]);c=cohorts["trial_hana"][i]
        exported[tag]=dict(index=i,id=c.question_id,board=c.board)
        for name,val in [("page",c.p),("question",c.q),("reference",c.source[0])]:
            save_image(out/(tag+"_"+name+".png"),val)
        for arm in ARMS:
            y,_=load_streams(WORK/"evaluation"/arm/"trial_hana.npz")
            save_image(out/(tag+"_"+arm+".png"),y[i])
    for i,tag in ((2,"yan_oracle"),(3,"yan_bronze")):
        c=controls["yan"][i]
        ys,_=load_streams(WORK/"evaluation/mapped/yan.npz")
        for name,data in (("page",c.p),("question",c.q),("reference",c.refs[0]),("output",ys[i])):
            save_image(out/(tag+"_"+name+".png"),data)
    s=summary["mapped"]
    selected=bool(s["hana"]["complete"]>max(summary[a]["hana"]["complete"] for a in ("parent","replay"))
                  and s["regular"]["complete"]>=11 and outcomes["mapped","yan"][2:].all() and s["historical"]["correct"]>=190)
    record(PUBLIC/"result.json",dict(scores=summary,paired=deltas,selected=selected,examples=exported,sources=sources))


if __name__=="__main__":
    audit()

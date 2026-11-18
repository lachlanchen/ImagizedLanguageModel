"""Recount the stored physical answers; keep all DuReader content on this workstation."""
from collections import Counter
from hashlib import sha256
import json

import numpy as np
from scripts.train_composed_answer_writer import DATA, OUT, ROOT, ARMS
from scripts.train_paired_passage import unpack_sequences
from scripts.prepare_pictured_entry_answers import load, sha, write_json


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(prev[j]+1,row[-1]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1]


def recount():
    path=DATA/"development"
    src=load(path/"sources.json")
    tiles=list(map(bytes,np.load(path/"pixels.npy")))
    ps,qs,ys=[[[tiles[int(i)] for i in seq] for seq in unpack_sequences(path/name)]
              for name in ("passages.npz","questions.npz","answers.npz")]
    train_tiles=set(map(bytes,np.load(DATA/"train/pixels.npy")))
    memory=set(map(bytes,np.load(OUT/"memory.npy")))
    execution=load(OUT/"execution.json")
    report={}
    for arm in ("released",*ARMS):
        archive=OUT/(arm+".npz")
        assert execution["archives"][arm]==sha(archive)
        with np.load(archive) as z:
            indexes=z["offsets"]
            assert len(indexes)==len(src)+1 and indexes[0]==0 and np.all(np.diff(indexes)>=0)
            assert len(z["pixels"])==indexes[-1]==len(z["origin"])
            streams=[list(map(bytes,s)) for s in np.split(z["pixels"],indexes[1:-1])]
            origins=np.split(z["origin"],indexes[1:-1])
            stops=z["stopped"]
        bag={}
        for i,(meta,out,actions,stop) in enumerate(zip(src,streams,origins,stops,strict=True)):
            ref=ys[i]
            visible=set(ps[i]+qs[i])
            presented=set(ps[i]+qs[(i+1)%len(qs)]) if arm=="exchanged" else visible
            assert all(int(a) in (0,1) and (a==1 or g in presented) for g,a in zip(out,actions,strict=True))
            if arm in ("copy","released"):
                assert not np.any(actions)
            if arm=="finite":
                assert all(a==0 or g in memory for g,a in zip(out,actions,strict=True))
            want=Counter(g for g in ref if g not in visible)
            gave=Counter(g for g in out if g not in visible)
            counts=dict(records=1,exact=int(stop and out==ref),stopped=int(stop),
                        nonempty=int(bool(out)),edit_distance=distance(ref,out),reference_cells=len(ref),
                        output_cells=len(out),input_absent_targets=sum(want.values()),
                        input_absent_emitted=sum(gave.values()),input_absent_overlap=sum((want&gave).values()),
                        input_absent_exact_position=sum(g==y and y not in visible for g,y in zip(out,ref)),
                        write_actions=int(actions.sum()),written_outside_memory=sum(a==1 and g not in memory for g,a in zip(out,actions)),
                        emitted_outside_input_and_training=sum(g not in presented and g not in train_tiles for g in out))
            for group in ("all",meta["kind"]):
                bag.setdefault(group,Counter()).update(counts)
        report[arm]={k:dict(v) for k,v in bag.items()}
    return dict(scores=report,fit=load(OUT/"fit.json"),execution=execution)


if __name__=="__main__":
    result=recount()
    write_json(OUT/"audit.json",result)
    print(json.dumps({k:v["all"] for k,v in result["scores"].items()},indent=2))

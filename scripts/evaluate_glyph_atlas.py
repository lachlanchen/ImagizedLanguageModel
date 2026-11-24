"""Score question reading, catalogue search and complete returned source pixels."""
import json
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.glyph_atlas import address,mentions,requested,payload
from scripts.atlas_questions import FIELDS
from scripts.prepare_glyph_atlas import PUB,WORK,MODEL,dump,sha
from scripts.read_glyph_atlas import get_book
from scripts.train_glyph_atlas import datafile,load_model,banks,gpu_owner,snapshot,consumed


def decide(net,bank,keys,data,face):
    cols=[[] for _ in range(5)]
    for i in range(0,len(data["head"]),96):
        sl=slice(i,i+96)
        q=data["query"][sl].astype(np.int64)
        width=int((q!=0).sum(1).max())
        q=torch.tensor(q[:,:width],device=bank.device)
        ink=bank[1 if face=="serif" else 0,q]
        if face=="droid-head":
            row=torch.arange(len(q),device=q.device)
            h=data["head"][sl].astype(np.int64)
            ink[row,h]=bank[2,q[row,h]]
        result=address(net,ink,q!=0,bank[0,keys],approximate=face!="canonical")
        for j,v in enumerate(result):
            if j==2:
                v=np.pad(v,((0,0),(0,data["query"].shape[1]-width)))
            cols[j].append(v)
    return [np.concatenate(v) for v in cols]


def evaluate(book,data,entries,tags,pool=None,change_ink=False):
    n=len(entries)
    flags=np.zeros((n,6),bool)
    for i,e in enumerate(entries):
        right=data["entry"][i]
        actual=requested(mentions(data["tag"][i]),book["spans"][right])
        reference=payload(book,right,actual)
        flags[i,0]=e==right
        flags[i,4]=all(len(s) for s in reference) and bool(reference)
        flags[i,5]=any(len(s) for s in reference)
        if e<0:
            continue
        readfields=requested(mentions(tags[i]),book["spans"][e])
        emitted=payload(book,e,readfields)
        if change_ink:
            # Keep selected headings and lengths, replace each selected glyph image by the next
            # entry in the actual optical palette. Different indices have different packed bits.
            emitted=[(s+1)%len(book["tiles"]) for s in emitted]
        flags[i,1]=actual==readfields
        flags[i,2]=len(emitted)==len(reference) and all(np.array_equal(a,b) for a,b in zip(emitted,reference))
        flags[i,3]=bool(flags[i,:3].all())
    return flags


def summarize(flags,data):
    out={}
    masks={k:data["kind"]==k for k in np.unique(data["kind"])}
    masks.update(all=np.ones(len(flags),bool),english=data["lang"]==1,chinese=data["lang"]==0)
    for name,sel in masks.items():
        x=flags[sel]
        out[str(name)]=dict(pages=len(x),entry=int(x[:,0].sum()),fields=int(x[:,1].sum()),pixels=int(x[:,2].sum()),
                           exact=int(x[:,3].sum()),available=int(x[:,4].sum()),exact_available=int((x[:,3]&x[:,4]).sum()),
                           with_ink=int(x[:,5].sum()),exact_with_ink=int((x[:,3]&x[:,5]).sum()))
    for kind in ("familiar","novel"):
        sel=data["kind"]==kind
        if not sel.any():
            continue
        x=flags[sel,3].reshape(-1,4)
        groups=data["group"][sel].reshape(-1,4)
        if not (groups==groups[:,:1]).all():
            raise ValueError("Crossed questions lost alignment")
        out[kind]["all_four"]=int(x.all(1).sum());out[kind]["quartets"]=len(x)
    return out


def main():
    obs=datafile("observations.npz")
    data=datafile("development.npz")
    book,sources=get_book(WORK)
    keys=obs["keys"]
    lookup={int(k):i for i,k in enumerate(keys)}
    naive=np.array([next((lookup[int(v)] for v in q if int(v) in lookup),-1) for q in data["query"]])
    output={};summary={};costs={}
    with gpu_owner():
        for mode in ("ink","byte"):
            net=load_model(mode,"cuda")
            bank=banks(obs,mode,"cuda")
            for face in (("canonical","serif","droid-head") if mode=="ink" else ("canonical",)):
                clock=monotonic();energy=snapshot()
                entry,pos,tag,raw,exact=decide(net,bank,keys,data,face)
                name=mode+"-"+face
                values={name:(entry,tag)}
                if mode=="ink" and face=="canonical":
                    values.update(supplied=(data["entry"],tag),first_match=(naive,tag))
                    # Reverse the entire predicted query within each two-head/two-order group.
                    exchange=np.arange(len(entry))
                    for offset in range(0,len(entry),5):
                        exchange[offset:offset+4]=exchange[offset:offset+4][::-1]
                    values["question_exchanged"]=(entry[exchange],tag[exchange])
                    values["changed_payload"]=(entry,tag)
                elif face!="canonical":
                    values[name+"-raw"]=(raw,tag)
                for k,(sel,label) in values.items():
                    flags=evaluate(book,data,sel,label,change_ink=k=="changed_payload")
                    output[k+"_flags"]=flags
                    summary[k]=summarize(flags,data)
                output[name+"_entry"]=entry;output[name+"_tag"]=tag;output[name+"_focus"]=pos
                costs[name]=dict(seconds=monotonic()-clock,energy=consumed(energy,snapshot()),
                                 head_position=int((pos==data["head"]).sum()),literal=int(exact.sum()))
                print(name,summary[name]["all"],flush=True)
    a=PUB/"outputs.npz"
    np.savez_compressed(a,**output)
    dump(PUB/"results.json",dict(summary=summary,costs=costs,outputs=sha(a),corpus=sha(PUB/"corpus.json"),
                                writers=sha(MODEL/"weights.json"),droid_head_coverage=int(obs["tiles"][2,keys[data["entry"]]].any(1).sum())))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()

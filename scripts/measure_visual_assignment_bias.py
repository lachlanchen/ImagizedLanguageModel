"""Disentangle target preference from source discrimination on paired translations."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.reserve_independent_optical_actions import RUN as CORPUS,PUB as PRIOR
from scripts.reserve_source_supported_writing import ROOT,RUN,PUB,BASE,read,sha,dump
from scripts.score_optical_action_sources import encoded_targets
from scripts.write_independent_optical_actions import ImageWriter


@torch.inference_mode()
def log_pairs(body,tiles,table,inputs,targets,t):
    lens=torch.tensor(list(map(len,inputs)))
    x=torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True)
    mem,mask=body.encode(pool_tiles(x),lens)
    hist,keys=body.actions(tiles,table)
    scores=[]
    for path in targets:
        previous=hist[path]
        z=torch.cat((body.begin[t,None],previous),0).expand(len(inputs),-1,-1)
        dec,_=body.decode(z,mem,mask)
        logits=(dec@keys.T)*keys.size(-1)**-.5
        labels=torch.tensor([*path,0]).expand(len(inputs),-1)
        scores.append(logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1).double().sum(1).numpy())
    return np.stack(scores,axis=1)


def main():
    torch.set_num_threads(2)
    books=read(RUN/"panel.json")
    roles=read(PRIOR/"role-reversal.json")["cases"]
    rows=books+roles
    train=np.load(CORPUS/"cells.npy",mmap_mode="r")
    lens=np.count_nonzero(train,axis=2)
    bits=torch.from_numpy(np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32))
    writer=ImageWriter("strip","cpu")
    records=[]
    for t in (0,1):
        paths=encoded_targets(rows,t)["strip"]
        pairs=[]
        for title in dict.fromkeys(r["doc"] for r in books):
            order=sorted((i for i,r in enumerate(books) if r["doc"]==title),key=lambda i:(len(books[i]["cells"][1-t]),i))
            pairs+=list(zip(order[::2],order[1::2],strict=True))
        pairs+=list(zip(range(len(books),len(rows),2),range(len(books)+1,len(rows),2),strict=True))
        order=np.random.default_rng(13941+t).permutation(len(train))
        background={}
        for a,b in pairs:
            actual=[r["cells"][1-t] for r in (rows[a],rows[b])]
            length=(len(actual[0])+len(actual[1]))//2
            if length not in background:
                rank=np.argsort(np.abs(lens[order,1-t]-length),kind="stable")[:16]
                background[length]=order[rank]
            selected=background[length]
            inputs=[bits[v] for v in actual]+[bits[train[j,1-t,:lens[j,1-t]].astype(int)] for j in selected]
            scores=log_pairs(writer.model,writer.tiles,writer.tables[t],inputs,[paths[a],paths[b]],t)
            orig=scores[0,0]-scores[0,1],scores[1,1]-scores[1,0]
            delta=float((scores[2:,0]-scores[2:,1]).mean())
            u,v=map(float,orig)
            assert ((u>0 and v>0)==(max(u-v,v-u)<u+v))
            n=scores[:2]/np.array([len(paths[a])+1,len(paths[b])+1])
            records.append(dict(a=a,b=b,t=t,group="book" if a<len(books) else "roles",u=u,v=v,
                        same_bag=Counter(paths[a])==Counter(paths[b]),background=selected.tolist(),
                        scores=scores.tolist(),prior=delta,joint=u+v>0,both=u>0 and v>0,
                        corrected=u-delta>0 and v+delta>0,normalized=bool(n[0,0]>n[0,1] and n[1,1]>n[1,0])))
    counts={}
    for group in ("book","roles","same_role_bag"):
        r=[v for v in records if v["group"]==group or (group=="same_role_bag" and v["group"]=="roles" and v["same_bag"])]
        counts[group]=dict(n=len(r),**{k:sum(v[k] for v in r) for k in ("joint","both","corrected","normalized")})
    dump(PUB/"assignment-bias.json",dict(counts=counts,rows=records,
         proof=sha(ROOT/"formal/span_list/IndependentVisualChoices.lean"),program=sha(Path(__file__)),
         protocol=sha(ROOT/"references/visual_assignment_bias_protocol.md")))
    print(counts,flush=True)


if __name__=="__main__":
    main()

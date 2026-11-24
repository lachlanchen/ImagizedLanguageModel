"""Prepare cross-fitted suffix probabilities and the shared deployment count memory."""
import json
from time import monotonic
import numpy as np
from ilm.visual_lm.prefix_counts import PrefixCounts,history_windows
from scripts.prepare_book_prefix import corpus,MODEL,WORK,PUB
from scripts.prepare_dialogue_ink import sha,dump


def fit():
    m,data=corpus()
    clock=monotonic()
    stream=data["stream"]
    rec=[r for r in m["records"] if r[0]=="train"]
    history=history_windows(stream,rec)
    fold=np.full(len(stream),-1,np.int8)
    for _,_,_,f,a,b in rec:
        fold[a:b]=f
    features=np.zeros((len(stream),3),np.float32)
    prob=np.zeros(len(stream),np.float32)
    dest=MODEL/"counts.npz"
    cache=WORK/"oof.npz"
    MODEL.mkdir(parents=True,exist_ok=True)
    for f in range(5):
        train=(fold>=0)&(fold!=f)
        index=PrefixCounts.build(history[train],stream[train],m["train_shapes"])
        if f==4:
            np.savez_compressed(dest,**index.arrays)
            break
        select=fold==f
        nodes,features[select]=index.lookup(history[select])
        prob[select]=index.target_prob(nodes,stream[select])
    if not (prob[fold>=0]>0).all():
        raise ValueError("Unreachable cross-fitted observation")
    np.savez_compressed(cache,features=features,prob=prob)
    result=dict(corpus=m["corpus"],fold_positions=[int((fold==f).sum()) for f in range(4)],
                inference_sha=sha(dest),inference_bytes=dest.stat().st_size,oof=sha(cache),
                nodes=len(index.total)-1,edges=len(index.edges),seconds=monotonic()-clock)
    dump(PUB/"counts.json",result)
    print(result,flush=True)


if __name__=="__main__":
    fit()

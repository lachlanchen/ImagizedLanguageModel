"""Learn bilingual visual-strip probabilities by marginalizing source alignment."""
from pathlib import Path
from time import monotonic
import numpy as np
from numba import njit
import torch
from scripts.prepare_full_image_roles import BASE,BOOK,REHEARSAL,ROOT,Ink,candidates,read,dump,sha
from scripts.prepare_full_image_roles import PUB as FIRST_PUB,WORK as FIRST
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

PUB=ROOT/"references/evidence/optical_lexicon_v1"
WORK=ROOT/"artifacts/optical_lexicon_v1"
MODEL=ROOT/"models/optical_lexicon"
TABLE=ROOT/"models/independent_optical_actions/tables.json"
OPTICAL=ROOT/"artifacts/independent_optical_actions_v1/cells.npy"


def trie(table):
    nodes=[{}]
    emit=[0]
    for i,strip in enumerate(table[1:],1):
        n=0
        for c in strip:
            if c not in nodes[n]:
                nodes[n][c]=len(nodes)
                nodes.append({})
                emit.append(0)
            n=nodes[n][c]
        emit[n]=i
    # All single pictured glyphs are terminals; absent transitions are zero.
    edges=np.zeros((len(nodes),max(max(x,default=0) for x in table)+1),np.int16)
    for i,row in enumerate(nodes):
        for c,j in row.items():
            edges[i,c]=j
    return edges,np.array(emit,np.int16)


@njit
def segment(cells,edges,terminal):
    out=np.zeros_like(cells)
    for i in range(len(cells)):
        j=k=0
        while j<cells.shape[1] and cells[i,j]:
            p=j
            node=best=0
            while p<cells.shape[1] and 0<cells[i,p]<edges.shape[1]:
                node=edges[node,cells[i,p]]
                if node==0:
                    break
                p+=1
                if terminal[node]:
                    best=terminal[node]
                    j=p
            if best==0:
                raise ValueError("Unsupported physical strip")
            out[i,k]=best
            k+=1
    return out


def compact(ids):
    x=np.zeros(96,np.uint16)
    x[:len(ids)]=ids
    return x.tobytes()


def prepare():
    for d in (PUB,WORK,MODEL):
        d.mkdir(parents=True,exist_ok=True)
    first=np.load(BOOK/"cells.npy")
    with np.load(REHEARSAL) as z:
        first[z["indices"]]=z["cells"]
    allpairs=np.concatenate((first,np.load(OPTICAL)))
    keys=allpairs.reshape(len(allpairs),-1).view("V384").ravel()
    _,idx=np.unique(keys,return_index=True)
    allpairs=allpairs[np.sort(idx)]
    dev_rows=read(BOOK/"dev.json")
    # Same chronological selection in each domain; avoid bitmap overlap on either side.
    dev=[]
    bydomain={}
    for r in dev_rows:
        if bydomain.get(r["domain"],0)<32:
            dev.append(r)
            bydomain[r["domain"]]=bydomain.get(r["domain"],0)+1
    prev=read(FIRST/"panel.json")
    held=[{compact(r["cells"][t]) for r in dev+prev} for t in (0,1)]
    keep=np.fromiter((all(p[t].tobytes() not in held[t] for t in (0,1)) for p in allpairs),bool)
    allpairs=allpairs[keep]
    seen=[{p.tobytes() for p in allpairs[:,t]}|held[t] for t in (0,1)]
    inv=list(map(set,read(BASE/"inventories.json")))
    face=Ink(np.load(BASE/"palette.npy"))
    panel=[]
    # Preserve the original work split. Exclude earlier selected passages and all training sides.
    for title,digest in read(FIRST_PUB/"corpus.json")["editions"].items():
        ordered,observed=candidates(title)
        assert observed==digest
        n=0
        seen_norm=[{''.join(c for c in r["text"][t].casefold() if c.isalnum()) for r in prev+panel} for t in (0,1)]
        for _,en,zh,ch,para,u in ordered:
            txt=[en,zh]
            keys=[''.join(c for c in v.casefold() if c.isalnum()) for v in txt]
            if any(not k or k in seen_norm[t] for t,k in enumerate(keys)):
                continue
            ids=list(map(face.encode,txt))
            if any(not 1<=len(v)<=96 or not set(v)<=inv[t] or face.bad.intersection(txt[t]) or compact(v) in seen[t] for t,v in enumerate(ids)):
                continue
            for t,v in enumerate(ids):
                seen[t].add(compact(v))
                seen_norm[t].add(keys[t])
            panel.append(dict(doc=title,domain="book",text=txt,cells=ids,chunk=ch,paragraph=para,unit=u))
            n+=1
            if n==12:
                break
        assert n==12
    dump(WORK/"panel.json",panel)
    dump(WORK/"dev.json",dev)
    tables=read(TABLE)["strip"]
    strips=[segment(np.ascontiguousarray(allpairs[:,t]),*trie(tables[t])) for t in (0,1)]
    dump(PUB/"corpus.json",dict(pairs=len(allpairs),split_exclusions=int((~keep).sum()),development=len(dev),
            tables=sha(TABLE),opus=sha(OPTICAL),books=sha(BOOK/"cells.npy"),substitutions=sha(REHEARSAL),
            reserved=sha(WORK/"panel.json"),dev=sha(WORK/"dev.json"),source=sha(Path(__file__)),
            protocol=sha(ROOT/"references/optical_lexicon_protocol.md")))
    return strips,tables


def fit(strips,tables,t):
    # Source index 0 is the NULL alignment, target index 0 is padding.
    folder=MODEL/("en","zh")[t]
    folder.mkdir(exist_ok=True)
    out=folder/"probabilities.npz"
    x,y=[torch.from_numpy(strips[z].astype(np.int32)).cuda() for z in (1-t,t)]
    lens=np.stack([np.count_nonzero(strips[z],axis=1) for z in (1-t,t)],1)
    bins=lens[:,0]//6*32+lens[:,1]//6
    order=np.argsort(bins,kind="stable")
    batches=[g[j:j+256] for g in np.split(order,np.flatnonzero(np.diff(bins[order]))+1) for j in range(0,len(g),256)]
    n,m=len(tables[1-t]),len(tables[t])
    freq=torch.bincount(y.flatten().long(),minlength=m).float()+.01
    freq[0]=0
    prior=freq/freq.sum()
    probs=prior.expand(n,m).contiguous()
    history=[]
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(5):
        counts=torch.zeros_like(probs)
        nll=0.
        for idx in batches:
            src=x[idx,:lens[idx,0].max()].long()
            tgt=y[idx,:lens[idx,1].max()].long()
            attached=torch.nn.functional.pad(src,(1,0))
            valid=torch.nn.functional.pad(src.ne(0),(1,0),value=True)
            index=attached.unsqueeze(2)*m+tgt.unsqueeze(1)
            weights=probs.flatten()[index]*valid.unsqueeze(2)
            norm=weights.sum(1).clamp_min(1e-30)
            nll-=((norm/valid.sum(1,keepdim=True)).log()*tgt.ne(0)).sum().item()
            weights=(weights/norm[:,None])*tgt.ne(0)[:,None]
            counts.view(-1).scatter_add_(0,index.flatten(),weights.flatten())
        probs=(counts+2*prior)/(counts.sum(1,keepdim=True)+2)
        history.append(nll/int(lens[:,1].sum()))
        if monotonic()-start>900 or torch.cuda.max_memory_allocated()>12*2**30:
            raise RuntimeError("Lexical resource ceiling")
    # Store frequent learned correspondences plus an exact prior tail approximation per row.
    # Renormalize after truncation so every target strip retains positive support.
    vals,ids=probs[:,1:].topk(128)
    ids+=1
    tail=(1-vals.double().sum(1)).clamp_min(0)
    np.savez_compressed(out,ids=ids.cpu().numpy().astype(np.uint16),mass=vals.cpu().numpy(),
                        tail=tail.cpu().numpy().astype(np.float32),prior=prior.cpu().numpy())
    info=dict(weights=sha(out),corpus=sha(PUB/"corpus.json"),seconds=monotonic()-start,
              peak_bytes=torch.cuda.max_memory_allocated(),mean_strip_nll=history,shape=[n,m],retained=128)
    dump(folder/"fit.json",info)
    print(t,info,flush=True)


def main():
    configure_cuda_cli()
    if all((MODEL/l/"fit.json").exists() for l in ("en","zh")):
        for l in ("en","zh"):
            assert sha(MODEL/l/"probabilities.npz")==read(MODEL/l/"fit.json")["weights"]
        return
    strips,tables=prepare()
    with gpu_owner():
        for t in (0,1):
            folder=MODEL/("en","zh")[t]
            if (folder/"fit.json").exists():
                info=read(folder/"fit.json")
                assert info["corpus"]==sha(PUB/"corpus.json") and info["weights"]==sha(folder/"probabilities.npz")
            else:
                fit(strips,tables,t)


if __name__=="__main__":
    main()

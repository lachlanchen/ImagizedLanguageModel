"""Estimate a compact bilingual phrase inventory and target strip language models."""
from pathlib import Path
from time import monotonic
import os
import resource
import numpy as np
from numba import njit
from scripts.prepare_covered_phrases import MODEL,PUB,WORK,base,prior

BITS=13
WIDTH=4


@njit
def links(n,m,back,forward):
    """Intersection, adjacent union links, and a final pass over unclaimed pairs."""
    grid=np.zeros((n,m),np.uint8)
    usedx=np.zeros(n,np.bool_)
    usedy=np.zeros(m,np.bool_)
    for i in range(n):
        j=back[i]
        if j>=0 and forward[j]==i:
            grid[i,j]=1
            usedx[i]=usedy[j]=True
    for stage in range(2):
        changed=True
        while changed:
            changed=False
            for direction in range(2):
                for k in range(n if direction==0 else m):
                    i,j=(k,int(back[k])) if direction==0 else (int(forward[k]),k)
                    if min(i,j)<0 or grid[i,j] or (usedx[i] and usedy[j]):
                        continue
                    if stage==0:
                        if not grid[max(0,i-1):min(n,i+2),max(0,j-1):min(m,j+2)].any():
                            continue
                    elif usedx[i] or usedy[j]:
                        continue
                    grid[i,j]=1
                    usedx[i]=usedy[j]=True
                    changed=True
    return grid


@njit
def extract(x,y,back,forward):
    sources=[np.int64(0)]
    targets=[np.int64(0)]
    for row in range(len(x)):
        n=np.count_nonzero(x[row])
        m=np.count_nonzero(y[row])
        grid=links(n,m,back[row],forward[row])
        left=np.full(n,m)
        right=np.full(n,-1)
        lower=np.full(m,n)
        upper=np.full(m,-1)
        for i in range(n):
            for j in range(m):
                if grid[i,j]:
                    left[i]=min(j,left[i])
                    right[i]=max(j,right[i])
                    lower[j]=min(i,lower[j])
                    upper[j]=max(i,upper[j])
        for a in range(n):
            lo,hi=m,-1
            src=np.int64(0)
            for b in range(a,min(a+WIDTH,n)):
                src=(src<<BITS)+x[row,b]
                lo=min(lo,left[b])
                hi=max(hi,right[b])
                if hi-lo>=WIDTH:
                    break
                if hi<0 or lower[lo:hi+1].min()<a or upper[lo:hi+1].max()>b:
                    continue
                # Unlinked target strips adjoining the aligned core may be absorbed.
                l=lo
                while l>=0 and hi-l<WIDTH:
                    phrase=np.int64(0)
                    for r in range(l,min(m,l+WIDTH)):
                        if r>hi and upper[r]>=0:
                            break
                        phrase=(phrase<<BITS)+y[row,r]
                        if r>=hi:
                            sources.append(src)
                            targets.append(phrase)
                    if l==0 or upper[l-1]>=0:
                        break
                    l-=1
    return np.array(sources[1:]),np.array(targets[1:])


def reduce_pairs(src,tgt,counts=None):
    order=np.lexsort((tgt,src))
    a,b=src[order],tgt[order]
    starts=np.r_[0,1+np.flatnonzero((a[1:]!=a[:-1])|(b[1:]!=b[:-1]))]
    weight=np.diff(np.r_[starts,len(a)]) if counts is None else np.add.reduceat(counts[order],starts)
    return a[starts],b[starts],weight.astype(np.int32)


def retained(a,b,w):
    order=np.lexsort((b,-w,a))
    a,b,w=a[order],b[order],w[order]
    starts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1]
    sizes=np.diff(np.r_[starts,len(a)])
    total=np.repeat(np.add.reduceat(w,starts),sizes)
    keep=(w>=2)&(np.arange(len(a))-np.repeat(starts,sizes)<8)
    return a[keep],b[keep],w[keep],(w[keep]/total[keep]).astype(np.float32)


@njit
def bigram_count(rows,size):
    counts=np.zeros((size+1,size+1),np.int32)
    for seq in rows:
        prev=size
        for q in seq:
            if q==0:
                break
            counts[prev,q]+=1
            prev=q
        counts[prev,size]+=1
    return counts


def main():
    start=monotonic()
    MODEL.mkdir(exist_ok=True,parents=True)
    seq=[np.load(prior.WORK/f"strips-{t}.npy",mmap_mode="r") for t in (0,1)]
    paths=[WORK/f"links-{t}.npy" for t in (0,1)]
    receipts=[base.read(PUB/f"links-{t}.json") for t in (0,1)]
    assert [r["hash"] for r in receipts]==[base.sha(p) for p in paths]
    align=[np.load(p,mmap_mode="r") for p in paths]
    out=MODEL/"fit.json"
    if out.exists():
        assert all(base.sha(MODEL/p)==h for p,h in base.read(out)["hashes"].items())
        return
    pieces=[]
    observations=0
    for i in range(0,len(seq[0]),8192):
        span=slice(i,i+8192)
        src,tgt=extract(seq[0][span],seq[1][span],align[0][span],align[1][span])
        observations+=len(src)
        pieces.append(reduce_pairs(src,tgt))
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>20*1024**2 or monotonic()-start>3600:
            raise RuntimeError("Registered phrase counting resource ceiling reached")
    combined=[np.concatenate(col) for col in zip(*pieces,strict=True)]
    del pieces
    source,target,weight=reduce_pairs(*combined)
    del combined
    inventory=base.read(base.TABLE)["strip"]
    files={}
    summary={}
    for t,lang in enumerate(("en","zh")):
        a,b=(target,source) if t==0 else (source,target)
        src,tgt,count,law=retained(a,b,weight)
        # Reverse relative frequency includes every observed candidate before pruning.
        order=np.argsort(b,kind="stable")
        keys,idx=np.unique(b[order],return_index=True)
        totals=np.add.reduceat(weight[order],idx)
        inverse=(count/totals[np.searchsorted(keys,tgt)]).astype(np.float32)
        lm=bigram_count(seq[t],len(inventory[t]))
        row,col=np.nonzero(lm)
        path=MODEL/f"{lang}.npz"
        partial=path.with_suffix(".partial.npz")
        np.savez_compressed(partial,source=src,target=tgt,count=count,prob=law,reverse=inverse,
                            left=row.astype(np.uint16),right=col.astype(np.uint16),bigram=lm[row,col])
        os.replace(partial,path)
        files[path.name]=base.sha(path)
        summary[lang]=dict(pairs=len(src),sources=len(np.unique(src)),bigrams=len(row),bytes=path.stat().st_size)
        print(lang,summary[lang],flush=True)
    base.dump(out,dict(phrases=summary,observations=observations,distinct=len(source),training_pairs=len(seq[0]),
             seconds=monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             hashes=files,links=[r["hash"] for r in receipts],corpus=base.sha(PUB/"corpus.json"),code=base.sha(Path(__file__))))
    print("complete",observations,flush=True)


if __name__=="__main__":
    main()

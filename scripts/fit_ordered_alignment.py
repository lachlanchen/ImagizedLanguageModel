"""Estimate source movement by exact forward–backward inference on one GPU."""
from itertools import product
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.optical_lexicon import expand_table
from scripts.prepare_ordered_alignment import MODEL, PUB, WORK, base
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

JUMPS = (-1, 0, 1, 2)


def expectation(emission, lengths, target_lengths, mixing):
    """Return negative log likelihood and expected component uses per batch."""
    batch, steps, width = emission.shape
    index = torch.arange(width, device=emission.device)[None,:].expand(batch,-1)
    valid = index < lengths[:,None]
    uniform = valid.to(emission.dtype) / lengths[:,None]
    dest = [(index+d).clamp(min=0).minimum(lengths[:,None]-1) for d in JUMPS]
    alpha = uniform
    filtered, scales = [], []
    loss = emission.new_zeros(())
    for j in range(steps):
        if j:
            pred = mixing[0]*uniform
            for w,to in zip(mixing[1:],dest,strict=True):
                pred.scatter_add_(1,to,alpha*w)
        else:
            pred = uniform
        raw = pred*emission[:,j]
        scale = raw.sum(-1,keepdim=True)
        alpha = raw/scale
        filtered.append(alpha)
        scales.append(scale)
        loss -= (scale[:,0].log()*(j<target_lengths)).sum()
    counts = emission.new_zeros(5)
    beta = torch.ones_like(alpha)
    for j in range(steps-1,0,-1):
        live = (j<target_lengths)[:,None]
        likelihood = emission[:,j]*beta/scales[j]
        reset = (uniform*likelihood).sum(-1,keepdim=True)
        previous = mixing[0]*reset
        counts[0] += (previous*live).sum()
        for k,to in enumerate(dest,1):
            term = mixing[k]*likelihood.gather(1,to)
            counts[k] += (filtered[j-1]*term*live).sum()
            previous = previous+term
        beta = torch.where(live,previous,1.)
    return loss,counts


def check(device):
    rng = np.random.default_rng(3047)
    probabilities = rng.random((2,3,3))+.05
    mix = np.array([.3,.1,.2,.35,.05])
    ll,counts = expectation(torch.tensor(probabilities,device=device),torch.tensor([3,2],device=device),
                    torch.tensor([3,2],device=device),torch.tensor(mix,device=device))
    total,expected=0.,np.zeros(5)
    for e,n,steps in zip(probabilities,(3,2),(3,2),strict=True):
        mass=0.
        count=np.zeros(5)
        for states in product(range(n),repeat=steps):
            for moves in product(range(5),repeat=steps-1):
                p=e[0,states[0]]/n
                for j,m in enumerate(moves,1):
                    prev=states[j-1]
                    transition=1/n if m==0 else int(states[j]==np.clip(prev+JUMPS[m-1],0,n-1))
                    p*=mix[m]*transition*e[j,states[j]]
                count+=p*np.bincount(moves,minlength=5)
                mass+=p
        total-=np.log(mass)
        expected+=count/mass
    err=max(abs(float(ll)-total),float(abs(counts.cpu().numpy()-expected).max()))
    if err>1e-6:
        raise ValueError(f"GPU forward–backward disagrees with enumerated paths: {err}")
    return err


def fit(t,corpus):
    folder=MODEL/("en","zh")[t]
    folder.mkdir(exist_ok=True,parents=True)
    train=folder/"fit.json"
    lex=base.MODEL/folder.name/"probabilities.npz"
    inputs=dict(corpus=base.sha(PUB/"corpus.json"),emissions=base.sha(lex),implementation=base.sha(Path(__file__)))
    if train.exists():
        assert base.read(train)["inputs"]==inputs
        return
    with np.load(lex) as z:
        table=torch.from_numpy(expand_table(z["ids"],z["mass"],z["tail"],z["prior"])).cuda()
    seq=[np.load(WORK/f"strips-{q}.npy",mmap_mode="r") for q in (1-t,t)]
    lens=np.array([np.count_nonzero(z,axis=1) for z in seq]).T
    assert lens.min()>0
    src,tgt=[torch.from_numpy(np.array(z,dtype=np.int32)).cuda() for z in seq]
    lens_gpu=torch.tensor(lens,device="cuda")
    bucket=(lens[:,0]//8)*32+lens[:,1]//8
    order=np.argsort(bucket,kind="stable")
    batches=[g[i:i+1536] for g in np.split(order,np.flatnonzero(np.diff(bucket[order]))+1)
             for i in range(0,len(g),1536)]
    batches=[(torch.tensor(b,device="cuda"),*map(int,lens[b].max(0))) for b in batches]
    mixing=torch.tensor([.40,.10,.15,.25,.10],device="cuda")
    start=monotonic()
    trace=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(4):
        totals=torch.zeros(5,device="cuda",dtype=torch.float64)
        loss=torch.zeros((),device="cuda",dtype=torch.float64)
        for ids,n,m in batches:
            a=src[ids,:n].long()
            b=tgt[ids,:m].long()
            length,steps=lens_gpu[ids].unbind(1)
            # Padded target positions are neutral; masked source columns have no mass.
            b=b.clamp(min=1)
            size=length[:,None,None]
            emission=(size*table[a[:,None,:],b[:,:,None]]+table[0,b][:,:,None])/(size+1)
            nll,counts=expectation(emission,length,steps,mixing)
            loss+=nll
            totals+=counts
        assert abs(float(totals.sum())-(int(lens[:,1].sum())-len(lens)))<5.
        mixing=((totals+1)/(totals.sum()+5)).float()
        trace.append(dict(mean_nll=float(loss)/int(lens[:,1].sum()),after=mixing.tolist()))
        print(folder.name,epoch+1,trace[-1],flush=True)
        if monotonic()-start>1800 or torch.cuda.max_memory_allocated()>12*2**30:
            raise RuntimeError("Registered fit budget exceeded")
    assert all(b["mean_nll"]<=a["mean_nll"]+1e-5 for a,b in zip(trace,trace[1:]))
    base.dump(train,dict(inputs=inputs,pairs=corpus["pairs"],target_strips=int(lens[:,1].sum()),history=trace,
                   mixing=mixing.tolist(),seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated()))


def main():
    configure_cuda_cli()
    corpus=base.read(PUB/"corpus.json")
    assert corpus["strips"]==[base.sha(WORK/f"strips-{t}.npy") for t in (0,1)]
    with gpu_owner():
        err=check("cuda")
        base.dump(PUB/"kernel-check.json",dict(max_abs_error=err,device=torch.cuda.get_device_name()))
        for t in (0,1):
            fit(t,corpus)


if __name__=="__main__":
    main()

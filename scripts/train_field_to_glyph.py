"""Matched independent pixel-writer fits under two frozen kinds of optical source supervision."""
from pathlib import Path
import hashlib
import math
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.field_to_glyph import MODELS, restore
from scripts.reserve_field_transfer import ROOT, RUN, PUB, read, sha, dump, library
from scripts.train_visual_prefix import atomic_save, gpu_owner, material, pack
from scripts.train_book_breadth import schedule
from scripts.verse_gpu_runtime import configure_cuda_cli

ARMS=("regression","prediction")


@torch.no_grad()
def cache_field(reader,features,x,lengths):
    """Store one actual 512-vector per occupied source glyph, with row offsets."""
    starts=np.cumsum(np.r_[0,lengths[:-1]])
    grid=torch.empty((int(lengths.sum())+1,512),device=x.device)
    grid[-1].zero_()
    for j in range(0,len(x),64):
        ls=lengths[j:j+64]
        n=int(ls.max())
        src=x[j:j+64,:n]
        field,_,_=reader(features,src)
        flat=field[:,1:n+1][src.ne(0)]
        grid[starts[j]:starts[j]+int(ls.sum())]=flat
    return grid,torch.as_tensor(starts,device=x.device)


def scores(net,bank,source,target,field,direction,keys):
    context,valid=net.writer.encode(bank,source)
    memory=net.add_field(context,field)
    x=torch.cat((net.writer.begin[direction].expand(len(source),1,-1),bank[target[:,:-1]]),1)
    hidden,_=net.writer.decode(x,memory,valid)
    return hidden @ keys.T / math.sqrt(hidden.shape[-1])


def data_for(rows,t,device):
    x=[r["cells"][1-t] for r in rows]
    y=[[*r["cells"][t],1] for r in rows]
    lens=np.asarray([(len(a),len(b)) for a,b in zip(x,y,strict=True)])
    return pack(x,device),pack(y,device),lens


@torch.inference_mode()
def dev_losses(net,features,bank,keys,maps,rows):
    result={}
    for t,lang in enumerate(("en","zh")):
        x,y,lens=data_for(rows,t,"cuda")
        nll={"real":0.,"exchanged":0.}
        # Exchange both the base and continuous source, keep the true target prefix fixed.
        order=np.argsort(lens[:,0],kind="stable")
        swap=np.empty(len(rows),dtype=int)
        swap[order]=order.reshape(-1,2)[:,::-1].flatten()
        for kind in nll:
            for j in range(0,len(x),24):
                ix=np.arange(j,min(len(x),j+24))
                src=ix if kind=="real" else swap[ix]
                w=int(lens[src,0].max())
                a=x[src,:w]
                b=y[ix,:int(lens[ix,1].max())]
                z=net.readers[t](features,a)[0][:,1:w+1]
                v=scores(net,bank,a,b,z,t,keys[t])
                nll[kind]+=float(F.cross_entropy(v.flatten(0,1),maps[t][b].flatten(),ignore_index=-100,reduction="sum"))
        result[lang]={k:v/int(lens[:,1].sum()) for k,v in nll.items()}
    return result


def fit(arm,features,inventories,maps,rows,dev):
    dest=MODELS/arm
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/"weights.safetensors"
    binding={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/"ilm/visual_lm/field_to_glyph.py",
             ROOT/"references/field_to_glyph_protocol.md",PUB/"panel.json")}
    if (dest/"fit.json").exists():
        rec=read(dest/"fit.json")
        if rec["binding"]!=binding or sha(path)!=rec["weights"]: raise ValueError("Changed completed fit")
        return
    torch.manual_seed(91822)
    net,parents=restore(arm,"cuda",initial=True)
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        bank=net.writer.optical(features)
        keys=[net.writer.prototypes(bank,v) for v in inventories]
    data=[data_for(rows,t,"cuda") for t in (0,1)]
    cache=[cache_field(net.readers[t],features,x,lens[:,0]) for t,(x,_,lens) in enumerate(data)]
    initial=dev_losses(net,features,bank,keys,maps,dev)
    opt=torch.optim.AdamW(net.trainable().values(),lr=.0002,fused=True)
    history=[]
    for ep in range(4):
        rng=np.random.default_rng(19931+ep)
        batches=[(t,idx) for t in (0,1) for idx in schedule(data[t][2],ep)]
        fingerprint=hashlib.sha256()
        stats=np.zeros((2,2))
        for order,b in enumerate(rng.permutation(len(batches))):
            t,idx=batches[b]
            fingerprint.update(bytes([t])+idx.astype("<i4").tobytes())
            x,y,lens=data[t]
            w,v=lens[idx].max(0)
            ii=torch.as_tensor(idx,device="cuda")
            src,target=x[ii,:w],y[ii,:v]
            grid,starts=cache[t]
            addresses=starts[ii,None]+torch.arange(w,device="cuda")
            field=grid[addresses.masked_fill(src.eq(0),len(grid)-1)]
            opt.zero_grad(set_to_none=True)
            progress=(ep*len(batches)+order)/(4*len(batches)-1)
            opt.param_groups[0]["lr"]=.000025+.0000875*(1+math.cos(math.pi*progress))
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=scores(net,bank,src,target,field,t,keys[t])
            labels=maps[t][target]
            loss=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.trainable().values(),1.,error_if_nonfinite=True)
            opt.step()
            positions=int(lens[idx,1].sum())
            stats[t]+=(float(loss.detach())*positions,positions)
        if not all(stats[t,1]==data[t][2][:,1].sum() for t in (0,1)): raise ValueError("Missing targets")
        peak=torch.cuda.max_memory_allocated()
        elapsed=monotonic()-start
        if peak>12*2**30 or elapsed>3600: raise RuntimeError("Training resource ceiling")
        history.append(dict(epoch=ep+1,order=fingerprint.hexdigest(),updates=len(batches),positions=stats[:,1].astype(int).tolist(),
                            nll=(stats[:,0]/stats[:,1]).tolist()))
        print(arm,history[-1],flush=True)
    terminal=dev_losses(net,features,bank,keys,maps,dev)
    atomic_save(lambda p:save_file({k:v.detach().cpu() for k,v in net.trainable().items()},p),path)
    dump(dest/"fit.json",dict(binding=binding,parents=parents,weights=sha(path),history=history,initial=initial,
                             terminal=terminal,peak_bytes=torch.cuda.max_memory_allocated(),seconds=monotonic()-start,
                             trainable=sum(p.numel() for p in net.trainable().values()),total=sum(p.numel() for p in net.parameters())))


def main():
    configure_cuda_cli()
    if sha(RUN/"panel.json")!=read(PUB/"panel.json")["sha256"]: raise ValueError("Reservation mismatch")
    with gpu_owner():
        torch.set_num_threads(2)
        features,inv,maps=material("cuda")
        rows,dev=library("train"),library("dev")
        for arm in ARMS:
            fit(arm,features,inv,maps,rows,dev)
            torch.cuda.empty_cache()
    a,b=[read(MODELS/arm/"fit.json") for arm in ARMS]
    assert a["initial"]==b["initial"] and [r["order"] for r in a["history"]]==[r["order"] for r in b["history"]]
    dump(PUB/"training.json",{a:sha(MODELS/a/"fit.json") for a in ARMS})


if __name__=="__main__":
    main()

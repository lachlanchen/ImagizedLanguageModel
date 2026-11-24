"""Fit optical and byte question readers on the same rendered reference questions."""
import hashlib
import math
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file

from ilm.visual_lm.glyph_atlas import AtlasReader
from scripts.prepare_glyph_atlas import WORK,MODEL,PUB,read,sha,dump
from scripts.train_verse_canvas import banks
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot,consumed

STEPS=2400
BATCH=96


def datafile(name):
    path=WORK/name
    if sha(path)!=read(PUB/"corpus.json")["files"][name]:
        raise ValueError("Changed atlas corpus "+name)
    with np.load(path) as f:
        return dict(f)


def load_model(mode,device):
    path=MODEL/(mode+".safetensors")
    manifest="wording-weights.json" if mode.endswith("-wording") else "weights.json"
    if sha(path)!=read(MODEL/manifest)["weights"][mode]:
        raise ValueError("Changed atlas reader")
    net=AtlasReader().to(device).eval()
    net.load_state_dict(load_file(str(path),device=str(device)))
    return net


def train(mode,obs,data,signature,edition=""):
    name=mode+edition
    path=MODEL/(name+".safetensors")
    receipt=PUB/(name+"-fit.json")
    if receipt.exists():
        item=read(receipt)
        if item["weights"]!=sha(path) or item["corpus"]!=signature or item["steps"]!=STEPS:
            raise ValueError("Incompatible fit")
        return item
    torch.manual_seed(47911)
    rng=np.random.default_rng(47911)
    digest=hashlib.sha256()
    net=AtlasReader().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,fused=True)
    bank=banks(obs,mode,"cuda")
    keys=torch.as_tensor(obs["keys"],device="cuda")
    freq=np.ones(8,np.float32)
    freq[1:]=18
    weights=torch.as_tensor(freq,device="cuda")
    trace=[]
    torch.cuda.reset_peak_memory_stats()
    start=monotonic()
    energy=snapshot()
    used=np.zeros(len(data["query"]),bool)
    for step in range(1,STEPS+1):
        ix=rng.integers(len(used),size=BATCH)
        used[ix]=True
        face=rng.integers(2,size=(BATCH,1))
        digest.update(ix.tobytes());digest.update(face.tobytes())
        q=data["query"][ix]
        n=int((q!=0).sum(1).max())
        q=torch.as_tensor(q[:,:n].astype(np.int64),device="cuda")
        valid=q!=0
        tags=torch.as_tensor(data["tag"][ix,:n].astype(np.int64),device="cuda")
        head=torch.as_tensor(data["head"][ix].astype(np.int64),device="cuda")
        ids=keys[data["entry"][ix]]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            ink=bank[torch.as_tensor(face,device="cuda"),q]
            pointer,classes=net(ink,valid)
            target=tags.masked_fill(~valid,-100)
            loss=F.cross_entropy(pointer,head)+F.cross_entropy(classes.flatten(0,1),target.flatten(),weight=weights)
            query=net.key(ink[torch.arange(BATCH,device="cuda"),head])
            logits=(query@net.key(bank[0,ids]).T).float()*24
            compatible=ids[:,None]==ids[None]
            loss=loss+.5*(logits.logsumexp(1)-logits.masked_fill(~compatible,-torch.inf).logsumexp(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.param_groups[0]["lr"]=.001*min(1,step/100)*(.12+.88*(1+math.cos(math.pi*step/STEPS))/2)
        opt.step()
        if step==1 or step%200==0:
            scalar=loss.item();elapsed=monotonic()-start
            trace.append([step,scalar,elapsed])
            print(name,step,round(scalar,4),round(elapsed,2),flush=True)
            if not math.isfinite(scalar) or elapsed>900 or torch.cuda.max_memory_allocated()>8*2**30:
                raise RuntimeError("Numerical or resource limit")
    torch.cuda.synchronize()
    elapsed=monotonic()-start
    cost=consumed(energy,snapshot())
    temp=path.with_suffix(".tmp")
    save_file({k:p.cpu().contiguous() for k,p in net.state_dict().items()},str(temp))
    temp.replace(path)
    result=dict(corpus=signature,weights=sha(path),schedule=digest.hexdigest(),steps=STEPS,
                exposures=STEPS*BATCH,distinct=int(used.sum()),seconds=elapsed,energy=cost,
                peak=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in net.parameters()),trace=trace)
    dump(receipt,result)
    return result


def main():
    obs=datafile("observations.npz");data=datafile("train.npz")
    with gpu_owner():
        fits={m:train(m,obs,data,sha(PUB/"corpus.json")) for m in ("ink","byte")}
    if fits["byte"]["schedule"]!=fits["ink"]["schedule"]:
        raise ValueError("Scheduling mismatch")
    dump(MODEL/"weights.json",dict(weights={k:v["weights"] for k,v in fits.items()}))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()

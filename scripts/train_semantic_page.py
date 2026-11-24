"""Fit a visual or matched byte reader with the same causal physical glyph writer."""
import hashlib
import json
import math
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_page import SemanticPage,BLOCK
from scripts.prepare_semantic_page import WORK,PUB,MODEL,sha,dump
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot,consumed

STEPS=9000
BATCH=48


def read(p):
    return json.loads(p.read_text())


def archive(p,digest):
    if sha(p)!=digest:raise ValueError("Changed input "+p.name)
    with np.load(p) as z:return dict(z)


def palette(device):
    meta=read(PUB/"corpus.json")
    p=MODEL/"palette.npy"
    if sha(p)!=meta["palette"]:raise ValueError("Changed output images")
    return torch.as_tensor(observations({"tiles":np.load(p)},"ink"),device=device)


def load(mode,device):
    info=read(PUB/(mode+"-fit.json"))
    p=MODEL/(mode+".safetensors")
    if sha(p)!=info["weights"]:raise ValueError("Changed weights")
    net=SemanticPage(info["classes"],info["concepts"]).to(device).eval()
    net.load_state_dict(load_file(str(p),device=str(device)))
    return net


def train(mode,data,bank,meta):
    receipt=PUB/(mode+"-fit.json")
    path=MODEL/(mode+".safetensors")
    if receipt.exists():
        r=read(receipt)
        if r["weights"]!=sha(path) or r["data"]!=meta["archives"]["train"]:raise ValueError("Changed fit")
        return r
    obs=torch.as_tensor(observations(bank,mode),device="cuda")
    rows=obs[bank["surface"]]
    output=palette("cuda")
    levels=rows.shape[1]
    torch.manual_seed(29041)
    rng=np.random.default_rng(29041)
    net=SemanticPage(len(output),levels).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,fused=True,weight_decay=.025)
    n=(data["y"]>=0).sum(1)
    bucket=(n-1)//32
    pools=[np.flatnonzero(bucket==i) for i in range(max(bucket)+1)]
    touched=np.zeros(len(n),bool);digest=hashlib.sha256()
    energy=snapshot();clock=monotonic();trace=[];cells=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,STEPS+1):
        ix=rng.choice(pools[bucket[rng.integers(len(n))]],BATCH)
        touched[ix]=True
        lang=rng.integers(2,size=(BATCH,1))
        order=rng.random((BATCH,data["slots"].shape[1])).argsort(1)
        for a in (ix,lang,order):digest.update(a.tobytes())
        labels=np.take_along_axis(data["slots"][ix],order,1)
        selected=torch.as_tensor(labels,device="cuda").long()
        visible=selected!=0
        width=int(math.ceil(n[ix].max()/BLOCK)*BLOCK)
        raw=data["y"][ix,:width]
        if raw.shape[1]<width:raw=np.pad(raw,((0,0),(0,width-raw.shape[1])),constant_values=-1)
        y=torch.as_tensor(raw.astype(np.int64),device="cuda")
        cells+=int(n[ix].sum())
        with torch.autocast("cuda",dtype=torch.bfloat16):
            # Encode each distinct physical row once; index gathering accumulates all derivatives.
            table=net.encode_rows(rows.flatten(0,1)).reshape(2,levels,-1)
            memory=table[torch.as_tensor(lang,device="cuda"),selected]
            logits=net(memory,visible,output[y.clamp_min(0)])
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-1)
            concept_loss=F.cross_entropy(net.concept(memory).float().flatten(0,1),selected.flatten())
            total=loss+.2*concept_loss
        opt.zero_grad(set_to_none=True);total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.param_groups[0]["lr"]=.001*min(1.,step/250)*(.12+.88*(1+math.cos(math.pi*step/STEPS))/2)
        opt.step()
        if step==1 or step%500==0:
            value=total.item();elapsed=monotonic()-clock
            trace.append([step,value,elapsed]);print(mode,step,round(value,3),round(elapsed,1),flush=True)
            if not math.isfinite(value) or elapsed>1200 or torch.cuda.max_memory_allocated()>12*2**30:
                raise RuntimeError("Fit exceeded numerical or resource ceiling")
    torch.cuda.synchronize();elapsed=monotonic()-clock;cost=consumed(energy,snapshot())
    temp=path.with_suffix(".tmp");save_file({k:v.cpu().contiguous() for k,v in net.state_dict().items()},str(temp));temp.replace(path)
    info=dict(weights=sha(path),classes=len(output),concepts=levels,data=meta["archives"]["train"],
              distinct=int(touched.sum()),exposures=STEPS*BATCH,targets=cells,schedule=digest.hexdigest(),
              seconds=elapsed,peak=torch.cuda.max_memory_allocated(),energy=cost,trace=trace,parameters=sum(p.numel() for p in net.parameters()))
    dump(receipt,info);return info


def main():
    m=read(PUB/"corpus.json")
    data=archive(WORK/"train.npz",m["archives"]["train"])
    b=archive(WORK/"bank.npz",m["bank"])
    with gpu_owner():
        fits=[train(mode,data,b,m) for mode in ("ink","byte")]
    if fits[0]["schedule"]!=fits[1]["schedule"]:raise ValueError("Mismatched schedule")


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()

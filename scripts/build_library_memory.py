"""Build train-only image memories and estimate their value on separate books."""

import math
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.glyph_association import GlyphMemory, source_code
from scripts.sample_visual_library import ROOT, RUN as PARENT, EVIDENCE as LIB, MODELS, RECEIPT, library, read, sha
from scripts.prepare_teacher_translation import dump
from scripts.train_visual_prefix import gpu_owner, pack, material, atomic_save
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli

RUN=ROOT/"artifacts/library_memory_v1"
PUB=ROOT/"references/evidence/library_memory_v1"
TEMPERATURES=(.02,.05,.1)
MIX=(0.,.1,.25,.5,.75)


def hidden(net,bank,x,y,t):
    ctx,valid=net.encode(bank,x)
    query=torch.cat([net.begin[t].expand(len(x),1,-1),bank[y[:,:-1]]],dim=1)
    states,_=net.decode(query,ctx,valid)
    return states,source_code(ctx,valid)


@torch.inference_mode()
def index(net, rows, bank, mapping,t):
    lengths=np.asarray([len(r["cells"][t])+1 for r in rows])
    offsets=np.r_[0,np.cumsum(lengths)]
    width=bank.size(1)
    states=np.empty((offsets[-1],width),np.float16)
    sources=np.empty((len(rows),width),np.float16)
    labels=np.empty(offsets[-1],np.int16)
    preceding=np.empty_like(labels)
    for i in range(0,len(rows),96):
        batch=rows[i:i+96]
        x=pack([r["cells"][1-t] for r in batch],"cuda")
        y=pack([r["cells"][t]+[1] for r in batch],"cuda")
        h,s=hidden(net,bank,x,y,t)
        mask=y!=0
        take=slice(offsets[i],offsets[i+len(batch)])
        states[take]=F.normalize(h[mask],dim=-1).half().cpu().numpy()
        labels[take]=mapping[y[mask]].cpu().numpy()
        previous=F.pad(y[:,:-1],(1,0))
        preceding[take]=previous[mask].cpu().numpy()
        sources[i:i+len(batch)]=s.half().cpu().numpy()
    if labels.min()<0:
        raise ValueError("Glyph outside direction repertoire")
    shuffled=labels.copy()
    rng=np.random.default_rng(193663+t)
    order=np.argsort(preceding,kind="stable")
    for group in np.split(order,np.flatnonzero(np.diff(preceding[order]))+1):
        shuffled[group]=rng.permutation(labels[group])
    return dict(states=states,sources=sources,labels=labels,shuffled=shuffled,offsets=offsets)


@torch.inference_mode()
def validation(net, bank, inv, mapping, dev,t, memory):
    total={}
    n=0
    accessible=0
    for j in range(0,len(dev),12):
        rr=dev[j:j+12]
        x=pack([r["cells"][1-t] for r in rr],"cuda")
        y=pack([r["cells"][t]+[1] for r in rr],"cuda")
        h,s=hidden(net,bank,x,y,t)
        sim,observed,shuffled=memory.neighbors(h,memory.passages(s))
        mask=y!=0
        label=mapping[y].clamp_min(0).unsqueeze(-1)
        logits=h@net.prototypes(bank,inv).T/math.sqrt(h.size(-1))
        base=logits.float().softmax(-1).gather(-1,label).squeeze(-1)[mask].double()
        accessible+=int((observed==label).any(-1)[mask].sum())
        for method,values in (("observed",observed),("shuffled",shuffled)):
            eq=values==label
            for temp in TEMPERATURES:
                mass=((sim/temp).softmax(-1)*eq).sum(-1)[mask].double()
                for mix in MIX:
                    k=(method,temp,mix)
                    total[k]=total.get(k,0.)-float(((1-mix)*base+mix*mass).log().sum())
        n+=int(mask.sum())
    return total,n,accessible


def main():
    configure_cuda_cli()
    RUN.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    binding={str(p.relative_to(ROOT)):sha(p) for p in (RECEIPT,Path(__file__).resolve(),
              ROOT/"ilm/visual_lm/glyph_association.py",ROOT/"references/library_associative_memory_protocol.md",
              MODELS/"broad/weights.safetensors")}
    train,dev=library("train"),library("dev")
    receipt=PUB/"calibration.json"
    if receipt.exists():
        r=read(receipt)
        if r["binding"]!=binding or any(sha(RUN/name)!=digest for name,digest in r["files"].items()):
            raise ValueError("Existing memory differs")
        return
    total={}
    count=reachable=0
    files={}
    positions=[]
    started=monotonic()
    with gpu_owner(),torch.inference_mode():
        net=restore("library-broad","cuda")
        f,inv,maps=material("cuda")
        bank=net.optical(f)
        for t in (0,1):
            p=RUN/f"memory-{t}.npz"
            stamp=p.with_suffix(".json")
            if not stamp.exists():
                data=index(net,train,bank,maps[t],t)
                def write(tmp):
                    with open(tmp,"wb") as fp:
                        np.savez(fp,**data)
                atomic_save(write,p)
                dump(stamp,dict(sha256=sha(p),binding=binding))
                del data
            r=read(stamp)
            if r["binding"]!=binding or sha(p)!=r["sha256"]:
                raise ValueError("Corrupt memory index")
            files[p.name]=r["sha256"]
            with np.load(p) as archive:
                memory=GlyphMemory(archive,"cuda")
            positions.append(len(memory.states))
            vv,n,hits=validation(net,bank,inv[t],maps[t],dev,t,memory)
            for k,v in vv.items():
                total[k]=total.get(k,0.)+v
            count+=n
            reachable+=hits
            del memory
        peak=torch.cuda.max_memory_allocated()
    values=[dict(method=k[0],temperature=k[1],mix=k[2],nll=v/count) for k,v in total.items()]
    best=min((v for v in values if v["method"]=="observed"),key=lambda v:(v["nll"],v["mix"],v["temperature"]))
    result=dict(binding=binding,files=files,positions=positions,validation_positions=count,
                neighbor_contains_target=reachable,grid=values,selected=best,seconds=monotonic()-started,
                bytes=sum((RUN/k).stat().st_size for k in files),peak_bytes=peak)
    dump(receipt,result)
    print(best,positions,count,reachable,result["bytes"],flush=True)


if __name__=="__main__":
    main()

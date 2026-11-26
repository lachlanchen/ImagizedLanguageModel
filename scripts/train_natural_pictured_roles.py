"""Matched visual context versus marked-layout training on authentic UD clauses."""
import math
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.natural_roles import NaturalRoleReader,tiles
from scripts.prepare_natural_pictured_roles import RUN,PUB,ROOT,MODELS,WINDOW,dump,read,sha
from scripts.train_field_to_glyph import atomic_save,configure_cuda_cli,gpu_owner

SEED=47651


def arrays(rows,codes):
    cells=np.zeros((len(rows),WINDOW),np.int32)
    marks=np.zeros_like(cells)
    for i,r in enumerate(rows):
        cells[i,:len(r["text"])]=[codes[c] for c in r["text"]]
        for label,(start,end) in enumerate((r["verb"],*r["candidates"]),1):
            assert 0<=start<end<=len(r["text"]) and not marks[i,start:end].any()
            marks[i,start:end]=label
    return cells,marks


def fit(arm,rows,palette):
    folder=MODELS/arm;folder.mkdir(exist_ok=True)
    out=folder/"weights.safetensors"
    binding={str(p.relative_to(ROOT)):sha(p) for p in (PUB/"corpus.json",Path(__file__),
        ROOT/"ilm/visual_lm/natural_roles.py",RUN/"codes.json",ROOT/"references/natural_pictured_roles_protocol.md")}
    if (folder/"fit.json").exists():
        receipt=read(folder/"fit.json")
        assert receipt["binding"]==binding and receipt["weights"]==sha(out)
        return
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    x,m=[torch.from_numpy(a).cuda() for a in arrays(rows,read(RUN/"codes.json"))]
    labels=torch.tensor([r["label"] for r in rows],device="cuda")
    groups=[np.array([i for i,r in enumerate(rows) if r["voice"]==v]) for v in ("active","passive")]
    body=NaturalRoleReader().cuda()
    optimizer=torch.optim.AdamW(body.parameters(),lr=.0007,weight_decay=.01,fused=True)
    epochs=[];start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(10):
        loss_sum=hits=0
        for j in range(100):
            step=epoch*100+j+1
            optimizer.param_groups[0]["lr"]=.0007*min(1,step/60)*(.15+.85*(1+math.cos(math.pi*step/1000))/2)
            indices=torch.from_numpy(rng.permutation(np.concatenate([rng.choice(g,32) for g in groups]))).cuda()
            ids=x[indices].long()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                scores=body.read(body.eye(palette)[ids],m[indices],ids.ne(0),arm=="layout")
                loss=F.cross_entropy(scores,labels[indices])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(body.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum+=loss.item();hits+=(scores.argmax(1)==labels[indices]).sum().item()
        epochs.append(dict(epoch=epoch+1,sampled=6400,mean_loss=loss_sum/100,training_correct=hits))
        print(arm,epoch+1,round(loss_sum/100,4),hits,flush=True)
        if monotonic()-start>1200 or torch.cuda.max_memory_allocated()>12*2**30:raise RuntimeError("Resource ceiling")
    atomic_save(lambda p:save_file({k:v.detach().cpu().contiguous() for k,v in body.state_dict().items()},p),out)
    dump(folder/"fit.json",dict(weights=sha(out),binding=binding,epochs=epochs,seed=SEED,
         parameters=sum(v.numel() for v in body.parameters()),seconds=monotonic()-start,peak=torch.cuda.max_memory_allocated()))


def main():
    configure_cuda_cli()
    receipt=read(PUB/"corpus.json")
    assert sha(RUN/"train.json")==receipt["split_hashes"]["train"] and sha(MODELS/"palette.npy")==receipt["palette"]
    with gpu_owner():
        rows=read(RUN/"train.json")
        palette=tiles(torch.from_numpy(np.load(MODELS/"palette.npy")).cuda())
        for arm in ("layout","context"):
            fit(arm,rows,palette)


if __name__=="__main__": main()

"""Fit a source-conditioned span locator and a matched positional control."""
import math
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from ilm.visual_lm.role_localization import RoleLocator
from ilm.visual_lm.natural_roles import tiles
from scripts.prepare_natural_pictured_roles import ROOT,RUN,PUB,MODELS as MARKED,read,dump,sha
from scripts.train_natural_pictured_roles import arrays
from scripts.train_field_to_glyph import atomic_save,configure_cuda_cli,gpu_owner

OUT=ROOT/"models/visual_role_locator"
SEED=196021


def fit(arm,rows,palette):
    folder=OUT/arm;folder.mkdir(parents=True,exist_ok=True)
    weights=folder/"weights.safetensors"
    binding={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/"ilm/visual_lm/role_localization.py",
        ROOT/"ilm/visual_lm/natural_roles.py",ROOT/"ilm/visual_lm/ordered_translation.py",PUB/"corpus.json",
        ROOT/"references/visual_role_localization_protocol.md")}
    if (folder/"fit.json").exists():
        receipt=read(folder/"fit.json");assert receipt["binding"]==binding and sha(weights)==receipt["weights"]
        return
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    cells,marks=arrays(rows,read(RUN/"codes.json"))
    x=torch.from_numpy(cells).cuda();verb=torch.from_numpy(marks==1).cuda()
    gold=[]
    for r in rows:
        a,b=[r["candidates"][k] for k in (r["label"],1-r["label"])]
        assert a[0]<a[1] and b[0]<b[1]
        gold.append((a[0],a[1]-1,b[0],b[1]-1))
    labels=torch.tensor(gold,device="cuda")
    groups=[np.array([i for i,r in enumerate(rows) if r["voice"]==v]) for v in ("active","passive")]
    model=RoleLocator().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0007,fused=True)
    start=monotonic();stats=[];loss_sum=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,2401):
        idx=torch.from_numpy(rng.permutation(np.r_[rng.choice(groups[0],48),rng.choice(groups[1],16)])).cuda()
        ids=x[idx].long()
        opt.param_groups[0]["lr"]=.0007*min(1,step/100)*(.15+.85*(1+math.cos(math.pi*step/2400))/2)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            scores=model.read(model.eye(palette)[ids],verb[idx],ids.ne(0),arm=="position")
            loss=F.cross_entropy(scores.flatten(0,1),labels[idx].flatten())
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();loss_sum+=loss.item()
        if step%200==0:
            stats.append(dict(updates=step,loss=loss_sum/200));loss_sum=0
            print(arm,stats[-1],flush=True)
            if monotonic()-start>900 or torch.cuda.max_memory_allocated()>12*2**30:raise RuntimeError("Resource ceiling")
    atomic_save(lambda p:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},p),weights)
    dump(folder/"fit.json",dict(binding=binding,seed=SEED,weights=sha(weights),stats=stats,examples=2400*64,
           parameters=sum(p.numel() for p in model.parameters()),seconds=monotonic()-start,peak=torch.cuda.max_memory_allocated()))


def main():
    configure_cuda_cli()
    with gpu_owner():
        receipt=read(PUB/"corpus.json")
        assert sha(RUN/"train.json")==receipt["split_hashes"]["train"] and sha(MARKED/"palette.npy")==receipt["palette"]
        rows=read(RUN/"train.json")
        palette=tiles(torch.from_numpy(np.load(MARKED/"palette.npy")).cuda())
        for arm in ("position","context"):fit(arm,rows,palette)


if __name__=="__main__":main()

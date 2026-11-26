"""Fit matched aligned and shuffled-source visual content predictors."""
from hashlib import sha256
import math
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from ilm.visual_lm.optical_actions import OpticalActions,pool_tiles
from ilm.visual_lm.source_support import SourceSupport
from scripts.reserve_independent_optical_actions import RUN as CORPUS, PUB as PREP
from scripts.reserve_source_supported_writing import BASE,PROTOCOL,PUB,ROOT,RUN,dump,read,sha
from scripts.train_independent_optical_actions import schedule, atomic_save, configure_cuda_cli, gpu_owner

MODELS = ROOT/"models/source_supported_writing"
PARENT = ROOT/"models/independent_optical_actions/strip"
ARMS = ("aligned","shuffled")
EPOCHS = 4


def incidences(targets, num):
    hits = np.zeros(num,dtype=np.int64)
    for batch in np.array_split(targets, 256):
        rows,cols = np.nonzero(batch)
        ids = np.unique(rows*num + batch[rows,cols].astype(np.int64)) % num
        hits += np.bincount(ids,minlength=num)
    return torch.from_numpy(((hits+.5)/(len(targets)+1)).astype(np.float32)).logit()


def fit(arm):
    prep = read(PREP/"preparation.json")
    assert sha(RUN/"panel.json")==read(PUB/"panel.json")["sha256"]
    for name in ("cells.npy","strip_0.npy","strip_1.npy","tables.json"):
        assert sha(CORPUS/name)==prep["files"][name]
    parent = PARENT/"weights.safetensors"
    assert sha(parent)==read(PARENT/"fit.json")["weights"]
    binding = dict(parent=sha(parent),panel=sha(PUB/"panel.json"),prep=sha(PREP/"preparation.json"),
                   protocol=sha(PROTOCOL),code=sha(Path(__file__)),head=sha(ROOT/"ilm/visual_lm/source_support.py"),
                   schedule=sha(ROOT/"scripts/train_independent_optical_actions.py"))
    out = MODELS/arm
    out.mkdir(exist_ok=True,parents=True)
    if (out/"fit.json").exists():
        rec=read(out/"fit.json")
        assert rec["binding"]==binding and rec["weights"]==sha(out/"weights.safetensors")
        return
    torch.manual_seed(21877)
    torch.set_num_threads(2)
    body = OpticalActions().cuda().eval().requires_grad_(False)
    body.load_state_dict(load_file(str(parent)))
    raw = np.load(CORPUS/"cells.npy")
    lens = np.count_nonzero(raw,axis=2)
    buckets = (lens[:,0]//12)*10+lens[:,1]//12
    src = torch.tensor(raw.astype(np.int32),device="cuda")
    targets = [np.load(CORPUS/f"strip_{t}.npy") for t in (0,1)]
    choices = [len(x) for x in read(CORPUS/"tables.json")["strip"]]
    head = SourceSupport(choices).cuda()
    for t in (0,1):
        getattr(head,f"prior_{t}").copy_(incidences(targets[t],choices[t]))
    gold = [torch.tensor(a.astype(np.int32),device="cuda") for a in targets]
    tiles = pool_tiles(torch.from_numpy(np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32)).cuda())
    opt = torch.optim.AdamW(head.parameters(),lr=.0015,fused=True)
    resume = RUN/f"{arm}-epoch.pt"
    records = []
    if resume.exists():
        ck=torch.load(resume,map_location="cuda",weights_only=True)
        assert ck["binding"]==binding
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        records=ck["epochs"]
    prior_seconds = sum(r["seconds"] for r in records)
    started = monotonic()
    for epoch in range(len(records),EPOCHS):
        clock=monotonic()
        batches=schedule(buckets,epoch+4)
        total=singles=0
        cost=torch.zeros((),dtype=torch.float64,device="cuda")
        order=sha256()
        for j,(indices,t) in enumerate(batches):
            source_indices = np.roll(indices,1) if arm=="shuffled" else indices
            singles += int(len(indices)==1)
            order.update(bytes([t])+indices.astype("<i4").tobytes())
            width=int(lens[source_indices,1-t].max())
            ix=torch.from_numpy(source_indices).cuda()
            ids=src[ix,1-t,:width].long()
            refs=gold[t][torch.from_numpy(indices).cuda()].long()
            step=epoch*len(batches)+j+1
            rate=min(1,step/200)*(.2+.8*(1+math.cos(math.pi*step/(EPOCHS*len(batches))))/2)
            opt.param_groups[0]["lr"]=.0015*rate
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                memory,valid=body.encode(tiles[ids],ids.ne(0).sum(1))
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=head.loss(memory,valid,refs,t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),2.,error_if_nonfinite=True)
            opt.step()
            total+=len(indices)
            cost+=loss.detach()*len(indices)
            if j%500==0:
                print(arm,epoch+1,j,len(batches),round(loss.item(),5),round(monotonic()-clock,1),flush=True)
                if prior_seconds+monotonic()-started>10800 or torch.cuda.max_memory_allocated()>20*2**30:
                    raise RuntimeError("Training resource ceiling reached")
        assert total==2*len(raw)
        records.append(dict(epoch=epoch+1,examples=total,updates=len(batches),order=order.hexdigest(),
                            singleton_batches=singles,loss=cost.item()/total,seconds=monotonic()-clock,
                            allocated=torch.cuda.max_memory_allocated()))
        atomic_save(lambda p:torch.save(dict(binding=binding,epochs=records,head=head.state_dict(),opt=opt.state_dict()),p),resume)
    path=out/"weights.safetensors"
    atomic_save(lambda p:save_file({k:v.detach().cpu() for k,v in head.state_dict().items()},p),path)
    dump(out/"fit.json",dict(binding=binding,arm=arm,weights=sha(path),epochs=records,choices=choices,
                             parameters=sum(p.numel() for p in head.parameters()),gpu=torch.cuda.get_device_name()))


if __name__=="__main__":
    configure_cuda_cli()
    with gpu_owner():
        for arm in ARMS:
            torch.cuda.reset_peak_memory_stats()
            fit(arm)

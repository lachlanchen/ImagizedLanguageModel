"""Train and contrast a conditional visual fact reader with an induced bitmap writer."""
import fcntl
import hashlib
import json
import os
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.bound_fact_reader import FactHead, features_batch, render_fact, contextualize
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from scripts.evaluate_pointing_causal_writer import READER
from scripts.prepare_visual_facts import Face,FONTS,WORK,PUBLIC,ROOT,RELATIONS,SEED,statement,write_json


def rows(split, *, eligible_only=False):
    from scripts.span_reservation import screened
    file=WORK/(split+".jsonl")
    expected=json.loads((PUBLIC/"corpus.json").read_text())["sha256"][split]
    if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError("Changed split")
    return screened([json.loads(line) for line in file.read_text().splitlines()], eligible_only=eligible_only)


def visual_cache(split,data,face,reader):
    folder=WORK/("features_"+split)
    folder.mkdir(exist_ok=True)
    cache=[]
    for start in range(0,len(data),256):
        chunk=data[start:start+256]
        path=folder/f"{start:06}.pt"
        identity=hashlib.sha256(json.dumps([(r["p"],r["q"]) for r in chunk],ensure_ascii=False).encode()).hexdigest()
        if path.exists():
            obj=torch.load(path,weights_only=True,map_location="cpu")
            if identity!=obj["inputs"]:raise ValueError("Mismatched saved feature batch")
            values=obj["features"]
        else:
            values=contextualize(reader,[(face(x["p"]),face(x["q"])) for x in chunk],batch_size=24)
            temp=path.with_suffix(".partial")
            torch.save(dict(inputs=identity,features=values),temp);temp.replace(path)
        cache.extend(values)
        if start%4096==0:print(split,start,len(data),flush=True)
    return cache


def frame_table(data,face):
    frames=[None]*len(RELATIONS)
    for x in data:
        s,r,o=x["s"],x["r"],x["o"]
        target=statement(s,r,o)
        a,b=target.index(s),target.rindex(o)
        fr=[face(target[:a]),face(target[a+len(s):b]),face(target[b+len(o):])]
        k=RELATIONS.index(r)
        if frames[k] is None:frames[k]=fr
        if fr!=frames[k] or render_fact(fr,face(s),face(o))!=face(target):raise ValueError("Inconsistent aligned examples")
    if any(f is None for f in frames):raise ValueError("Uncovered relation")
    return [[v.hex() for v in fr] for fr in frames]


def fit(arm,data,values,frames):
    torch.manual_seed(SEED)
    head=FactHead(len(frames)).cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=0.001,weight_decay=.01)
    truth=[]
    for x in data:
        p,s,o=x["p"],x["s"],x["o"]
        a,c=p.index(s),p.index(o)
        truth.append([RELATIONS.index(x["r"]),a,a+len(s)-1,c,c+len(o)-1])
    labels=np.asarray(truth)
    if arm=="shuffled":
        noise=np.random.default_rng(SEED+1)
        labels[:,0]=noise.permutation(labels[:,0])
        swap=noise.random(len(labels))<.5
        labels[swap,1:]=labels[swap,1:][:,[2,3,0,1]]
    labels=torch.tensor(labels,device="cuda")
    rng=np.random.default_rng(SEED+2)
    losses=[]; begin=monotonic(); updates=0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(6):
        cost=0.
        # Shuffle buckets before sorting small pools by length to reduce padding.
        for pool in np.array_split(rng.permutation(len(data)),max(1,len(data)//768)):
            ordered=sorted(pool,key=lambda i:len(values[i][1]))
            for start in range(0,len(ordered),48):
                ids=ordered[start:start+48]
                x=features_batch([values[i] for i in ids],"cuda")
                role,ends=head(*x)
                target=labels[ids]
                loss=F.cross_entropy(role,target[:,0])+sum(F.cross_entropy(ends[:,:,k],target[:,k+1]) for k in range(4))
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),2.);opt.step()
                cost+=float(loss.detach())*len(ids);updates+=1
        losses.append(cost/len(data));print(arm,epoch+1,losses[-1],flush=True)
    dest=WORK/arm;dest.mkdir(exist_ok=False)
    save_file({k:v.cpu().contiguous() for k,v in head.state_dict().items()},str(dest/"head.safetensors"))
    write_json(dest/"frames.json",frames)
    write_json(PUBLIC/(arm+"_fit.json"),dict(rows=len(data),updates=updates,loss=losses,seconds=monotonic()-begin,
        parameters=sum(p.numel() for p in head.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        weights=hashlib.sha256((dest/"head.safetensors").read_bytes()).hexdigest()))


def train():
    face=Face((FONTS[0],2))
    reader=FontInvariantReader.restore(READER,"cuda")
    data=rows("training")
    frames=frame_table(data,face)
    t=monotonic();values=visual_cache("training",data,face,reader)
    write_json(PUBLIC/"feature_cost.json",dict(elapsed_seconds=monotonic()-t,
        trained_backbone_parameters=sum(p.numel() for p in reader.parameters()),peak_bytes=torch.cuda.max_memory_allocated()))
    for arm in ("genuine","shuffled"):
        fit(arm,data,values,frames)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(3);train()

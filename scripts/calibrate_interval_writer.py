"""Calibrate only the decision to copy or compose using original training targets."""
import fcntl
import os
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F

from ilm.visual_lm.interval_writer import IntervalWriter
from scripts.train_general_entry_writer import Entries, small
from scripts.train_interval_writer import (
    Corpus, DEVICE, PRIVATE, PUBLIC, ROOT, START, WORK, PairCorpus, SentenceCorpus,
    load, load_visual_writer, memory_bank, physical_tables, references_in, sha, to_batch, write_json)
from scripts.train_inverse_role_writer import InverseScenes
from scripts.train_visual_role_writer import Tiles
from scripts.visual_role_examples import TRAIN_NAMES,predicates,situation

DEST=WORK/"calibrated"
SEED=77227


def fit():
    DEST.mkdir()
    torch.set_num_threads(2)
    rng=np.random.default_rng(SEED)
    bank=memory_bank()
    ds=SentenceCorpus(PairCorpus(bank,"train"),bank,"train")
    nli=Corpus(bank,"train");terms=predicates()["fit"];scenes=InverseScenes(bank,terms)
    du,_=small(bank,"train");ent=Entries(bank,"train")
    writer=load_visual_writer(START,DEVICE).requires_grad_(False)
    feat,mem,_=physical_tables(bank,DEVICE);ink=writer.ink(feat).detach()
    head=IntervalWriter().to(DEVICE);initial=WORK/"matched/interval.safetensors"
    assert sha(initial)==load(WORK/"matched/fit.json")["weights"]
    old=load_file(str(initial));head.load_state_dict(old)
    head.requires_grad_(False);head.action.requires_grad_(True)
    opt=torch.optim.AdamW(head.action.parameters(),lr=.001)
    costs=[];totals=np.zeros(2,dtype=np.int64)
    with np.load(PRIVATE) as z:train=z["fit"]
    ctrl=np.random.default_rng(SEED+1)
    torch.cuda.reset_peak_memory_stats();start=monotonic()
    for step in range(640):
        data=[]
        for i in rng.integers(len(ds.data.questions),size=24):
            p,q,span=ds.common(i,int(rng.integers(2)));data.append((p,q,bool(span)))
        for j in rng.integers(len(ds.pairs),size=3):
            data.extend((p,q,bool(span)) for p,q,span in ds.pair(ds.pairs[j]))
        examples=scenes.sample(rng,ctrl,False)[::3]
        for _ in range(3):
            names=rng.choice(TRAIN_NAMES,2,replace=False)
            rows=situation(terms[rng.integers(len(terms))],names,bool(rng.integers(2)),int(rng.integers(2)),"write")
            examples.extend(tuple(scenes.tile(s) for s in row) for row in rows)
        for i in rng.choice(train,4,replace=False):
            examples.append((nli.p[i],nli.q[i],nli.y[i]))
        valid,holes,ys=nli.holes(rng.choice(train,4,replace=False),rng)
        examples.extend((nli.p[i],q,y) for i,q,y in zip(valid,holes,ys,strict=True))
        for i in rng.integers(len(ent.rows),size=4):
            k=ent.rows[i];examples.append((ent.cards[ent.card_of[k]],ent.q[i],ent.y[k]))
        for i in rng.integers(len(du.y),size=8):
            examples.append((du.p[i],du.q[i],du.y[i]))
        data.extend((p,q,bool(references_in(p,y))) for p,q,y in examples)
        data.sort(key=lambda x:len(x[0]))
        opt.zero_grad(set_to_none=True);total=0.
        for i in range(0,len(data),24):
            chunk=data[i:i+24];ps,qs,labels=zip(*chunk,strict=True)
            p,q=to_batch(ps,qs);target=torch.tensor(labels,device=DEVICE,dtype=torch.float32)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                a,_=head(writer.encode(ink,p,q),p)
                loss=F.binary_cross_entropy_with_logits(a,target,reduction="sum")/len(data)
            loss.backward();total+=float(loss.detach())
            totals+=np.bincount(np.array(labels,int),minlength=2)
        torch.nn.utils.clip_grad_norm_(head.action.parameters(),1.,error_if_nonfinite=True)
        opt.step();costs.append(total)
        if monotonic()-start>300 or torch.cuda.max_memory_allocated()>8*1024**3:raise RuntimeError("Budget exceeded")
    new={k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}
    assert all(torch.equal(v,old[k]) for k,v in new.items() if not k.startswith("action."))
    path=DEST/"interval.safetensors";save_file(new,str(path))
    write_json(DEST/"fit.json",dict(base=sha(START/"final.safetensors"),weights=sha(path),source=sha(initial),
              seconds=monotonic()-start,parameters=sum(p.numel() for p in head.action.parameters()),steps=len(costs),
              peak_bytes=torch.cuda.max_memory_allocated(),fallback_examples=int(totals[0]),read_examples=int(totals[1]),
              mean_loss=float(np.mean(costs)),all_interval_tensors_unchanged=True))


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("Use GPU0")
    binding=load(PUBLIC/"calibration_execution.json")["files"]
    assert all(sha(ROOT/k)==v for k,v in binding.items())
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fit()
        from scripts.evaluate_calibrated_writer import evaluate
        evaluate()

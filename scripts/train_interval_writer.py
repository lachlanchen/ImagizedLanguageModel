"""Train a source-run action from pictured evidence and paired questions on one GPU."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.interval_writer import IntervalWriter, MAX_RUN
from scripts.shared_sentence_corpus import SentenceCorpus
from scripts.train_inverse_role_writer import (Corpus, DEVICE, DRCD, PARENT, PRIVATE, ROOT, WHO, WORK as PREVIOUS,
    PairCorpus, load, load_visual_writer, memory_bank, physical_tables, sha, to_batch, write_json)

WORK=ROOT/"artifacts/shared_sentence_writer_v1"
PUBLIC=ROOT/"references/evidence/shared_sentence_writer_20261119"
START=PREVIOUS/"matched"
SEED,EPOCHS,BATCH=91991,8,56


def bindings():
    files=[Path(__file__), Path(__file__).with_name("evaluate_interval_writer.py"),
           Path(__file__).with_name("shared_sentence_corpus.py"), ROOT/"ilm/visual_lm/interval_writer.py",
           ROOT/"references/shared_sentence_writer_protocol.md",DRCD/"preparation.json",WHO/"cohort.npz",
           START/"final.safetensors",PARENT/"memory.npy",PRIVATE,
           ROOT/"references/evidence/inverse_role_writer_20261119/result.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def references_in(p,y):
    if not 0<len(y)<=min(MAX_RUN,len(p)):
        return []
    matches=np.flatnonzero(p[:len(p)-len(y)+1]==y[0])
    return [(int(a),int(a)+len(y)) for a in matches if np.array_equal(p[a:a+len(y)],y)]


def objective(locator,writer,ink,rows):
    passages,questions,refs=zip(*rows,strict=True)
    p,q=to_batch(passages,questions)
    with torch.no_grad():
        encoded=writer.encode(ink,p,q)
    action,scores=locator(encoded,p)
    positive = torch.tensor([bool(s) for s in refs],device=p.device)
    gate=F.binary_cross_entropy_with_logits(action,positive.float())
    if not positive.any():
        return gate
    # Only score reading rows. A fallback-only row has no legal target.
    indices=[[(a*MAX_RUN+b-a-1) for a,b in spans] for spans in refs if spans]
    width=max(map(len,indices))
    target=torch.tensor([s+[-1]*(width-len(s)) for s in indices],device=p.device)
    flat=scores[positive].flatten(1)
    correct=flat.gather(1,target.clamp_min(0)).masked_fill(target<0,-torch.inf)
    return gate+(flat.logsumexp(-1)-correct.logsumexp(-1)).sum()/len(rows)


def fit(arm,data,nli,writer,ink):
    dest=WORK/arm;dest.mkdir()
    torch.manual_seed(SEED)
    locator=IntervalWriter().to(DEVICE)
    optimizer=torch.optim.AdamW(locator.parameters(),lr=.001)
    with np.load(PRIVATE) as z: train=z["fit"]
    count=math.ceil(len(data.data.questions)/BATCH)
    torch.cuda.reset_peak_memory_stats()
    started=monotonic()
    trace=[];updates=0;presentations=np.zeros(3,dtype=int)
    for epoch in range(EPOCHS):
        rng=np.random.default_rng(SEED+epoch)
        swaps=np.random.default_rng(SEED+100+epoch)
        singles=np.array_split(rng.permutation(len(data.data.questions)),count)
        hard=list(data.pairs)+3*list(data.people)
        paired=np.array_split(rng.permutation(len(hard)),count)
        cost=0.
        for ids,js in zip(singles,paired,strict=True):
            rows=[data.common(i,epoch) for i in ids]
            for j in js:
                flip=swaps.integers(2)
                rows.extend(data.pair(hard[j],exchange=bool(flip and arm=="randomized")))
            ni=rng.choice(train,16,replace=False)
            replay=[(nli.p[i],nli.q[i],nli.y[i]) for i in ni[:8]]
            valid,holes,ys=nli.holes(ni[8:],rng)
            replay.extend((nli.p[i],q,y) for i,q,y in zip(valid,holes,ys,strict=True))
            rows.extend((p,q,references_in(p,y)) for p,q,y in replay)
            presentations += [len(ids),2*len(js),len(replay)]
            updates+=1
            optimizer.param_groups[0]["lr"]=min(1,updates/50)*(.00012+.00088*(1+math.cos(math.pi*updates/(EPOCHS*count)))/2)
            optimizer.zero_grad(set_to_none=True)
            # Use lengths only for microbatch packing. All rows retain identical loss weight.
            rows.sort(key=lambda x:len(x[0]))
            for j in range(0,len(rows),32):
                micro=rows[j:j+32]
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss=objective(locator,writer,ink,micro)*len(micro)/len(rows)
                loss.backward()
                cost+=loss.detach().item()
            torch.nn.utils.clip_grad_norm_(locator.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if monotonic()-started>1200 or torch.cuda.max_memory_allocated()>8*1024**3:
                raise RuntimeError("Fit exceeded its resource ceiling")
        trace.append(dict(epoch=epoch+1,mean_loss=cost/count,updates=updates))
        print(arm,trace[-1],flush=True)
    path=dest/"interval.safetensors"
    save_file({k:v.detach().cpu().contiguous() for k,v in locator.state_dict().items()},str(path))
    write_json(dest/"fit.json",dict(weights=sha(path),base=sha(START/"final.safetensors"),epochs=trace,
        updates=updates,presentations=presentations.tolist(),seconds=monotonic()-started,
        peak_bytes=torch.cuda.max_memory_allocated(),new_parameters=sum(p.numel() for p in locator.parameters()),
        frozen_parameters=sum(p.numel() for p in writer.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--register",action="store_true")
    reg=PUBLIC/"registration.json"
    if parser.parse_args().register:
        PUBLIC.mkdir(exist_ok=True,parents=True)
        if reg.exists():raise FileExistsError(reg)
        write_json(reg,bindings());return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or bindings()!=load(reg):raise ValueError("Inputs changed")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir();torch.set_num_threads(2)
        bank=memory_bank();data=SentenceCorpus(PairCorpus(bank,"train"),bank,"train")
        nli=Corpus(bank,"train");writer=load_visual_writer(START,DEVICE).requires_grad_(False)
        features,_,_=physical_tables(bank,DEVICE)
        ink=writer.ink(features).detach()
        print("train pairs",len(data.pairs),"human who",len(data.people),flush=True)
        for arm in ("matched","randomized"):
            fit(arm,data,nli,writer,ink)
        from scripts.evaluate_interval_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()

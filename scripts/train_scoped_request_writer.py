"""Compare identical global question adapters with and without relative reading distance."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.scoped_question import ScopedQuestion
from scripts.prepare_long_scope_questions import TRAIN, long_question, requested_words
from scripts.train_english_request_writer import (DEVICE, FACT, FORM, PARENT, PUBLIC as PREVIOUS_PUBLIC,
          ROOT, WORK as PREVIOUS, SEED, english_batch, inputs, load, load_writer, mean_logp, memory_bank,
          padded, physical_tables, sha, write_json)

WORK=ROOT/"artifacts/scoped_request_writer_v1"
PUBLIC=ROOT/"references/evidence/scoped_request_writer_20261119"
REG=ROOT/"references/evidence/scoped_request_writer_registration.json"
BATCH,EPOCHS=48,8


def binding():
    paths=[Path(__file__),ROOT/"scripts/evaluate_scoped_request_writer.py",ROOT/"scripts/load_scoped_writer.py",
           ROOT/"ilm/visual_lm/scoped_question.py",ROOT/"scripts/prepare_long_scope_questions.py",
           ROOT/"references/scoped_request_writer_protocol.md",ROOT/"references/evidence/english_request_writer_registration.json",
           PREVIOUS/"matched/final.safetensors",PARENT/"memory.npy",PREVIOUS_PUBLIC/"local_window_equivalence.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def long_batch(meta,bank,raster,epoch):
    rng=np.random.default_rng(30149+epoch)
    ids={c:bank.lookup[raster.tile(c)] for c in ALPHABET}
    qs=[]
    for i in range(0,len(meta),4):
        words=[rng.choice(w) for w in FACT]
        forms={k:rng.choice(v) for k,v in FORM.items()}
        marked_first,script_first=rng.integers(2,size=2)
        frame=TRAIN[(epoch+i//4)%len(TRAIN)]
        head=meta[i]["head"]
        ids[head]=bank.lookup[raster.tile(head)]
        for j in range(i,i+4):
            txt=long_question(requested_words(meta,j,words,forms),frame,marked_first,script_first)
            qs.append(np.fromiter((ids[c] for c in txt),np.int64,len(txt)))
    return qs


ALPHABET=tuple(sorted(set("For ."+''.join(''.join(t) for t in TRAIN)+
                             ''.join(''.join(t) for t in (*FACT,*FORM.values())))))


def fit(mode,data,bank,tables):
    d,r,cropped,du,dr=data
    feats,mem,lookup=tables
    path=WORK/mode
    path.mkdir()
    torch.manual_seed(30149)
    net=load_writer(PREVIOUS/"matched",DEVICE)
    encoder=ScopedQuestion(ordered=mode=="ordered").to(DEVICE)
    encoder.convs.load_state_dict(net.question.convs.state_dict())
    encoder.norms.load_state_dict(net.question.norms.state_dict())
    net.question=encoder
    net.train()
    for k,p in net.named_parameters():
        p.requires_grad_(k.startswith(("question.","q_key.","q_pool.")))
    trained=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trained,lr=.00075)
    ink=net.ink(feats).detach()
    keys=net.write_key(ink[mem]).detach()
    n=len(d.meta)
    total=EPOCHS*math.ceil(n/BATCH)
    steps=0
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    logs=[]
    def batch(x):
        return [torch.as_tensor(padded(v),device=DEVICE) for v in x]
    for epoch in range(EPOCHS):
        rng=np.random.default_rng(30149+epoch)
        questions=long_batch(d.meta,bank,r,epoch)
        short=english_batch(d.meta,bank,r,epoch)
        perm=rng.permutation(n)
        sums=torch.zeros(5,dtype=torch.float64,device=DEVICE)
        for j in range(0,n,BATCH):
            rows=perm[j:j+BATCH]
            requests=[batch(([cropped[i] for i in rows],[questions[i] for i in rows],[d.y[i] for i in rows]))]
            ix=rng.choice(n,16,replace=False)
            requests.append(batch(([cropped[i] for i in ix],[short[i] for i in ix],[d.y[i] for i in ix])))
            ix=rng.choice(len(d.rows),16,replace=False)
            real=d.rows[ix]
            requests.append(batch(([d.cards[d.card_of[i]] for i in real],[d.q[i] for i in ix],[d.y[i] for i in real])))
            for ds in (dr,du):
                ix=rng.choice(len(ds.y),12,replace=False)
                requests.append([*ds.inputs(ix,ix,DEVICE),ds.targets(ix,DEVICE)])
            steps+=1
            opt.param_groups[0]["lr"]=min(1.,steps/50)*(.00015+.00060*(1+math.cos(math.pi*steps/total))/2)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                values=torch.stack([-mean_logp(net,ink,keys,lookup,b).mean() for b in requests])
                loss=values[0]+.4*values[1:].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained,1.,error_if_nonfinite=True)
            opt.step()
            sums+=values.detach()*len(rows)
            if steps%100==0 and (monotonic()-start>1200 or torch.cuda.max_memory_allocated()>8*1024**3):
                raise RuntimeError("Training resource ceiling exceeded")
        logs.append(dict(epoch=epoch+1,steps=steps,loss=(sums/n).tolist(),seconds=monotonic()-start))
        print(mode,logs[-1],flush=True)
    assert steps==total
    out=path/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(out))
    write_json(path/"fit.json",dict(sha256=sha(out),question_architecture=mode,steps=steps,epochs=logs,
               peak_bytes=torch.cuda.max_memory_allocated(),seconds=monotonic()-start,
               parameters=sum(p.numel() for p in net.parameters()),trainable=sum(p.numel() for p in trained),
               long_examples=n*EPOCHS,short_english=steps*16,chinese_entries=steps*16,natural_per_corpus=steps*12))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register",action="store_true")
    if p.parse_args().register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,binding())
        return
    if binding()!=load(REG) or os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Changed sources or GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True
        bank=memory_bank()
        data=inputs(bank)
        bank.merge([data[1].tile(c) for c in ALPHABET])
        tables=physical_tables(bank,DEVICE)
        for mode in ("ordered","unordered"):
            fit(mode,data,bank,tables)
        from scripts.evaluate_scoped_request_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()

"""Fit the pixel-stream writer on opposed human answers while preserving the quoted evidence."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from scripts.audit_paired_stance_images import INSTRUCTION
from scripts.evaluate_general_entry_writer import english_request, pictured
from scripts.load_scoped_writer import load_visual_writer
from scripts.prepare_paired_stance_images import DEST, Rasterizer, sha, write_json
from scripts.prepare_pictured_entry_answers import load
from scripts.split_paired_stance import PUBLIC, ROOT, SPLIT, WORK
from scripts.train_english_request_writer import DEVICE, Entries, PARENT, memory_bank, padded, physical_tables
from scripts.train_grounded_glyph_writer import Triples
from scripts.train_paired_passage import unpack_sequences
from scripts.train_scoped_request_writer import WORK as PRIOR

REG = PUBLIC / "registration.json"
SEED, EPOCHS, PAIR_BATCH = 31613, 16, 24
ARMS = ("matched", "reassigned")
CHECKPOINT = PRIOR / "ordered"


def bindings():
    paths = [Path(__file__), SPLIT, CHECKPOINT/"final.safetensors", CHECKPOINT/"fit.json",
             PARENT/"memory.npy", DEST/"preparation.json", ROOT/"references/paired_stance_writer_protocol.md"]
    for name in ("scripts/evaluate_paired_stance_writer.py", "scripts/split_paired_stance.py",
                 "scripts/load_scoped_writer.py", "scripts/train_same_passage_writer.py",
                 "scripts/evaluate_same_passage_writer.py","scripts/audit_paired_stance_images.py",
                 "ilm/visual_lm/grounded_glyph_decoder.py", "ilm/visual_lm/scoped_question.py"):
        paths.append(ROOT/name)
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


def corpus(bank, split):
    folder = DEST/split
    for name,digest in load(DEST/"preparation.json")["splits"][split]["files"].items():
        if sha(folder/name)!=digest:
            raise ValueError(f"Changed stance corpus {split}/{name}")
    ids = bank.merge(np.load(folder/"pixels.npy"))
    instruction = pictured(bank,Rasterizer(),INSTRUCTION)
    p,q,y = [[ids[s] for s in unpack_sequences(folder/(f+".npz"))] for f in ("passages","questions","answers")]
    assert all(np.array_equal(a,b) for a,b in zip(q[::2],q[1::2],strict=True))
    return Triples(p,[np.r_[instruction,s] for s in q],y)


def replay_data(bank):
    entries = Entries(bank,"train")
    raster = Rasterizer()
    english = [pictured(bank,raster,english_request(m)) for m in entries.meta]
    return entries, english


def to_batch(*streams):
    return [torch.as_tensor(padded(s),device=DEVICE) for s in streams]


def fit(arm, train, pair_ids, replay, tables):
    directory = WORK/arm
    directory.mkdir()
    net = load_visual_writer(CHECKPOINT,DEVICE).train()
    for name,param in net.named_parameters():
        param.requires_grad_(not name.startswith(("ink.","write_key.")))
    learned = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(learned,lr=.0004)
    features,memory,lookup = tables
    ink = net.ink(features).detach()
    keys = net.write_key(ink[memory]).detach()
    entries,english = replay
    total = EPOCHS * math.ceil(len(pair_ids)/PAIR_BATCH)
    step, flipped = 0, 0
    log = []
    started = monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        # Each control pair spends exactly eight epochs assigned each orientation.
        flip = np.random.default_rng(SEED+epoch//2).integers(2,size=len(train.p)//2) ^ (epoch%2)
        if arm=="matched":
            flip.fill(0)
        rng = np.random.default_rng(SEED+epoch)
        order = rng.permutation(pair_ids)
        sums = torch.zeros(3,device=DEVICE,dtype=torch.float64)
        for j in range(0,len(order),PAIR_BATCH):
            pairs = order[j:j+PAIR_BATCH]
            rows = (2*pairs[:,None]+np.arange(2)).ravel()
            labels = rows ^ flip[rows//2]
            flipped += int((rows!=labels).sum())
            targets = [np.r_[train.y[label][:3],train.y[i][3:]] for label,i in zip(labels,rows,strict=True)]
            p,q = train.inputs(rows,rows,DEVICE)
            (y,) = to_batch(targets)
            ids = rng.integers(len(entries.meta),size=16)
            ep,eq,ey = to_batch([entries.cards[entries.card_of[i]] for i in ids],
                                [english[i] if k%2 else entries.q[i] for k,i in enumerate(ids)],
                                [entries.y[i] for i in ids])
            step+=1
            lr = .00008+.00032*(1+math.cos(math.pi*step/total))/2
            opt.param_groups[0]["lr"] = lr*min(1,step/30)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                ll,valid = net.likelihood(ink,p,q,y,keys,lookup)
                ell,ev = net.likelihood(ink,ep,eq,ey,keys,lookup)
                prefix = -ll[:,:3].mean()
                body = -(ll[:,3:]*valid[:,3:]).sum(1).div(valid[:,3:].sum(1)).mean()
                recall = -(ell*ev).sum(1).div(ev.sum(1)).mean()
                loss = prefix+body+.25*recall
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learned,1.,error_if_nonfinite=True)
            opt.step()
            sums += torch.stack((prefix,body,recall)).detach()*len(pairs)
            if monotonic()-started>1200 or torch.cuda.max_memory_allocated()>8*1024**3:
                raise RuntimeError("Resource ceiling exceeded")
        log.append(dict(epoch=epoch+1, losses=(sums/len(pair_ids)).tolist(),steps=step))
        print(arm,log[-1],flush=True)
    assert step==total and flipped==(0 if arm=="matched" else len(pair_ids)*EPOCHS)
    path = directory/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(path))
    write_json(directory/"fit.json",dict(question_architecture="ordered",sha256=sha(path),steps=step,
        losses=log,peak_bytes=torch.cuda.max_memory_allocated(),seconds=monotonic()-started,
        signed_responses=len(pair_ids)*2*EPOCHS,reassigned=flipped,rehearsal=16*step,
        trainable=sum(p.numel() for p in learned),parameters=sum(p.numel() for p in net.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    if parser.parse_args().register:
        if REG.exists():
            raise FileExistsError(REG)
        write_json(REG,bindings())
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or bindings()!=load(REG):
        raise ValueError("Device or registration changed")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True
        bank=memory_bank()
        train=corpus(bank,"train")
        replay=replay_data(bank)
        tables=physical_tables(bank,DEVICE)
        for arm in ARMS:
            fit(arm,train,np.asarray(load(SPLIT)["fit"]),replay,tables)
        from scripts.evaluate_paired_stance_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()

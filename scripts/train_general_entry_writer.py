"""Use one pixel-stream writer for pictured Hanziyuan records and natural Chinese passages."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic
import numpy as np
from safetensors.torch import save_file
import torch

from scripts.prepare_entry_wordings import DEST as WORDINGS
from scripts.prepare_pictured_entry_answers import DEFAULT as ENTRIES, Rasterizer, load, sha, write_json
from scripts.train_grounded_glyph_writer import Triples, padded
from scripts.train_paired_passage import unpack_sequences
from scripts.train_same_passage_writer import (DEVICE, PARENT, ROOT, WORK as PREVIOUS, PairCorpus,
                    load_writer, mean_logp, memory_bank, physical_tables, small)

WORK = ROOT/"artifacts/general_entry_writer_v1"
REG = ROOT/"references/evidence/general_entry_writer_registration.json"
AUTHORED = ROOT/"references/evidence/independent_entry_20261119/questions.jsonl"
SEED, BATCH, EPOCHS = 19473, 96, 5


def binding():
    paths=[Path(__file__), ROOT/"scripts/evaluate_general_entry_writer.py", ROOT/"scripts/train_same_passage_writer.py",
           ROOT/"scripts/evaluate_same_passage_writer.py", ROOT/"ilm/visual_lm/grounded_glyph_decoder.py",
           ROOT/"references/general_entry_writer_protocol.md", PREVIOUS/"contrast/final.safetensors",
           ROOT/"references/evidence/same_passage_writer_registration.json", PARENT/"memory.npy",
           ENTRIES/"report.json", WORDINGS/"report.json", AUTHORED]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


class Entries:
    def __init__(self,bank,split):
        d=ENTRIES/split
        for name,expected in load(ENTRIES/"report.json")["splits"][split]["files"].items():
            assert sha(d/name)==expected
        ids=bank.merge(np.load(d/"pixels.npy"))
        self.cards, self.q, self.y = [[ids[s] for s in unpack_sequences(d/(f+".npz"))]
                                      for f in ("cards","questions","answers")]
        self.meta=load(d/"index.json")
        self.rows=np.arange(len(self.q))
        if split=="train":
            for name,expected in load(WORDINGS/"report.json")["files"].items():
                assert sha(WORDINGS/name)==expected
            ids=bank.merge(np.load(WORDINGS/"pixels.npy"))
            self.q += [ids[s] for s in unpack_sequences(WORDINGS/"queries.npz")]
            self.rows = np.r_[self.rows,np.load(WORDINGS/"target.npy")]
        else:
            self.paraphrases=[ids[s] for s in unpack_sequences(d/"question_paraphrases.npz")]
        self.card_of=np.array([m["card"] for m in self.meta])
        self.by_card=np.argsort(self.card_of[self.rows],kind="stable").reshape(len(self.cards),-1)
        assert np.all(self.card_of[self.rows[self.by_card]]==np.arange(len(self.cards))[:,None])


def drcd_replay(bank):
    ds=PairCorpus(bank,"train")
    unique=np.unique(ds.rows[:,1:3])
    p=[ds.passages[ds.groups[i]] for i in unique]
    q=[ds.questions[i] for i in unique]
    y=[s[slice(*ds.spans[i][0])] for s,i in zip(p,unique,strict=True)]
    return Triples(p,q,y)


def fit(arm,entries,du,drcd,tables):
    target=WORK/arm
    target.mkdir()
    torch.manual_seed(SEED)
    net=load_writer(PREVIOUS/"contrast",DEVICE).train()
    opt=torch.optim.AdamW(net.parameters(),lr=.00045)
    features,memory,lookup=tables
    steps=0
    total=EPOCHS*math.ceil(len(entries.rows)/BATCH)
    started=monotonic()
    logs=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        rng=np.random.default_rng(SEED+epoch)
        order=rng.permutation(len(entries.rows))
        # Build the same within-card permutation in both arms to keep replay RNGs matched.
        perm=entries.by_card.copy()
        for row in perm:
            rng.shuffle(row)
        qrow=np.arange(len(entries.rows))
        if arm=="scrambled":
            qrow[entries.by_card.ravel()]=perm.ravel()
        sums=torch.zeros(3,device=DEVICE,dtype=torch.float64)
        for j in range(0,len(order),BATCH):
            ix=order[j:j+BATCH]
            rows=entries.rows[ix]
            streams=([entries.cards[k] for k in entries.card_of[rows]],
                     [entries.q[k] for k in qrow[ix]],[entries.y[k] for k in rows])
            batches=[[torch.as_tensor(padded(s),device=DEVICE) for s in streams]]
            for ds in (du,drcd):
                take=rng.choice(len(ds.y),12,replace=False)
                batches.append([*ds.inputs(take,take,DEVICE),ds.targets(take,DEVICE)])
            steps+=1
            opt.param_groups[0]["lr"]=min(1.,steps/50)*(.00012+.00033*(1+math.cos(math.pi*steps/total))/2)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                ink=net.ink(features)
                keys=net.write_key(ink[memory])
                losses=torch.stack([-mean_logp(net,ink,keys,lookup,b).mean() for b in batches])
                loss=losses[0]+.4*(losses[1]+losses[2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            sums+=losses.detach()*len(ix)
            if monotonic()-started>1800 or torch.cuda.max_memory_allocated()>12*1024**3:
                raise RuntimeError("Training resource ceiling exceeded")
        logs.append(dict(epoch=epoch+1,steps=steps,losses=(sums/len(order)).tolist(),seconds=monotonic()-started))
        print(arm,logs[-1],flush=True)
    assert steps==total
    path=target/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(path))
    write_json(target/"fit.json",dict(sha256=sha(path),epochs=logs,seconds=monotonic()-started,
               peak_bytes=torch.cuda.max_memory_allocated(),steps=steps,entry_records=len(order)*EPOCHS,
               natural_records_per_corpus=steps*12,parameters=sum(p.numel() for p in net.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    args=parser.parse_args()
    if args.register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,binding())
        return
    if load(REG)!=binding() or os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Changed registration or GPU")
    with (PARENT.parent/"grounded_glyph_writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True
        bank=memory_bank()
        entries=Entries(bank,"train")
        du,_=small(bank,"train")
        drcd=drcd_replay(bank)
        tables=physical_tables(bank,DEVICE)
        for arm in ("paired","scrambled"):
            fit(arm,entries,du,drcd,tables)
        from scripts.evaluate_general_entry_writer import evaluate
        evaluate()


if __name__=="__main__":main()

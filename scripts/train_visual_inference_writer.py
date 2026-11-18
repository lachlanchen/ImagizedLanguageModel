"""Learn to assess a pictured claim and restore missing glyphs from that claim."""
import argparse
from collections import defaultdict
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from scripts.audit_visual_inference_pairs import AUDIT, PRIVATE, PUBLIC
from scripts.prepare_visual_inference_pairs import DEST
from scripts.train_paired_stance_writer import (DEVICE, PARENT, ROOT, Rasterizer, WORK as STANCE, load,
        load_visual_writer, memory_bank, physical_tables, pictured, replay_data, sha, to_batch, write_json)
from scripts.train_paired_passage import unpack_sequences
from scripts.train_same_passage_writer import mean_logp

WORK=ROOT/"artifacts/visual_inference_writer_v1"
REG=PUBLIC/"registration.json"
START=STANCE/"matched"
SEED,EPOCHS,BATCH=32719,8,64
ARMS=("paired","mispaired")
DECIDE="这段材料支持或反驳下面的说法，还是无法确定？"
RESTORE="请补出说法中方框遮住的原文。"


class Corpus:
    def __init__(self,bank,split):
        path=DEST/split
        for filename,expected in load(DEST/"preparation.json")["splits"][split]["files"].items():
            if sha(path/filename)!=expected:
                raise ValueError(f"Changed input file {filename}")
        ids=bank.merge(np.load(path/"pixels.npy"))
        self.p,self.h,self.y=[[ids[s] for s in unpack_sequences(path/(k+".npz"))] for k in ("premises","claims","responses")]
        self.meta=load(path/"rows.json")
        r=Rasterizer()
        self.decide=pictured(bank,r,DECIDE)
        self.restore=pictured(bank,r,RESTORE)
        self.box=bank.merge([self.mask_tile()])[0]
        stops=set(pictured(bank,r," ，。！？、；：‘’“”（）《》【】,.!?;:'\"()[]-—…"))
        max_mem=len(np.load(PARENT/"memory.npy",mmap_mode="r"))
        self.choices=[]
        self.q=[np.r_[self.decide,h] for h in self.h]
        for h in self.h:
            eligible=np.fromiter((int(x)<=max_mem and int(x) not in stops for x in h),dtype=bool)
            by_length=[]
            for length in (1,2,3):
                if len(h)<=length:
                    by_length.append(np.empty(0,dtype=int))
                else:
                    ok=np.lib.stride_tricks.sliding_window_view(eligible,length).all(1)
                    by_length.append(np.flatnonzero(ok))
            self.choices.append(by_length)

    @staticmethod
    def mask_tile():
        img=np.zeros((32,32),np.uint8)
        img[3:29,3:29]=1
        img[5:27,5:27]=0
        img[15:17,9:23]=1
        return np.packbits(img).tobytes()

    def holes(self,rows,rng):
        valid=[];inputs=[];targets=[]
        for i in rows:
            options=self.choices[i]
            lengths=[j+1 for j,opt in enumerate(options) if len(opt)]
            if not lengths:
                continue
            n=lengths[rng.integers(len(lengths))]
            candidates=options[n-1]
            a=candidates[rng.integers(len(candidates))]
            h=self.h[i]
            inputs.append(np.r_[self.restore,h[:a],self.box,h[a+n:]])
            targets.append(h[a:a+n])
            valid.append(i)
        return np.asarray(valid,dtype=int),inputs,targets


def mispair(ds,rows,rng):
    """Cyclic derangement within genre with equal exposure and distinct normalized premises."""
    by_genre=defaultdict(list)
    for i in rows:
        by_genre[ds.meta[i]["genre"]].append(i)
    mapping=np.arange(len(ds.meta))
    for group in by_genre.values():
        buckets=defaultdict(list)
        for i in group:
            buckets[ds.meta[i]["premise"]].append(i)
        keys=list(buckets);rng.shuffle(keys)
        order=np.concatenate([rng.permutation(buckets[k]) for k in keys])
        largest=max(map(len,buckets.values()))
        shift=rng.integers(largest,len(order)-largest+1)
        mapping[order]=np.roll(order,shift)
    assert set(mapping[rows])==set(rows)
    assert all(ds.meta[a]["premise"]!=ds.meta[b]["premise"] for a,b in zip(rows,mapping[rows],strict=True))
    return mapping


def binding():
    files=[AUDIT,PRIVATE,START/"final.safetensors",START/"fit.json",PARENT/"memory.npy",
           DEST/"preparation.json",ROOT/"references/visual_inference_writer_protocol.md",Path(__file__)]
    files += [ROOT/p for p in ("scripts/evaluate_visual_inference_writer.py","scripts/train_paired_stance_writer.py",
              "scripts/prepare_visual_inference_pairs.py","scripts/audit_visual_inference_pairs.py",
              "scripts/train_same_passage_writer.py","scripts/load_scoped_writer.py","ilm/visual_lm/grounded_glyph_decoder.py",
              "ilm/visual_lm/scoped_question.py")]
    return {str(f.relative_to(ROOT)):sha(f) for f in files}


def fit(arm,ds,rows,replay,tables):
    dest=WORK/arm
    dest.mkdir()
    net=load_visual_writer(START,DEVICE).train()
    for name,p in net.named_parameters():
        p.requires_grad_(not name.startswith(("ink.","write_key.")))
    trained=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trained,lr=.0004)
    feat,memory,lookup=tables
    ink=net.ink(feat).detach()
    keys=net.write_key(ink[memory]).detach()
    entries,english=replay
    total=EPOCHS*math.ceil(len(rows)/BATCH)
    step=exposures=0
    started=monotonic()
    log=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        rng=np.random.default_rng(SEED+epoch)
        order=rng.permutation(rows)
        mapping=mispair(ds,rows,np.random.default_rng(SEED+100+epoch)) if arm=="mispaired" else np.arange(len(ds.p))
        sums=np.zeros(3)
        for j in range(0,len(rows),BATCH):
            ix=order[j:j+BATCH]
            valid,masked,targets=ds.holes(ix,rng)
            exposures+=len(valid)
            batch=to_batch([ds.p[mapping[i]] for i in ix],[ds.q[i] for i in ix],[ds.y[i] for i in ix])
            missing=to_batch([ds.p[mapping[i]] for i in valid],masked,targets)
            ids=rng.integers(len(entries.meta),size=12)
            recall=to_batch([entries.cards[entries.card_of[i]] for i in ids],
                            [english[i] if k%2 else entries.q[i] for k,i in enumerate(ids)],[entries.y[i] for i in ids])
            step+=1
            opt.param_groups[0]["lr"]=min(1,step/75)*(.00008+.00032*(1+math.cos(math.pi*step/total))/2)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                ell,mask=net.likelihood(ink,*batch,keys,lookup)
                # Equal emphasis on the discriminating opening and on finishing the sentence.
                decision=-(ell[:,0].mean()+(ell*mask).sum(1).div(mask.sum(1)).mean())/2
                recover=-mean_logp(net,ink,keys,lookup,missing).mean()
                dictionary=-mean_logp(net,ink,keys,lookup,recall).mean()
                loss=decision+recover+.15*dictionary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained,1.,error_if_nonfinite=True)
            opt.step()
            sums+=torch.stack((decision,recover,dictionary)).detach().float().cpu().numpy()*len(ix)
            if monotonic()-started>1500 or torch.cuda.max_memory_allocated()>8*1024**3:
                raise RuntimeError("Exceeded fit ceiling")
        log.append(dict(epoch=epoch+1,steps=step,loss=(sums/len(rows)).tolist()))
        print(arm,log[-1],flush=True)
    assert step==total
    path=dest/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(path))
    write_json(dest/"fit.json",dict(sha256=sha(path),question_architecture="ordered",epochs=log,steps=step,
        records=len(rows)*EPOCHS,masked=exposures,seconds=monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(p.numel() for p in net.parameters()),trainable=sum(p.numel() for p in trained)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    if parser.parse_args().register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,binding())
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or binding()!=load(REG):
        raise ValueError("Changed inputs or wrong device")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
        bank=memory_bank()
        ds=Corpus(bank,"train")
        replay=replay_data(bank)
        tables=physical_tables(bank,DEVICE)
        with np.load(PRIVATE) as split:
            rows=split["fit"]
        for arm in ARMS:
            fit(arm,ds,rows,replay,tables)
        from scripts.evaluate_visual_inference_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()

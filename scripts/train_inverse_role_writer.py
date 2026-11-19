"""Fit the recurrent glyph writer to recover unnamed participants from pictured properties."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from scripts.measure_natural_role_questions import DEST as WHO, PUBLIC as WHO_EVIDENCE, SOURCE as DRCD
from scripts.train_general_entry_writer import PairCorpus
from scripts.train_visual_inference_writer import Corpus, PRIVATE, ROOT, DEVICE, PARENT, load, load_visual_writer, sha, write_json
from scripts.train_visual_role_writer import Tiles, WORK as FORWARD, memory_bank, objective, physical_tables, to_batch
from scripts.visual_role_examples import TRAIN_NAMES, predicates, situation

WORK=ROOT/"artifacts/inverse_role_writer_v1"
PUBLIC=ROOT/"references/evidence/inverse_role_writer_20261119"
PARENT_FIT=FORWARD/"matched"
SEED,EPOCHS,BATCH=65171,6,48
QUESTIONS=("谁{}？","{}的是谁？请写名字。","文中哪位{}？","请指出{}的那个人。")
SEPARATORS=("，","；","。")


def tensors(rows):
    return to_batch(*zip(*rows,strict=True))


def source_example(dr,i,rng):
    p=dr.passages[dr.groups[i]]
    a,b=dr.spans[i][rng.integers(len(dr.spans[i]))]
    return p,dr.questions[i],p[a:b]


class InverseScenes:
    def __init__(self,bank,items):
        self.tile=t=Tiles(bank)
        self.words=[(p.positive+p.complement,p.negative+p.complement) for p in items]
        self.pred={s:t(s) for pair in self.words for s in pair}
        self.q={s:[t(form.format(s)) for form in QUESTIONS] for s in self.pred}
        self.sep=[t(s) for s in SEPARATORS]
        self.end=t("。")
        self.names=[t(n) for n in TRAIN_NAMES]
        raw=list(dict.fromkeys(int(x) for a in self.names for x in a))
        shifted=[]
        for i in raw:
            bits=np.unpackbits(np.frombuffer(bank.tiles[i],np.uint8)).reshape(32,32)
            for dx,dy in ((1,0),(-1,1),(0,-1)):
                pad=np.pad(bits,2)
                moved=pad[2-dy:34-dy,2-dx:34-dx]
                shifted.append(np.packbits(moved).tobytes())
        self.pool=np.r_[raw,bank.merge(shifted)]
        for pred in items:
            for layout in (0,1):
                for task in ("decision","write"):
                    for fields in situation(pred,TRAIN_NAMES[:2],True,layout,task):
                        for text in fields:t(text)

    def aliases(self,n,exclude,rng):
        if rng.random()<.45:
            pool=[x for x in self.names if not all(v in exclude for v in x)]
            return [pool[i] for i in rng.choice(len(pool),n,replace=False)]
        pool=np.array([x for x in self.pool if x not in exclude])
        lengths=rng.integers(1,4,n)
        cells=rng.choice(pool,int(sum(lengths)),replace=False)
        return np.split(cells,np.cumsum(lengths)[:-1])

    def sample(self,rng,ctrl_rng,randomized):
        cases=[]
        for count,opposed in ((2,True),(2,True),(2,False),(3,False)):
            if opposed:
                words=list(self.words[rng.integers(len(self.words))])
            else:
                words=[]
                while len(words)<count:
                    word=self.words[rng.integers(len(self.words))][0]
                    if all(w not in word and word not in w for w in words):
                        words.append(word)
            rng.shuffle(words)
            qs=[self.q[w][rng.integers(len(QUESTIONS))] for w in words]
            aliases=self.aliases(count,set(np.concatenate(qs)),rng)
            sep=self.sep[rng.integers(len(self.sep))]
            shift=ctrl_rng.integers(count) if randomized else 0
            for rotation in range(count):
                names=aliases[rotation:]+aliases[:rotation]
                chunks=[]
                for i,(name,w) in enumerate(zip(names,words,strict=True)):
                    chunks.extend((name,self.pred[w],self.end if i==count-1 else sep))
                p=np.concatenate(chunks)
                cases.extend((p,q,names[(j+shift)%count]) for j,q in enumerate(qs))
        return cases


def binding():
    files=[Path(__file__),ROOT/"scripts/evaluate_inverse_role_writer.py",ROOT/"scripts/visual_role_examples.py",
           ROOT/"scripts/train_visual_role_writer.py",ROOT/"scripts/check_role_writer_historical.py",
           ROOT/"scripts/evaluate_visual_role_writer.py",ROOT/"scripts/check_stance_role_switch.py",
           ROOT/"references/inverse_role_writer_protocol.md",WHO/"cohort.npz",WHO_EVIDENCE/"baseline.json",
           DRCD/"preparation.json",FORWARD/"result.json",PARENT_FIT/"final.safetensors",PARENT/"memory.npy",PRIVATE]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def fit(dest,arm,ds,dr,items,scenes,tables):
    dest.mkdir()
    with np.load(WHO/"cohort.npz") as split:
        who=split["train"];pairs=who[split["train_pairs"]]
    other=np.setdiff1d(np.arange(len(dr.questions)),who)
    with np.load(PRIVATE) as split:nli=split["fit"]
    per_epoch=math.ceil(len(other)/BATCH)
    torch.manual_seed(SEED)
    net=load_visual_writer(PARENT_FIT,DEVICE).train()
    for name,p in net.named_parameters():
        p.requires_grad_(not name.startswith(("ink.","write_key.")))
    trainable=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=.0003)
    feat,memory,lookup=tables
    ink=net.ink(feat).detach();keys=net.write_key(ink[memory]).detach()
    started=monotonic()
    torch.cuda.reset_peak_memory_stats()
    trace=[];steps=0;exposures=np.zeros(6,np.int64)
    for e in range(EPOCHS):
        rng=np.random.default_rng(SEED+e);control=np.random.default_rng(SEED+100+e)
        commons=np.array_split(rng.permutation(other),per_epoch)
        singles=np.array_split(np.concatenate([rng.permutation(who) for _ in range(2)]),per_epoch)
        paired=np.array_split(rng.permutation(len(pairs)),per_epoch)
        sums=np.zeros(3)
        for ix,ws,pr in zip(commons,singles,paired,strict=True):
            src=[source_example(dr,i,rng) for i in np.r_[ix,ws]]
            for a,b in pairs[pr]:
                pa,qa,ya=source_example(dr,a,rng);pb,qb,yb=source_example(dr,b,rng)
                assert dr.groups[a]==dr.groups[b]
                if control.integers(2) and arm=="randomized":ya,yb=yb,ya
                src.extend(((pa,qa,ya),(pb,qb,yb)))
            inverse=scenes.sample(rng,control,arm=="randomized")
            replay=[]
            for task in ("decision","write"):
                names=tuple(rng.choice(TRAIN_NAMES,2,replace=False))
                rows=situation(items[rng.integers(len(items))],names,bool(rng.integers(2)),int(rng.integers(2)),task)
                replay.extend(tuple(scenes.tile(x) for x in row) for row in rows)
            ni=rng.choice(nli,24,replace=False)
            replay.extend((ds.p[i],ds.q[i],ds.y[i]) for i in ni[:16])
            valid,holes,ys=ds.holes(ni[16:],rng)
            replay.extend((ds.p[i],q,y) for i,q,y in zip(valid,holes,ys,strict=True))
            exposures+=np.array([len(ix),len(ws),2*len(pr),len(inverse),len(replay),len(valid)])
            steps+=1
            opt.param_groups[0]["lr"]=min(1,steps/70)*(.00005+.00025*(1+math.cos(math.pi*steps/(EPOCHS*per_epoch)))/2)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                costs=torch.stack([objective(net,ink,keys,lookup,tensors(rows)) for rows in (src,inverse,replay)])
                loss=costs[0]+.8*costs[1]+.8*costs[2]
            loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);opt.step()
            sums+=costs.detach().cpu().numpy()
            if monotonic()-started>1800 or torch.cuda.max_memory_allocated()>8*1024**3:
                raise RuntimeError("Training exceeded its resource ceiling")
        trace.append(dict(epoch=e+1,steps=steps,loss=(sums/per_epoch).tolist()))
        print(arm,trace[-1],flush=True)
    path=dest/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(path))
    write_json(dest/"fit.json",dict(sha256=sha(path),question_architecture="ordered",steps=steps,epochs=trace,
        exposures=dict(zip(("ordinary","who","competing","inverse","replay","masked_in_replay"),map(int,exposures),strict=True)),
        peak_bytes=torch.cuda.max_memory_allocated(),seconds=monotonic()-started,parameters=sum(p.numel() for p in net.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--register",action="store_true")
    parser.add_argument("--output",type=Path,default=WORK);args=parser.parse_args()
    reg=PUBLIC/"registration.json"
    if args.register:
        PUBLIC.mkdir(exist_ok=True,parents=True)
        if reg.exists():raise FileExistsError(reg)
        write_json(reg,binding());return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or binding()!=load(reg):raise ValueError("Changed study inputs or device")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);args.output.mkdir()
        torch.set_num_threads(2);bank=memory_bank();ds=Corpus(bank,"train");dr=PairCorpus(bank,"train")
        items=predicates()["fit"];scenes=InverseScenes(bank,items);tables=physical_tables(bank,DEVICE)
        for arm in ("matched","randomized"):
            fit(args.output/arm,arm,ds,dr,items,scenes,tables)
        from scripts.evaluate_inverse_role_writer import evaluate
        evaluate(args.output)


if __name__=="__main__":main()

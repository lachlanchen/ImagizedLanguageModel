"""Train matched and randomized role assignment on real pictured Chinese predicates."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from scripts.prepare_visual_inference_pairs import DEST, RAW
from scripts.train_visual_inference_writer import (Corpus, DEVICE, PARENT, PRIVATE, ROOT, WORK as PREVIOUS,
    Rasterizer, load, load_visual_writer, memory_bank, physical_tables, pictured, sha, to_batch, write_json)
from scripts.visual_role_examples import TRAIN_NAMES, predicates, situation

WORK = ROOT/"artifacts/visual_role_writer_v1"
PUBLIC = ROOT/"references/evidence/visual_role_writer_20261119"
START = PREVIOUS/"paired"
SEED, EPOCHS, BATCH, ROLES = 41963, 5, 64, 6


def files():
    paths = [Path(__file__), Path(__file__).with_name("visual_role_examples.py"), ROOT/"references/visual_role_writer_protocol.md",
             ROOT/"scripts/evaluate_visual_role_writer.py", ROOT/"scripts/train_visual_inference_writer.py",
             ROOT/"scripts/probe_visual_inference_switches.py", ROOT/"scripts/check_stance_role_switch.py",
             PRIVATE, DEST/"preparation.json", RAW/"train.50k.json", START/"final.safetensors", PARENT/"memory.npy"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


class Tiles:
    def __init__(self, bank):
        self.bank, self.raster, self.address = bank, Rasterizer(), {}

    def __call__(self, text):
        for c in set(text) - self.address.keys():
            self.address[c] = int(pictured(self.bank,self.raster,c)[0])
        return np.fromiter((self.address[c] for c in text), np.int64)


def role_batch(t, items, rng, control, control_rng):
    rows = []
    for i in range(ROLES):
        pred = items[rng.integers(len(items))]
        names = tuple(rng.choice(TRAIN_NAMES,2,replace=False))
        cases = situation(pred,names,bool(rng.integers(2)),int(rng.integers(2)),
                          "decision" if i%2==0 else "write",bool(rng.integers(2)))
        flip = bool(control_rng.integers(2)) and control
        for j,(_,question,answer) in enumerate(cases):
            source = cases[j^2 if flip else j][0]
            rows.append((t(source),t(question),t(answer)))
    return rows


def objective(net,ink,keys,lookup,batch):
    ll,mask=net.likelihood(ink,*batch,keys,lookup)
    return -(.5*ll[:,0]+.5*(ll*mask).sum(1)/mask.sum(1)).mean()


def fit(dest, arm, ds, rows, items, tiles, tables):
    dest.mkdir()
    torch.manual_seed(SEED)
    net = load_visual_writer(START,DEVICE).train()
    for name,p in net.named_parameters():
        p.requires_grad_(not name.startswith(("ink.","write_key.")))
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable,lr=.00025)
    feat,memory,lookup = tables
    ink = net.ink(feat).detach()
    keys = net.write_key(ink[memory]).detach()
    total = EPOCHS*math.ceil(len(rows)/BATCH)
    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    trace = []
    steps = 0
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(SEED+epoch)
        control_rng = np.random.default_rng(SEED+100+epoch)
        order = rng.permutation(rows)
        sums = np.zeros(3)
        for offset in range(0,len(order),BATCH):
            ix = order[offset:offset+BATCH]
            roles = role_batch(tiles,items,rng,arm=="randomized",control_rng)
            valid,holes,missing = ds.holes(ix[:16],rng)
            natural = to_batch([ds.p[i] for i in ix],[ds.q[i] for i in ix],[ds.y[i] for i in ix])
            role = to_batch(*zip(*roles,strict=True))
            masked = to_batch([ds.p[i] for i in valid],holes,missing)
            steps += 1
            opt.param_groups[0]["lr"] = min(1,steps/65)*(.00004+.00021*(1+math.cos(math.pi*steps/total))/2)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                losses = torch.stack([objective(net,ink,keys,lookup,batch) for batch in (natural,role,masked)])
                loss = losses[0]+losses[1]+.55*losses[2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            opt.step()
            sums += losses.detach().float().cpu().numpy()
            if monotonic()-started>1200 or torch.cuda.max_memory_allocated()>8*1024**3:
                raise RuntimeError("Registered training ceiling exceeded")
        trace.append(dict(epoch=epoch+1,steps=steps,loss=(sums/math.ceil(len(rows)/BATCH)).tolist()))
        print(arm,trace[-1],flush=True)
    assert steps==total
    path=dest/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(path))
    write_json(dest/"fit.json",dict(question_architecture="ordered",sha256=sha(path),epochs=trace,steps=steps,
            natural_rows=len(rows)*EPOCHS,role_rows=steps*ROLES*4,role_situations=steps*ROLES,
            seconds=monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in net.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    parser.add_argument("--output",type=Path,default=WORK)
    args=parser.parse_args()
    reg=PUBLIC/"registration.json"
    if args.register:
        PUBLIC.mkdir(parents=True,exist_ok=True)
        if reg.exists():raise FileExistsError(reg)
        write_json(reg,files())
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or files()!=load(reg):
        raise ValueError("Training inputs or device differ from the registration")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.mkdir()
        torch.set_num_threads(2)
        bank=memory_bank()
        ds=Corpus(bank,"train")
        items=predicates()
        tiles=Tiles(bank)
        for p in items["fit"]:
            for layout in (0,1):
                for task in ("decision","write"):
                    for parts in situation(p,TRAIN_NAMES[:2],True,layout,task):
                        for text in parts: tiles(text)
        tiles("".join(TRAIN_NAMES))
        tables=physical_tables(bank,DEVICE)
        with np.load(PRIVATE) as split: rows=split["fit"]
        for arm in ("matched","randomized"):
            fit(args.output/arm,arm,ds,rows,items["fit"],tiles,tables)
        from scripts.evaluate_visual_role_writer import evaluate
        evaluate(args.output)


if __name__=="__main__":
    main()

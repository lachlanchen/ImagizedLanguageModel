"""Matched Tang continuation fits with and without reciprocal source supervision."""
from collections import defaultdict
import hashlib
import json
import math
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.verse_phrase import phrase_score
from scripts.prepare_verse_canvas import corpus, MODEL
from scripts.prepare_verse_contrast import load_song, PUB
from scripts.prepare_dialogue_ink import dump, sha
from scripts.write_verse_canvas import load
from scripts.train_verse_canvas import banks
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

STEPS = 4000
BATCH = 64
SEED = 187450


def source_pairs(rows, rng):
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[int(row[0]),int(row[1])].append(i)
    pairs = {5:[],7:[]}
    for (_,n),indices in groups.items():
        if len(indices)<2:
            continue
        perm = rng.permutation(indices)
        other = np.roll(perm,1)
        valid = (rows[perm,2:2+n] != rows[other,2:2+n]).any(1)
        valid &= (rows[perm,9:9+n] != rows[other,9:9+n]).any(1)
        pairs[n].extend(zip(perm[valid],other[valid]))
    return {n:np.asarray(p) for n,p in pairs.items()}


def fit(mode, contrast, meta, data, song_hash):
    name = mode+("-contrast" if contrast else "-paired")
    receipt = PUB / (name+"-fit.json")
    dest = MODEL / (name+".safetensors")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        parent_name = mode+"-bound"
        manifest = json.loads((MODEL/"bound-writer.json").read_text())
        expected = dict(name=name, weights=sha(dest), song_reservation=song_hash,
                        tang=meta["corpus"], exposures=STEPS*BATCH,
                        parent=sha(MODEL/(parent_name+".safetensors")))
        if expected["parent"] != manifest["weights"][parent_name] or any(
                old.get(key) != value for key,value in expected.items()):
            raise ValueError("Checkpoint, parent, corpus or fit reservation changed")
        return old
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows = data["train"]
    pairs = source_pairs(rows,rng)
    frac = len(pairs[5]) / (len(pairs[5])+len(pairs[7]))
    bank = banks(data,mode,"cuda")
    table = torch.as_tensor(rows,device="cuda")
    net,parent = load(MODEL,mode+"-bound","cuda")
    net.train()
    opt = torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.03,fused=True)
    digest = hashlib.sha256()
    trace = []
    exposed = np.zeros(len(rows),bool)
    torch.cuda.reset_peak_memory_stats()
    energy = snapshot()
    clock = monotonic()
    for step in range(1,STEPS+1):
        n = 5 if rng.random()<frac else 7
        pair = pairs[n][rng.integers(len(pairs[n]),size=BATCH//2)]
        indices = pair.T.flatten()
        exposed[indices] = True
        face = np.tile(rng.integers(2,size=BATCH//2),2)
        start = np.tile(rng.integers(n-1,size=BATCH//2),2)
        size = rng.integers(1,n+1,size=BATCH)
        size[rng.random(BATCH)<.25] = n
        mask = rng.random((BATCH,n)).argsort(1)<size[:,None]
        for arr in (indices,face,start,mask):
            digest.update(arr.tobytes())
        batch = table[indices].long()
        src,truth = batch[:,2:2+n],batch[:,9:9+n]
        mask = torch.as_tensor(mask,device="cuda")
        start = torch.as_tensor(start,device="cuda")
        ink = bank[torch.as_tensor(face,device="cuda")[:,None],src]
        anchor = bank[0,truth.masked_fill(mask,0)]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits = net(ink,anchor,~mask)
            loss = F.cross_entropy(logits.flatten(0,1),(truth-1).masked_fill(~mask,-100).flatten())
            obs = torch.cat([net.source(ink),net.target(anchor)],1)
            labels = torch.cat([src-1,(truth-1).masked_fill(mask,-100)],1)
            visible_logits = net.ink(F.layer_norm(obs,(obs.shape[-1],)))
            loss = loss + .5*F.cross_entropy(visible_logits.flatten(0,1),labels.flatten())
            pos = phrase_score(net,ink,truth,bank[0],start,meta["train_shapes"])
            loss = loss - .25*pos.mean()
            if contrast:
                neg = phrase_score(net,ink.roll(BATCH//2,0),truth,bank[0],start,meta["train_shapes"])
                margins = (pos-neg).reshape(2,-1).sum(0)
                loss = loss + .5*F.softplus(-margins/2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.param_groups[0]["lr"] = .0003*min(1.,step/120)*(.1+.9*(1+math.cos(math.pi*step/STEPS))/2)
        opt.step()
        if step == 1 or step%250==0:
            scalar = loss.item()
            elapsed = monotonic()-clock
            if not math.isfinite(scalar) or elapsed>1200 or torch.cuda.max_memory_allocated()>8*2**30:
                raise RuntimeError("Numerical or resource limit exceeded")
            trace.append([step,scalar,elapsed])
            print(name,step,round(scalar,3),round(elapsed,2),flush=True)
    torch.cuda.synchronize()
    elapsed = monotonic()-clock
    cost = consumed(energy,snapshot())
    temp = dest.with_suffix(".tmp")
    save_file({k:p.cpu().contiguous() for k,p in net.state_dict().items()},str(temp))
    temp.replace(dest)
    report = dict(name=name,weights=sha(dest),parent=parent["weights"][mode+"-bound"],
                  tang=meta["corpus"],song_reservation=song_hash,schedule=digest.hexdigest(),
                  exposures=STEPS*BATCH,distinct_pages=int(exposed.sum()),seconds=elapsed,
                  peak_allocated=torch.cuda.max_memory_allocated(),energy=cost,trace=trace)
    dump(receipt,report)
    return report


def main():
    meta,data = corpus()
    song,_ = load_song()
    with gpu_owner():
        receipts = [fit(mode,c,meta,data,song["sha256"]) for mode in ("ink","byte") for c in (False,True)]
    if len({r["schedule"] for r in receipts}) != 1:
        raise ValueError("Mismatched training schedules")
    dump(MODEL/"phrase-writer.json",dict(vocab=meta["train_shapes"],palette=meta["palette"],
                                       weights={r["name"]:r["weights"] for r in receipts}))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()

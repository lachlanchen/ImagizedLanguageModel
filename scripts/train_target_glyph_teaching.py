"""Matched autonomous writers under ordinary likelihood and optical teacher lower bounds."""
from pathlib import Path
from time import monotonic
import hashlib
import math

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.target_glyph_teaching import ARMS, MODELS, FIELD_MODELS, deficit_loss
from scripts.reserve_target_glyph_book_panel import PUB, ROOT, RUN, dump, library, read, sha
from scripts.train_field_to_glyph import (atomic_save, cache_field, configure_cuda_cli, data_for, dev_losses,
                                         gpu_owner, material, restore, schedule, scores)


def fit(arm, rows, dev, optical, binding):
    out = MODELS/arm
    out.mkdir(parents=True, exist_ok=True)
    receipt, path = out/"fit.json", out/"weights.safetensors"
    if receipt.exists():
        prior = read(receipt)
        if sha(path) != prior["weights"] or prior["binding"] != binding:
            raise ValueError("Existing target-side fit differs")
        return
    torch.manual_seed(271803)
    net, parents = restore("prediction","cuda")
    parents.append(read(FIELD_MODELS/"prediction/fit.json")["weights"])
    features, inv, maps = optical
    torch.cuda.reset_peak_memory_stats()
    start = monotonic()
    with torch.no_grad():
        bank = net.writer.optical(features)
        keys = [net.writer.prototypes(bank, v) for v in inv]
    data = [data_for(rows, t, "cuda") for t in (0,1)]
    fields = [cache_field(net.readers[t],features,x,lens[:,0]) for t,(x,_,lens) in enumerate(data)]
    targets = []
    for lang in ("en","zh"):
        path_in = RUN/f"{lang}-optical.npz"
        if sha(path_in) != read(PUB/"targets.json")[lang]["sha256"]:
            raise ValueError("Cached target distribution changed")
        with np.load(path_in) as a:
            targets.append(tuple(torch.as_tensor(a[k],device="cuda") for k in ("addr","choices","weights")))
    initial = dev_losses(net,features,bank,keys,maps,dev)
    opt = torch.optim.AdamW(net.trainable().values(), lr=.0002, fused=True)
    history = []
    for epoch in range(4):
        batches = [(t,ix) for t in (0,1) for ix in schedule(data[t][2],epoch)]
        rng = np.random.default_rng(19931+epoch)
        digest = hashlib.sha256()
        stats = np.zeros((2,4))
        for step,b in enumerate(rng.permutation(len(batches))):
            t,ix = batches[b]
            digest.update(bytes([t])+ix.astype("<i4").tobytes())
            x,y,lens = data[t]
            w,v = lens[ix].max(0)
            ii = torch.as_tensor(ix,device="cuda")
            src, target = x[ii,:w], y[ii,:v]
            grid, starts = fields[t]
            addr = (starts[ii,None]+torch.arange(w,device="cuda")).masked_fill(src.eq(0),len(grid)-1)
            field = grid[addr]
            progress = (epoch*len(batches)+step)/(4*len(batches)-1)
            opt.param_groups[0]["lr"] = .000025+.0000875*(1+math.cos(math.pi*progress))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits = scores(net,bank,src,target,field,t,keys[t])
            log_probs = logits.float().log_softmax(-1)
            labels = maps[t][target]
            human = F.nll_loss(log_probs.flatten(0,1),labels.flatten(),ignore_index=-100)
            positions, codes, floors = targets[t]
            group = positions[ii,:v]
            aligned = group.ge(0)
            ii_align = group[aligned].long()
            # Compute both objectives in both arms to record comparable diagnostics.
            guidance = deficit_loss(log_probs[aligned], codes[ii_align].long(), floors[ii_align]).mean()
            loss = human+guidance if arm=="guided" else human
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.trainable().values(),1.,error_if_nonfinite=True)
            opt.step()
            count = int(lens[ix,1].sum())
            n = len(ii_align)
            stats[t] += (float(human.detach())*count,count,float(guidance.detach())*n,n)
        if any(int(stats[t,1])!=data[t][2][:,1].sum() or int(stats[t,3])!=len(targets[t][1]) for t in (0,1)):
            raise ValueError("Missing target or boundary exposure")
        if torch.cuda.max_memory_allocated()>12*2**30 or monotonic()-start>3600:
            raise RuntimeError("Resource ceiling")
        history.append(dict(epoch=epoch+1,updates=len(batches),order=digest.hexdigest(),
                            positions=stats[:,1].astype(int).tolist(),boundaries=stats[:,3].astype(int).tolist(),
                            nll=(stats[:,0]/stats[:,1]).tolist(),deficit=(stats[:,2]/stats[:,3]).tolist()))
        print(arm, history[-1], flush=True)
    terminal = dev_losses(net,features,bank,keys,maps,dev)
    atomic_save(lambda f:save_file({k:p.detach().cpu() for k,p in net.trainable().items()},f),path)
    dump(receipt,dict(arm=arm,binding=binding,parents=parents,weights=sha(path),initial=initial,terminal=terminal,
                      history=history,seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
                      trainable=sum(p.numel() for p in net.trainable().values()),total=sum(p.numel() for p in net.parameters())))


def main():
    configure_cuda_cli()
    if sha(RUN/"panel.json") != read(PUB/"panel.json")["sha256"]:
        raise ValueError("Fresh panel reservation changed")
    paths = [Path(__file__), ROOT/"ilm/visual_lm/target_glyph_teaching.py", PUB/"targets.json",
             PUB/"panel.json", ROOT/"references/target_glyph_teaching_protocol.md", ROOT/"scripts/train_field_to_glyph.py"]
    binding = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    with gpu_owner():
        torch.set_num_threads(2)
        rows, dev = library("train"), library("dev")
        optical = material("cuda")
        for arm in ARMS:
            fit(arm,rows,dev,optical,binding)
            torch.cuda.empty_cache()
    a,b = [read(MODELS/m/"fit.json") for m in ARMS]
    if a["parents"]!=b["parents"] or a["initial"]!=b["initial"] or any(
            x[k]!=y[k] for x,y in zip(a["history"],b["history"],strict=True)
            for k in ("order","positions","boundaries","updates")):
        raise ValueError("Unmatched fits")
    dump(PUB/"training.json", {arm:sha(MODELS/arm/"fit.json") for arm in ARMS})


if __name__=="__main__":
    main()

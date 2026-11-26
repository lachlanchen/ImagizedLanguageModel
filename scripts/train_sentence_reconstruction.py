"""Matched independent visual writers with and without reverse-translation preferences."""
from hashlib import sha256
import math
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.sentence_reconstruction import ARMS, MODELS, START, mean_log_score, restore
from scripts.collect_sentence_drafts import DRAFTS, PROTOCOL, PUB, RECEIPT, ROOT, dump, read, sha
from scripts.reserve_target_glyph_book_panel import library
from scripts.train_field_to_glyph import (atomic_save, cache_field, configure_cuda_cli, data_for, dev_losses,
                                         gpu_owner, material, pack, schedule, scores)


def draft_bank(records):
    local = {r["i"]:j for j,r in enumerate(records)}
    a,b = [pack([r[k]+[1] for r in records],"cuda") for k in ("a","b")]
    margins = [r["scores"][0]-r["scores"][1] if all(r["stops"]) and r["a"]!=r["b"] else 0.
               for r in records]
    return local, a, b, torch.tensor(margins,device="cuda")


def train(arm, rows, dev, optical, drafts, binding):
    folder = MODELS/arm
    folder.mkdir(exist_ok=True,parents=True)
    path, receipt = folder/"weights.safetensors", folder/"fit.json"
    if receipt.exists():
        assert read(receipt)["binding"]==binding and read(receipt)["weights"]==sha(path)
        return
    torch.manual_seed(61832)
    torch.cuda.reset_peak_memory_stats()
    start = monotonic()
    net = restore("parent","cuda")
    features,inv,maps = optical
    with torch.no_grad():
        bank = net.writer.optical(features)
        keys = [net.writer.prototypes(bank,v) for v in inv]
    data = [data_for(rows,t,"cuda") for t in (0,1)]
    fields = [cache_field(net.readers[t],features,x,lens[:,0]) for t,(x,_,lens) in enumerate(data)]
    initial = dev_losses(net,features,bank,keys,maps,dev)
    opt = torch.optim.AdamW(net.trainable().values(),lr=.00015,fused=True)
    history = []
    for epoch in range(4):
        batches = [(t,ix) for t in (0,1) for ix in schedule(data[t][2],epoch)]
        digest, stats = sha256(), np.zeros((2,6))
        for step,j in enumerate(np.random.default_rng(61308+epoch).permutation(len(batches))):
            t,ix = batches[j]
            digest.update(bytes([t])+ix.astype("<i4").tobytes())
            x,y,lens = data[t]
            width,height = lens[ix].max(0)
            ii = torch.as_tensor(ix,device="cuda")
            src,ref = x[ii,:width],y[ii,:height]
            grid,starts = fields[t]
            addresses = (starts[ii,None]+torch.arange(width,device="cuda")).masked_fill(src.eq(0),len(grid)-1)
            field = grid[addresses]
            opt.param_groups[0]["lr"] = .00002+.000065*(1+math.cos(math.pi*(epoch*len(batches)+step)/(4*len(batches)-1)))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits = scores(net,bank,src,ref,field,t,keys[t])
            human = F.cross_entropy(logits.float().flatten(0,1),maps[t][ref].flatten(),ignore_index=-100)
            local,a,b,margins = drafts[t]
            pos = [p for p,i in enumerate(ix) if i in local]
            loss = human
            if pos:
                di = [local[ix[p]] for p in pos]
                ds = []
                for choices in (a,b):
                    draft = choices[di]
                    draft = draft[:,:int(draft.ne(0).sum(1).max())]
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        z = scores(net,bank,src[pos],draft,field[pos],t,keys[t])
                    ds.append(mean_log_score(z,maps[t][draft]))
                margin = margins[di]
                oriented = (ds[0]-ds[1])*margin.sign()
                pref = (F.softplus(-4*oriented)*margin.abs().clamp_max(1.)).mean()
                loss = human + (.5 if arm=="preference" else 0.)*pref
                stats[t,2:] += (len(di),float(pref.detach())*len(di),int((oriented>0).sum()),int(margin.ne(0).sum()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.trainable().values(),1.,error_if_nonfinite=True)
            opt.step()
            n = int(lens[ix,1].sum())
            stats[t,:2] += (n,float(human.detach())*n)
        for t in (0,1):
            assert int(stats[t,0])==data[t][2][:,1].sum() and int(stats[t,2])==len(drafts[t][0])
        if monotonic()-start>3600 or torch.cuda.max_memory_allocated()>12*2**30:
            raise RuntimeError("Training exceeded registered resource ceiling")
        history.append(dict(epoch=epoch+1,updates=len(batches),order=digest.hexdigest(),
                            positions=stats[:,0].astype(int).tolist(),pairs=stats[:,2].astype(int).tolist(),
                            nll=(stats[:,1]/stats[:,0]).tolist(),preference_loss=(stats[:,3]/stats[:,2]).tolist(),
                            alignment=(stats[:,4]/stats[:,5]).tolist()))
        print(arm,history[-1],flush=True)
    terminal = dev_losses(net,features,bank,keys,maps,dev)
    atomic_save(lambda f:save_file({k:p.detach().cpu() for k,p in net.trainable().items()},f),path)
    dump(receipt,dict(arm=arm,binding=binding,initial=initial,terminal=terminal,history=history,
                      weights=sha(path),parent=read(START/"guided/fit.json")["weights"],
                      seconds=monotonic()-start,peak=torch.cuda.max_memory_allocated(),
                      total=sum(p.numel() for p in net.parameters()),trainable=sum(p.numel() for p in net.trainable().values())))


def main():
    configure_cuda_cli()
    assert sha(DRAFTS)==read(RECEIPT)["sha256"] and read(RECEIPT)["protocol"]==sha(PROTOCOL)
    paths = [Path(__file__), ROOT/"ilm/visual_lm/sentence_reconstruction.py", PROTOCOL, RECEIPT, PUB/"panel.json",
             ROOT/"scripts/collect_sentence_drafts.py", ROOT/"scripts/train_field_to_glyph.py", START/"guided/fit.json"]
    binding = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    with gpu_owner():
        torch.set_num_threads(2)
        rows, dev = library("train"),library("dev")
        banks = [draft_bank(r) for r in read(DRAFTS)]
        optical = material("cuda")
        for arm in ARMS:
            train(arm,rows,dev,optical,banks,binding)
            torch.cuda.empty_cache()
    a,b = [read(MODELS/arm/"fit.json") for arm in ARMS]
    assert a["initial"]==b["initial"] and a["weights"]!=b["weights"]
    for x,y in zip(a["history"],b["history"],strict=True):
        assert all(x[k]==y[k] for k in ("updates","order","positions","pairs"))
    dump(PUB/"training.json",dict(rows=len(rows),dev=len(dev),fits={arm:sha(MODELS/arm/"fit.json") for arm in ARMS}))


if __name__=="__main__":
    main()

"""Fine-tune the physical writer on grounded relation statements and matched controls."""
import argparse
from dataclasses import replace
import fcntl
import json
import os
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.grounded_proposition_corpus import WORK, PUBLIC, materialize
from scripts.pointing_glyph_corpus import digest, load
from scripts.train_pointing_causal_writer import ROOT, WORK as PRIOR, PUBLIC as PRIOR_PUBLIC, inputs, objective, sha, record

SEED=57013


def boards():
    path=WORK/"boards.json"
    expected=json.loads((PUBLIC/"corpus.json").read_text())["boards_sha256"]
    if sha(path)!=expected:raise RuntimeError("Prepared passages changed")
    return json.loads(path.read_text())


def save(net,arm):
    folder=WORK/(arm+".partial")
    folder.mkdir()
    for f in ("config.json","interface.safetensors","metric.safetensors"):
        os.link(PRIOR/"pointing"/f,folder/f)
    for model,name in ((net.writer.body,"body"),(net.pointer,"pointer")):
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(folder/(name+".safetensors")))
    dest=WORK/arm
    folder.rename(dest)
    return {p.name:sha(p) for p in dest.iterdir()}


def fit(rows,arm,start,epochs=2):
    if (WORK/arm).exists():raise FileExistsError(WORK/arm)
    fit_file=PUBLIC/(start+"_fit.json") if start!="pointing" else PRIOR_PUBLIC/"pointing_fit.json"
    checkpoint=WORK/start if start!="pointing" else PRIOR/"pointing"
    assert all(sha(checkpoint/k)==v for k,v in json.loads(fit_file.read_text())["weights"].items())
    torch.manual_seed(SEED)
    model=PointingCausalWriter.restore(checkpoint,"cuda")
    body=model.writer
    body.requires_grad_(False)
    body.body.h[-4:].requires_grad_(True)
    body.body.ln_f.requires_grad_(True)
    optimizer=torch.optim.AdamW([{"params":[p for p in body.parameters() if p.requires_grad],"lr":5e-5},
                                 {"params":model.pointer.parameters(),"lr":4e-4}],fused=True)
    read,bank,pairs,answers=inputs(body,rows)
    rng=np.random.default_rng(SEED)
    t=monotonic();losses=[];exposures=steps=0
    for _ in range(epochs):
        nll=cells=0
        for ids in np.array_split(rng.permutation(len(rows)),int(np.ceil(len(rows)/16))):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,unc,n=objective(model,read,bank,[pairs[i] for i in ids],[answers[i] for i in ids],False)
            if not torch.isfinite(loss) or loss.detach().item()>1e6:raise ValueError("Unwritable target")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            nll+=float(unc);cells+=n;steps+=1
            if monotonic()-t>1200 or torch.cuda.max_memory_allocated()>8*2**30:raise RuntimeError("Fit budget exceeded")
        exposures+=cells
        losses.append(nll/cells)
        print(arm,steps,losses[-1],flush=True)
    elapsed=monotonic()-t
    result=dict(arm=arm,starting_checkpoint=start,rows=len(rows),digest=digest(rows),updates=steps,answer_positions=exposures,
                conditional_nll=losses,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),weights=save(model,arm))
    record(PUBLIC/(arm+"_fit.json"),result)


def run(arms):
    ds=boards()
    original,_=materialize(ds["train"],"train")
    replay,_=load()
    g=np.random.default_rng(SEED+11)
    replay=[replay[i] for i in g.permutation(len(replay))[:1536]]
    flip=g.integers(2,size=len(ds["train"]))
    for arm in arms:
        torch.cuda.reset_peak_memory_stats()
        rows=original
        if arm=="shuffled":
            rows=[replace(r,q=r.foil) if flip[int(r.key.split("/")[1])] else r for r in original]
        fit(rows+replay,arm,"pointing")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms",nargs="+",choices=("matched","shuffled"))
    a=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        run(a.arms)

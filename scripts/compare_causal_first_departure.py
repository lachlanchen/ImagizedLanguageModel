"""Matched further answer training with additional credit at the first absent glyph."""
import fcntl
import json
import os
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from scripts.train_causal_glyph_writer import (
    BATCH, CausalGlyphWriter, OPEN, PUBLIC, ROOT, WORK, dataset, examples,
    loss_inputs, record, sha,
)
from scripts.evaluate_causal_glyph_writer import evaluate


def run(name, rows):
    writer=CausalGlyphWriter.restore(WORK/"matched","cuda")
    writer.requires_grad_(False)
    writer.body.h[-4:].requires_grad_(True)
    writer.body.ln_f.requires_grad_(True)
    params=[p for p in writer.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=4e-5,fused=True)
    read,out,prompts,answers=dataset(writer,rows)
    first=[]
    for p,a in zip(prompts,answers,strict=True):
        available=set(p)
        first.append(next((j for j,glyph in enumerate(a[:-1]) if glyph not in available),None))
    generator=np.random.default_rng(93654)
    t=monotonic()
    steps=0
    for _ in range(2):
        for ids in np.array_split(generator.permutation(len(rows)),int(np.ceil(len(rows)/BATCH))):
            x,positions,target=loss_inputs(ids,prompts,answers,range(len(rows)),read)
            weights=torch.ones(len(target),device=target.device)
            if name=="departure":
                offsets=np.r_[0,np.cumsum([len(answers[i]) for i in ids])[:-1]]
                ix=[int(k)+first[i] for k,i in zip(offsets,ids,strict=True) if first[i] is not None]
                weights[ix]=8
            with torch.autocast("cuda",dtype=torch.bfloat16):
                state=writer.body(inputs_embeds=x,use_cache=False).last_hidden_state
                logits=(state[positions[0],positions[1]]@out.T).float()
                loss=(F.cross_entropy(logits,target,reduction="none")*weights).sum()/weights.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step()
            steps+=1
            if monotonic()-t>900 or torch.cuda.max_memory_allocated()>19*1024**3:
                raise RuntimeError("Exceeded resource ceiling")
    dest=WORK/name
    dest.mkdir()
    for f in ("config.json","interface.safetensors","metric.safetensors"):
        os.link(WORK/"matched"/f,dest/f)
    save_file({k:v.detach().cpu().contiguous() for k,v in writer.body.state_dict().items()},str(dest/"body.safetensors"))
    return dict(seconds=monotonic()-t,updates=steps,first_departures=sum(v is not None for v in first),
                records=len(rows),peak_bytes=torch.cuda.max_memory_allocated(),body_sha256=sha(dest/"body.safetensors"))


@torch.inference_mode()
def departures(w, records):
    read,out,p,a=dataset(w,records)
    n=hit=wrong_source=0
    for start in range(0,len(records),16):
        ids=list(range(start,min(start+16,len(records))))
        x,pos,target=loss_inputs(ids,p,a,range(len(records)),read)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            h=w.body(inputs_embeds=x,use_cache=False).last_hidden_state
            pred=(h[pos[0],pos[1]]@out.T).argmax(-1).tolist()
        offset=0
        for i in ids:
            source=set(p[i])-{0,len(read)-1}
            j=next((k for k,c in enumerate(a[i][:-1]) if c not in source),None)
            if j is not None:
                n+=1; chosen=pred[offset+j]
                hit+=chosen==a[i][j]
                wrong_source+=chosen in source
            offset+=len(a[i])
    return dict(n=n,correct=hit,chose_source=wrong_source)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        training=examples("train")
        held=examples("development")
        groups=[s["kind"] for s in json.loads((OPEN/"development/sources.json").read_text())]
        results={}
        for name in ("replay","departure"):
            if (WORK/name).exists():raise FileExistsError(WORK/name)
            torch.cuda.reset_peak_memory_stats()
            fit=run(name,training)
            print(name,fit,flush=True)
            w=CausalGlyphWriter.restore(WORK/name,"cuda")
            results[name]={"fit":fit,"first":departures(w,held),
                           "open":evaluate(w,held,groups,name+"_open"),
                           "wrong":evaluate(w,held,groups,name+"_wrong",wrong_passage=True)}
            record(WORK/"departure_partial.json",results)
            print(name,results[name]["first"],results[name]["open"]["all"],flush=True)
            del w
        record(PUBLIC/"first_departure_comparison.json",results)

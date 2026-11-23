"""Fit a next-bitmap writer on visual facts, two sentence orders, and attested clauses."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from scripts.pointing_glyph_corpus import Row,digest
from scripts.grounded_proposition_corpus import LANG,noun
from scripts.prepare_visual_facts import Face,FONTS,ROOT,PUBLIC,WORK,SEED,statement,write_json
from scripts.train_visual_facts import rows
from scripts.train_grounded_propositions import PointingCausalWriter,inputs,objective,sha

START=ROOT/"artifacts/grounded_proposition_v1/varied"
COMMANDS=("先写主题，再说明对应的人或地点。","先写答案，再说明它与主题的关系。","用自然的一句话表述。")


def visual_fields(s,r,o):
    return s+bytes(128)+r+bytes(128)+o


def target(x,mode):
    return statement(x["s"],x["r"],x["o"]) if mode==0 else x["o"]+"是"+noun(x["s"],x["r"])+"。"


def authored_clause(x):
    if x["group"]!="duie":return None
    for clause in re.split(r"[。；;！？\n]",x["p"]):
        c=clause.strip(" ，,：:")
        if 6<=len(c)<=40 and x["s"] in c and x["o"] in c and c not in (target(x,0)[:-1],target(x,1)[:-1]):
            return c+"。"
    return None


def train():
    face=Face((FONTS[0],2));data=rows("training")
    weights=json.loads((ROOT/"references/evidence/grounded_proposition_20261119/varied_fit.json").read_text())["weights"]
    if any(sha(START/n)!=h for n,h in weights.items()):raise ValueError("Source writer differs")
    torch.manual_seed(SEED)
    model=PointingCausalWriter.restore(START,"cuda")
    rng=random.Random(SEED+15)
    pool=rng.sample(data,8000)
    training=[];clauses=0;unreachable=0
    printed={t.tobytes() for t in model.writer.packed.cpu().numpy()}
    def add(x,mode,y):
        s,r,o=x["s"],x["r"],x["o"]
        p=visual_fields(face(s),face(LANG[r][0]),face(o))
        q=face(COMMANDS[mode])
        target_ink=face(y)
        copyable={p[j:j+128] for j in range(0,len(p),128)}
        if any(target_ink[j:j+128] not in printed and target_ink[j:j+128] not in copyable for j in range(0,len(target_ink),128)):
            return False
        training.append(Row("realize",str(len(training)),p,q,target_ink,q))
        return True
    for x in pool:
        for k in range(2):
            if not add(x,k,target(x,k)):raise ValueError("Missing canonical writing support")
    for x in data:
        c=authored_clause(x)
        if c:
            if add(x,2,c):clauses+=1
            else:unreachable+=1
        if clauses>=4096:break
    body=model.writer; body.requires_grad_(False)
    body.body.h[-4:].requires_grad_(True);body.body.ln_f.requires_grad_(True)
    opt=torch.optim.AdamW([dict(params=[p for p in body.parameters() if p.requires_grad],lr=5e-5),
                           dict(params=model.pointer.parameters(),lr=4e-4)],fused=True)
    read,bank,pairs,answers=inputs(body,training)
    order=np.random.default_rng(SEED);t=monotonic();epochs=[];updates=positions=0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(2):
        loss_sum=count=0
        for ids in np.array_split(order.permutation(len(training)),int(np.ceil(len(training)/24))):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,nll,n=objective(model,read,bank,[pairs[i] for i in ids],[answers[i] for i in ids],False)
            if not torch.isfinite(loss) or float(loss.detach())>100:raise ValueError("Divergent next-bitmap loss")
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_sum+=float(nll);count+=n;updates+=1
        epochs.append(loss_sum/count);positions+=count;print("epoch",epoch+1,epochs[-1],flush=True)
    folder=WORK/"writer";tmp=WORK/"writer.partial";tmp.mkdir()
    for f in ("config.json","metric.safetensors","interface.safetensors"):os.link(START/f,tmp/f)
    for net,f in ((body.body,"body"),(model.pointer,"pointer")):
        save_file({k:v.cpu().contiguous() for k,v in net.state_dict().items()},str(tmp/(f+".safetensors")))
    tmp.rename(folder)
    write_json(PUBLIC/"writer_fit.json",dict(rows=len(training),attested_clauses=clauses,unreachable_candidate_clauses=unreachable,paired_records=len(pool),
        exposures=positions,updates=updates,objective_nll=epochs,seconds=monotonic()-t,
        peak_bytes=torch.cuda.max_memory_allocated(),digest=digest(training),weights={p.name:sha(p) for p in folder.iterdir()}))


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(3);train()

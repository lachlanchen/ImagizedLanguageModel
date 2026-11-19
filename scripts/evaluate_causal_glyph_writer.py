"""Retain complete autonomous physical answers and score controlled comparisons."""
from collections import Counter, defaultdict
import fcntl
import json
import os
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.causal_glyph_writer import CausalGlyphWriter, write_strip
from scripts.pictured_participant_scenes import fresh_cases
from scripts.train_causal_glyph_writer import (
    Face, FONTS, OPEN, PUBLIC, ROOT, WORK, dataset, derange, examples, in_cells,
    loss_inputs, record, sha,
)


def score(rows):
    n = len(rows)
    fields = ("exact","stopped","ref_in_source","whole_passage_echo","has_absent","all_absent_in_order","output_length")
    return {"n":n, **{k:sum(r[k] for r in rows) for k in fields},
            "mean_overlap":round(sum(r["overlap"] for r in rows)/n,6)}


@torch.inference_mode()
def evaluate(writer, records, groups, label, native=False, wrong_passage=False):
    t = monotonic()
    if wrong_passage:
        # Keep questions and references in their original order. Replace only the passage.
        ids = derange([p for p,q,y in records])
        inputs = [(records[j][0],q,y) for j,(_,q,y) in zip(ids,records,strict=True)]
    else:
        inputs = records
    read,out,prompts,answers = dataset(writer,inputs,native=native)
    bank = [bytes(v) for v in writer.packed.cpu().numpy()]
    outputs = []
    rows = []
    stats = Counter()
    for start in range(0,len(records),16):
        batch = list(range(start,min(start+16,len(records))))
        x,pos,target = loss_inputs(batch,prompts,answers,range(len(records)),read)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            states = writer.body(inputs_embeds=x,use_cache=False).last_hidden_state
            logits = (states[pos[0],pos[1]] @ out.T).float()
            ids,ended = writer.generate([read[prompts[i]] for i in batch],out,limit=48)
        known = target < len(out)
        stats["covered"] += int(known.sum())
        stats["positions"] += len(known)
        stats["correct_next"] += int((logits.argmax(-1)==target).sum())
        stats["covered_nll_sum"] += float(F.cross_entropy(logits[known],target[known],reduction="sum"))
        for i,idx,stop in zip(batch,ids,ended,strict=True):
            p,q,y = map(in_cells,records[i])
            emitted = [bank[j-1] for j in idx]
            shared = sum((Counter(y)&Counter(emitted)).values())
            absent = [b for b in y if b not in set(p+q)]
            it = iter(emitted)
            all_absent = bool(absent) and all(any(e==b for e in it) for b in absent)
            rows.append(dict(index=i,group=groups[i],exact=stop and emitted==y,stopped=stop,
                             output_length=len(idx),overlap=2*shared/(len(y)+len(emitted)),
                             ref_in_source=any(s[j:j+len(y)]==y for s in (p,q) for j in range(len(s))),
                             whole_passage_echo=emitted==p,has_absent=bool(set(absent)&set(emitted)),
                             all_absent_in_order=all_absent))
            outputs.append(idx)
    path=WORK/(label+".json")
    record(path,dict(outputs=outputs,rows=rows))
    sections = defaultdict(list)
    for row in rows:
        sections[row["group"]].append(row)
    return {"all":score(rows),"groups":{g:score(v) for g,v in sections.items()},
            "conditional":dict(stats)|{"nll_on_covered":stats["covered_nll_sum"]/stats["covered"]},
            "streams_sha256":sha(path),"seconds":monotonic()-t}


def readout():
    render = Face((FONTS[0],2))
    open_rows = examples("development")
    groups = [s["kind"] for s in json.loads((OPEN/"development/sources.json").read_text())]
    role_rows = [tuple(render(s) for s in triple) for triple in fresh_cases()]
    reports = {}
    for name in ("base","matched","shuffled"):
        w = CausalGlyphWriter.restore(WORK/name,"cuda")
        for variant,native,wrong in [("open",False,False)] + (
            [("native",True,False)] if name=="base" else
            [("wrong_passage",False,True),("role",False,False)] if name=="matched" else []):
            key = name+"_"+variant
            r,g = (role_rows,["role"]*len(role_rows)) if variant=="role" else (open_rows,groups)
            reports[key]=evaluate(w,r,g,key,native,wrong)
            print(key,reports[key]["all"],reports[key]["conditional"],flush=True)
            record(WORK/"partial_scores.json",reports)
    record(PUBLIC/"scores.json",reports)
    bank=w.packed.cpu().numpy()
    roles=json.loads((WORK/"matched_role.json").read_text())
    cases=list(fresh_cases())
    reads=json.loads((WORK/"audit_legend.json").read_text())
    originals=[]
    for i,((p,q,y),ids) in enumerate(zip(cases,roles["outputs"],strict=True)):
        originals.append(dict(index=i,passage=p,question=q,target=y,returned="".join(reads[j-1] for j in ids),
                              exact=roles["rows"][i]["exact"],stopped=roles["rows"][i]["stopped"]))
    record(PUBLIC/"role_readings.json",originals)
    for i in (0,1,2,3):
        for field,stream in zip(("passage","question","reference"),role_rows[i],strict=True):
            write_strip(np.frombuffer(stream,np.uint8).reshape(-1,128),WORK/f"role_{i}_{field}.png")
        ids=roles["outputs"][i]
        write_strip(bank[np.array(ids,dtype=int)-1],WORK/f"role_{i}_answer.png")


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Use the assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        readout()

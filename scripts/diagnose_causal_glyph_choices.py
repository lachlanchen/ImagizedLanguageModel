"""Locate the first source-absent glyph under supplied correct answer history."""
from collections import defaultdict
import fcntl
import json
import os

import numpy as np
import torch

from scripts.train_causal_glyph_writer import (
    CausalGlyphWriter, PUBLIC, ROOT, WORK, dataset, examples, loss_inputs, record,
)


@torch.inference_mode()
def main():
    records=examples("development")
    results={}
    for arm in ("base","shuffled","matched"):
        w=CausalGlyphWriter.restore(WORK/arm,"cuda")
        read,out,prompts,answers=dataset(w,records)
        cases=defaultdict(list)
        for start in range(0,len(records),16):
            ids=list(range(start,min(start+16,len(records))))
            x,positions,target=loss_inputs(ids,prompts,answers,range(len(records)),read)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                state=w.body(inputs_embeds=x,use_cache=False).last_hidden_state
                logits=(state[positions[0],positions[1]]@out.T).float()
            offset=0
            for i in ids:
                row=logits[offset:offset+len(answers[i])]
                offset+=len(row)
                source=torch.tensor(sorted(set(prompts[i])-{0,len(read)-1}),device="cuda")
                available=source[source<len(out)]
                gold=torch.tensor(answers[i],device="cuda")
                covered=gold<len(out)
                score=row.gather(1,gold.clamp_max(len(out)-1)[:,None])
                ranks=(row>score).sum(1)+1
                best=row.argmax(-1)
                best_in=torch.isin(best,source)
                present=torch.isin(gold,source)
                mass=row.softmax(1)[:,available].sum(1)
                missing=(~present & (gold!=0)).nonzero().flatten()
                groups={"in_page":present & (gold!=0),"outside_page":~present & (gold!=0),"end":gold==0}
                if len(missing):
                    first=torch.zeros_like(covered); first[missing[0]]=True
                    groups["first_outside_page"]=first
                for group,mask in groups.items():
                    if mask.any():
                        cases[group].append(torch.stack((covered,ranks,best==gold,best_in,best==0,mass),dim=1)[mask].cpu().numpy())
        results[arm]={}
        for group,parts in cases.items():
            a=np.concatenate(parts)
            covered=a[:,0].astype(bool)
            results[arm][group]=dict(positions=len(a),covered=int(covered.sum()),
                correct=int(a[:,2].sum()),best_in_page=int(a[:,3].sum()),best_end=int(a[:,4].sum()),
                median_gold_rank_covered=float(np.median(a[covered,1])),
                top10=int((covered & (a[:,1]<=10)).sum()),mean_page_probability=float(a[:,5].mean()))
    record(PUBLIC/"correct_history.json",results)
    print(json.dumps(results["matched"],indent=2))


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Use assigned GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()

"""Train ordinal span selection and a shared physical writer on single-clause plans."""
from dataclasses import replace
import fcntl
import os
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.pictured_clause_planner import ClausePlan,question_bank,padded_questions
from scripts.extend_grounded_propositions import WITHHELD,pair_key,request
from scripts.grounded_proposition_corpus import Face,FONTS,LANG,materialize,label,statement
from scripts.pointing_glyph_corpus import Row,load
from scripts.train_grounded_propositions import ROOT,WORK,PUBLIC,SEED,PointingCausalWriter,boards,fit,record,sha
from scripts.vary_grounded_questions import PHRASES


def reordered(facts,style):
    s,r,_=facts[0];_,u,_=facts[1];title=label(s,r);a,b=LANG[r][0],LANG[u][0]
    return (f"关于{title}，第二句写{b}，第一句写{a}。",
            f"请介绍{title}。等到说过{a}，再说{b}。",  # surface order follows the answer here
            f"{title}的{b}放在最后，前面先交代{a}。",
            f"谈谈{title}。后一句介绍{b}，开头那一句请介绍{a}。")[style]


def gold_spans(q,facts):
    s,r,_=facts[0]
    terms=(label(s,r),LANG[r][0],LANG[facts[1][1]][0])
    spans=[]
    for j,term in enumerate(terms):
        # A title may contain an ordinary role word. Restrict the generic pointer
        # label to the actual instruction, after the title in these constructions.
        begin=0 if j==0 else spans[0][1]
        first=q.index(term,begin)
        spans.append((first,first+len(term)))
    return spans


def build_sets():
    face=Face((FONTS[0],2))
    ds=boards()["train"]
    active=[b for b in ds if b["kind"]=="relation" and pair_key(b) not in WITHHELD]
    two,tm=materialize(active,"train",True)
    single,sm=materialize(ds,"train")
    earlier,_=load()
    rng=np.random.default_rng(SEED+680)
    train_plan={};clauses=[]
    for row,m in zip(two,tm):
        facts=m["facts"];i=m["board"]
        for style in range(4):
            for text in [request(facts,style),reordered(facts,style)]:
                spans=gold_spans(text,facts)
                train_plan[face(text)]=spans
        s,r,_=facts[0]
        for f in facts:
            sub=face(label(s,r)+LANG[f[1]][0])
            clauses.append(Row("planned",row.key,row.p,sub,face(statement(*f)),sub))
    retained=[]
    for n in rng.permutation(len(single))[:3600]:
        row=single[n];m=sm[n];s,r,_=m["facts"][0]
        if rng.random()<.65:
            prefix=("关于"+label(s,r)+"，") if row.task=="subject" or rng.random()<.4 else ""
            q=face(prefix+str(rng.choice(PHRASES[r]))+"？请写成一句话。")
            row=replace(row,q=q)
        train_plan[row.q]=None;retained.append(row)
    for n in rng.permutation(len(earlier))[:1536]:
        train_plan[earlier[n].q]=None;retained.append(earlier[n])
    # Complementary (fact, page) examples duplicate under the opposite instruction order.
    # A single call sees identical inputs and targets for these duplicates.
    unique={(r.p,r.q,r.y):r for r in clauses}
    return list(unique.values())+retained,train_plan,len(unique)


def train_plan_head(model,items):
    torch.manual_seed(SEED+681)
    g=np.random.default_rng(SEED+682)
    vectors,ids=question_bank(model.writer,list(items))
    labels=list(items.values())
    two=torch.tensor([v is not None for v in labels],device="cuda")
    coords=torch.tensor([[[a,b-1] for a,b in v] if v else [[0,0]]*3 for v in labels],device="cuda").reshape(-1,6)
    head=ClausePlan().cuda().train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001)
    t=monotonic();steps=0
    for epoch in range(10):
        for batch in np.array_split(g.permutation(len(ids)),int(np.ceil(len(ids)/64))):
            x,n=padded_questions([ids[i] for i in batch],vectors.device)
            count,pos=head(vectors[x],n)
            target=two[batch]
            loss=F.binary_cross_entropy_with_logits(count,target.float())
            if target.any():
                p=pos[target].transpose(1,2)
                loss=loss+F.cross_entropy(p.reshape(-1,p.size(-1)),coords[batch][target].reshape(-1))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            steps+=1
        if monotonic()-t>1200 or torch.cuda.max_memory_allocated()>8*2**30:raise RuntimeError("Planner exceeded allocation")
    file=WORK/"planned/plan.safetensors"
    save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(file))
    return dict(seconds=monotonic()-t,updates=steps,questions=len(ids),compound=int(two.sum()),parameters=sum(p.numel() for p in head.parameters()),weights_sha256=sha(file))


def main():
    data,labels,n=build_sets()
    fit(data,"planned","varied",epochs=2)
    net=PointingCausalWriter.restore(WORK/"planned","cuda")
    torch.cuda.reset_peak_memory_stats()
    info=train_plan_head(net,labels)
    record(PUBLIC/"plan_head.json",dict(**info,unique_clause_inputs=n,peak_bytes=torch.cuda.max_memory_allocated()))
    print(info,flush=True)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);main()

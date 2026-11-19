"""Learn to choose a pictured evidence window from the frozen writer's reading state."""
import fcntl
import json
import math
import os
from collections import Counter
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter, layout, padded
from scripts.evaluate_pointing_causal_writer import generate, store_streams
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_composed_chinese_answers import Rasterizer
from scripts.train_causal_glyph_writer import in_cells
from scripts.train_pointing_causal_writer import PUBLIC, ROOT, WORK, inputs, record, sha


@torch.inference_mode()
def representation(net, items, face):
    rows=[]
    offsets=[0]
    for item in items:
        q=face(item["question"])
        rows.extend(Row("candidate","",face(p),q,b"\0"*128,q) for p in item["passages"])
        offsets.append(len(rows))
    read,_,pairs,_=inputs(net.writer,rows)
    feats=[None]*len(rows)
    order=sorted(range(len(rows)),key=lambda i:sum(map(len,pairs[i])))
    for start in range(0,len(order),24):
        ix=order[start:start+24]
        prompts,first=zip(*(layout(*pairs[i],len(read)-1) for i in ix))
        ids,mask=padded(prompts,read.device)
        src,valid=padded([pairs[i][0] for i in ix],read.device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            h=net.writer.body(inputs_embeds=read[ids],attention_mask=mask,use_cache=False).last_hidden_state
            batch=torch.arange(len(ix),device=read.device)
            last=h[batch,mask.sum(1)-1][:,None]
            keys=net.source_keys(h,first,src.size(1),valid)
            score,gate=net.pointer.scores(last,keys,src,valid,torch.full((len(ix),1),-1,device=read.device),False)
            a=score[:,0].softmax(-1)
            pos=torch.tensor(first,device=read.device)[:,None]+torch.arange(src.size(1),device=read.device)
            context=(a[:,:,None]*h[batch[:,None],pos.clamp_max(h.size(1)-1)].float()).sum(1)
        vector=torch.cat([F.layer_norm(last[:,0].float(),(last.size(-1),)),F.layer_norm(context,(context.size(-1),)),
                          a.max(1).values[:,None],gate[:,0],-(a*a.clamp_min(1e-30).log()).sum(1,keepdim=True)],dim=1)
        for i,row in zip(ix,vector.cpu(),strict=True):feats[i]=row
    return [torch.stack(feats[x:y]) for x,y in zip(offsets[:-1],offsets[1:])]


def train_ranker(features, items, shuffled):
    torch.manual_seed(53210)
    rng=np.random.default_rng(53211)
    net=nn.Sequential(nn.Linear(features[0].size(1),96),nn.GELU(),nn.Linear(96,1)).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=5e-4)
    xs=[];ys=[]
    for f,item in zip(features,items,strict=True):
        labels=np.array([any(a in p for a in item["refs"] if 4<=len(a)<=48) for p in item["passages"]])
        if labels.all() or not labels.any():continue
        if shuffled:labels=np.roll(labels,1)
        xs.append(f);ys.append(torch.tensor(labels))
    if len(xs)<32:raise ValueError("Insufficient questions with positive and negative windows")
    pad=nn.utils.rnn.pad_sequence(xs,batch_first=True).cuda()
    y=nn.utils.rnn.pad_sequence(ys,batch_first=True).cuda()
    valid=torch.arange(pad.size(1),device="cuda")[None]<torch.tensor([len(x) for x in xs],device="cuda")[:,None]
    for _ in range(8):
        for ids in np.array_split(rng.permutation(len(xs)),math.ceil(len(xs)/32)):
            logits=net(pad[ids])[:,:,0].masked_fill(~valid[ids],-1e30)
            loss=-logits.log_softmax(1).masked_fill(~y[ids],-1e30).logsumexp(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net.eval(),len(xs)


def evaluate(net,items,features,rankers,face,split):
    decisions={}
    for i,(item,f) in enumerate(zip(items,features,strict=True)):
        ps=item["passages"]
        with torch.inference_mode():
            ids={name:int(m(f.cuda()).argmax()) for name,m in rankers.items()}
        ids.update(original=0,bm25=item["bm25"])
        ids["reference_assisted"]=next((j for j,p in enumerate(ps) if item["target"] in p),
                                    next((j for j,p in enumerate(ps) if any(a in p for a in item["refs"])),0))
        decisions[i]=ids
    keys=list(dict.fromkeys((i,j) for i,choices in decisions.items() for j in choices.values()))
    rows=[]
    for i,j in keys:
        a=items[i]
        rows.append(Row(a["group"],str(i),face(a["passages"][j]),face(a["question"]),face(a["target"]),b""))
    streams,stops=generate(net,rows)
    store_streams(WORK/(split+"_ranked_streams.npz"),streams,stops)
    index=dict(zip(keys,zip(streams,stops),strict=True))
    stats={}
    all_rows={}
    for method in decisions[0]:
        measured=[]
        for i,a in enumerate(items):
            j=decisions[i][method]
            p=a["passages"][j]
            y,stop=index[i,j]
            ref=face(a["target"])
            refs=[face(x) for x in a["refs"]]
            cy=Counter(in_cells(y))
            def overlap(v):
                bits=in_cells(v)
                return 2*sum((cy&Counter(bits)).values())/(sum(cy.values())+len(bits))
            measured.append(dict(key=a["key"],group=a["group"],selected=j,exact=stop and y==ref,
                                 multi=stop and y in refs,visible=a["target"] in p,any_visible=any(t in p for t in a["refs"]),
                                 novel=any(c.isalnum() and c not in p+a["question"] for c in a["target"]),
                                 f1=max(overlap(v) for v in refs),stopped=stop))
        def counts(rows):
            return dict(n=len(rows),**{k:sum(r[k] for r in rows) for k in ("exact","multi","visible","any_visible","novel","stopped")},
                        exact_still_novel=sum(r["exact"] and r["novel"] for r in rows),mean_overlap=float(np.mean([r["f1"] for r in rows])))
        stats[method]={"all":counts(measured),"groups":{g:counts([r for r in measured if r["group"]==g]) for g in sorted({r["group"] for r in measured})}}
        all_rows[method]=measured
    record(WORK/(split+"_ranked_cases.json"),dict(rows=all_rows,stream_index=keys))
    return stats


def main():
    corpus=json.loads((WORK/"windows.json").read_text())
    assert sha(WORK/"windows.json")==json.loads((PUBLIC/"windows.json").read_text())["prepared_sha256"]
    raster=Rasterizer()
    def face(s):return b"".join(raster.tile(c) for c in s)
    net=PointingCausalWriter.restore(WORK/"pointing","cuda")
    t=monotonic()
    features={k:representation(net,items,face) for k,items in corpus.items()}
    rankers={};train_count={}
    for name in ("learned","shuffled"):
        m,n=train_ranker(features["train"],corpus["train"],name=="shuffled")
        rankers[name]=m;train_count[name]=n
        save_file({k:v.cpu().contiguous() for k,v in m.state_dict().items()},str(WORK/(name+"_ranker.safetensors")))
    result={"training_questions":train_count,"parameters":sum(p.numel() for p in rankers["learned"].parameters())}
    for split in ("development","fresh"):
        result[split]=evaluate(net,corpus[split],features[split],rankers,face,split)
        print(split,{k:v["all"] for k,v in result[split].items()},flush=True)
    result.update(seconds=monotonic()-t,peak_bytes=torch.cuda.max_memory_allocated(),writer_sha256=sha(WORK/"pointing/body.safetensors"))
    record(PUBLIC/"evidence_ranking.json",result)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()

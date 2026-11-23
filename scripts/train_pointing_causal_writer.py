"""Compare question-conditioned visual pointing with two matched controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter, layout, padded
from scripts.pointing_glyph_corpus import load, digest
from scripts.train_causal_glyph_writer import CausalGlyphWriter, ROOT, WORK as PRIOR, dataset, record, sha


WORK = ROOT/"artifacts/pointing_causal_writer_v1"
PUBLIC = ROOT/"references/evidence/pointing_causal_writer_20261119"
TRAIN = "c97a475c90c9ec0dcc93026a6c73c985d0cba4665371f6c70013702f769b3b0d"
HELD = "10b29d73213e35d111b4815690b2dfd3bba092a0937054d8a6c6504d11710906"
SEED = 15324


def inputs(writer, rows, foil=False):
    triples=[(r.p,r.foil if foil else r.q,r.y) for r in rows]
    read,out,prompts,answers=dataset(writer,triples)
    pairs=[]
    for prompt in prompts:
        br=prompt.index(0)
        pairs.append((prompt[1:br],prompt[br+1:-1]))
    return read,out,pairs,answers


def starts(p,y):
    n=len(y)
    return [j for j,c in enumerate(p) if c==y[0] and p[j:j+n]==y]


def focused_action_loss(log_probs, answers, focus):
    """Mean next-ink NLL on one selected action of each eligible book request."""
    offsets = []
    start = 0
    for actions, index in zip(answers, focus, strict=True):
        if index is not None:
            if not 0 <= index < len(actions):
                raise ValueError("Focused action outside the supervised reply")
            offsets.append(start + index)
        start += len(actions)
    if start != log_probs.numel():
        raise ValueError("Incorrect number of supervised glyph probabilities")
    return -log_probs[offsets].mean() if offsets else log_probs.sum() * 0


def objective(model,read,out,pairs,answers,uniform,additional_loss=None,focus=None):
    device=out.device
    sequences=[]
    offsets=[]
    begin=[]
    prev=[]
    exact_starts=[]
    for (p,q),y in zip(pairs,answers,strict=True):
        prompt,first=layout(p,q,len(read)-1)
        sequences.append(prompt+y[:-1])
        offsets.append(len(prompt)-1)
        begin.append(first)
        prev.append([-1]+y[:-1])
        exact_starts.append(starts(p,y[:-1]))
    ids,mask=padded(sequences,device)
    source,valid=padded([p for p,q in pairs],device)
    targets,loss_mask=padded(answers,device)
    previous,_=padded(prev,device)
    h=model.writer.body(inputs_embeds=read[ids],attention_mask=mask,use_cache=False).last_hidden_state
    keys=model.source_keys(h,begin,source.size(1),valid)
    positions=torch.tensor(offsets,device=device)[:,None]+torch.arange(targets.size(1),device=device)
    states=h[torch.arange(len(answers),device=device)[:,None],positions.clamp_max(h.size(1)-1)]
    pointer,gate=model.pointer.scores(states,keys,source,valid,previous,uniform)
    vocab=(states[loss_mask]@out.T).float().log_softmax(-1)
    labels=targets[loss_mask]
    gen=vocab.gather(1,labels.clamp_max(len(out)-1)[:,None])[:,0].masked_fill(labels>=len(out),-1e30)
    same=(targets[:,:,None]==source[:,None,:]) & valid[:,None,:]
    att=pointer.log_softmax(-1)
    copy=att.masked_fill(~same,-1e30).logsumexp(-1)[loss_mask]
    model_gate=gate.squeeze(-1)
    gate=model_gate[loss_mask]
    likelihood=torch.logaddexp(gen+F.logsigmoid(gate),copy+F.logsigmoid(-gate))
    loss=-likelihood.mean()
    if focus is not None:
        loss = loss + focused_action_loss(likelihood, answers, focus)
    which=[i for i,idx in enumerate(exact_starts) if idx]
    if which:
        aligned=torch.zeros_like(valid)
        for i in which: aligned[i,exact_starts[i]]=True
        aux=-att[which,0].masked_fill(~aligned[which],-1e30).logsumexp(-1).mean()
        loss=loss+.35*aux
    if additional_loss is not None:
        stop_logp = pointer.new_zeros(loss_mask.shape).masked_scatter(loss_mask,vocab[:,0])
        loss = loss + additional_loss(pointer, model_gate, stop_logp, source, valid, targets, loss_mask)
    return loss, -likelihood.detach().sum(),len(labels)


def fit(rows,name):
    if (WORK/name).exists():raise FileExistsError(WORK/name)
    started=monotonic()
    torch.manual_seed(SEED)
    writer=CausalGlyphWriter.restore(PRIOR/"matched","cuda")
    expected=json.loads((ROOT/"references/evidence/causal_glyph_writer_20261119/matched_fit.json").read_text())["files"]
    assert all(sha(PRIOR/"matched"/f)==v for f,v in expected.items())
    net=PointingCausalWriter(writer).cuda().eval()
    writer.requires_grad_(False)
    writer.body.h[-4:].requires_grad_(True)
    writer.body.ln_f.requires_grad_(True)
    opt=torch.optim.AdamW([{"params":[p for p in writer.parameters() if p.requires_grad],"lr":6e-5},
                            {"params":net.pointer.parameters(),"lr":5e-4}],fused=True)
    read,out,pairs,answers=inputs(writer,rows,foil=name=="foil")
    rng=np.random.default_rng(SEED)
    trace=[]
    steps=0
    for epoch in range(2):
        total_nll=0.;total_cells=0
        for group in np.array_split(rng.permutation(len(rows)),int(np.ceil(len(rows)/16))):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,nll,n=objective(net,read,out,[pairs[i] for i in group],[answers[i] for i in group],name=="uniform")
            if float(loss)>1e8:raise ValueError("A training target is outside all output routes")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            total_nll+=float(nll);total_cells+=n;steps+=1
            if steps%100==0:print(name,steps,round(total_nll/total_cells,3),round(monotonic()-started,1),flush=True)
            if monotonic()-started>1200 or torch.cuda.max_memory_allocated()>20*2**30:
                raise RuntimeError("Exceeded training budget")
        trace.append(total_nll/total_cells)
    dest=WORK/name
    tmp=WORK/(name+".staging")
    tmp.mkdir()
    for f in ("config.json","interface.safetensors","metric.safetensors"):
        os.link(PRIOR/"matched"/f,tmp/f)
    for mod,f in ((net.pointer,"pointer"),(writer.body,"body")):
        save_file({k:v.detach().cpu().contiguous() for k,v in mod.state_dict().items()},str(tmp/(f+".safetensors")))
    tmp.rename(dest)
    report=dict(train_digest=TRAIN,held_digest=HELD,records=len(rows),updates=steps,positions=total_cells*2,
                mean_training_nll=trace,peak_bytes=torch.cuda.max_memory_allocated(),seconds=monotonic()-started,
                pointer_parameters=sum(p.numel() for p in net.pointer.parameters()),
                fitted=sum(p.numel() for p in net.parameters() if p.requires_grad),
                weights={p.name:sha(p) for p in dest.iterdir()})
    record(PUBLIC/(name+"_fit.json"),report)
    print(name,report,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms",nargs="+",choices=("pointing","uniform","foil"))
    a=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use the assigned GPU")
    WORK.mkdir(exist_ok=True)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        rows,held=load()
        assert digest(rows)==TRAIN and digest(held)==HELD
        for name in a.arms:
            torch.cuda.reset_peak_memory_stats()
            fit(rows,name)

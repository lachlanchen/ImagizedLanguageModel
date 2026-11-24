"""Fit matched visual dependency readers, sharing the pictured-question router."""

from collections import defaultdict
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.visual_clause_roles import VisualClauseRoles
from scripts.answer_visual_gloss import load_focus
from scripts.everyday_gloss_questions import TRAIN, BROAD
from scripts.prepare_visual_clause_roles import ROOT, PUBLIC, WORK, read, dump, sha
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face, FACES
from scripts.train_span_choice import gpu_owner
from scripts.write_visual_gloss import restore
from scripts.measure_writer_energy import snapshot, consumed

SEED=641203
ARMS=("uniform","balanced","exchanged")


def run():
    reg=read(PUBLIC/"split.json")
    assert sha(WORK/"train.json")==reg["train"]
    rows=read(WORK/"train.json")
    face=Face(FACES[0]); writer=restore(device="cuda").requires_grad_(False)
    start=monotonic(); energy=snapshot()
    lengths=torch.tensor([len(r["page"]) for r in rows],device="cuda")
    width=int(lengths.max())
    states=torch.zeros(len(rows),width,160,device="cuda",dtype=torch.float16)
    truth=np.array([r["focus"] for r in rows])
    groups=defaultdict(list)
    for j,r in enumerate(rows):
        groups[tuple(face.cell(r["page"][i]) for i in r["focus"])].append(j)
    for a in range(0,len(rows),96):
        batch=rows[a:a+96];raw,_=packed_batch([r["page"] for r in batch],face,"cuda")
        with torch.no_grad():
            ctx=encoding_windows(writer,cell_features(raw))
            states[a:a+len(batch),:ctx.shape[1]]=ctx
    torch.manual_seed(SEED)
    initial=VisualClauseRoles().cuda()
    parent=load_focus(ROOT/"models/everyday_gloss","cuda")
    initial.load_state_dict(parent.state_dict(),strict=False)
    questions=[s for pair in TRAIN+BROAD for s in pair]
    raw,ql=packed_batch(questions,face,"cuda")
    q=cell_features(raw)
    # Every arm sees the identical independently trained question classifier.
    qp=[*initial.q_input.parameters(),*initial.q_sequence.parameters(),*initial.query.parameters(),*initial.role.parameters()]
    opt=torch.optim.AdamW(qp,lr=.0006)
    role=torch.arange(len(q),device="cuda")%2
    for _ in range(160):
        loss=F.cross_entropy(initial.role(initial.read_questions(q,ql)),role)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        accuracy=(initial.role(initial.read_questions(q,ql)).argmax(-1)==role).sum().item()
    dump(PUBLIC/"preparation.json",dict(seconds=monotonic()-start,energy=consumed(energy,snapshot()),question_fit_correct=accuracy,questions=len(q),updates=160))
    front=np.where(truth[:,0]>truth[:,1])[0]
    post=np.where(truth[:,0]<truth[:,1])[0]
    ref=torch.as_tensor(truth,device="cuda")
    head_id=np.empty(len(rows),np.int32)
    for j,indices in enumerate(groups.values()):head_id[indices]=j
    for arm in ARMS:
        folder=WORK/arm;folder.mkdir(exist_ok=False)
        net=VisualClauseRoles().cuda()
        net.load_state_dict(initial.state_dict())
        opt=torch.optim.AdamW([p for n,p in net.named_parameters() if n.split('.')[0] in
                {"page","keys","endpoints","verb_link","object_link"}],lr=.001)
        start=monotonic(); energy=snapshot();torch.cuda.reset_peak_memory_stats()
        steps=0;exposures=0;front_count=0;trace=[]
        for epoch in range(12):
            rng=np.random.default_rng(SEED+epoch)
            order=rng.permutation(len(rows))
            if arm!="uniform":
                nf=len(rows)//4
                order=rng.permutation(np.r_[rng.choice(front,nf,replace=True),
                                           rng.choice(post,len(rows)-nf,replace=False)])
            reverse=np.zeros(len(order),dtype=bool)
            if arm=="exchanged":
                # Flip half of each physical headword pair's presentations.
                # Singleton groups receive a fair coin.
                for group in np.split(np.argsort(head_id[order],kind="stable"),np.cumsum(np.unique(head_id[order],return_counts=True)[1])[:-1]):
                    n=(len(group)+int(rng.integers(2)))//2
                    reverse[rng.permutation(group)[:n]]=True
            total=0.
            for a in range(0,len(order),128):
                ix=order[a:a+128];w=int(lengths[ix].max())
                y=ref[ix].clone()
                flip=reverse[a:a+len(ix)]
                y[flip]=y[flip].flip(-1)
                scores=net.arc_scores(states[ix,:w].float(),lengths[ix])
                loss=F.cross_entropy(scores.flatten(1),y[:,0]*w+y[:,1])
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
                total+=float(loss.detach())*len(ix);steps+=1
            trace.append(total/len(order)); exposures+=len(order)
            front_count+=int((truth[order,0]>truth[order,1]).sum())
            if monotonic()-start>300 or torch.cuda.max_memory_allocated()>6*2**30:
                raise RuntimeError("Visual arc fit exceeded its registered ceiling")
        file=folder/"focus.safetensors"
        save_file({k:v.cpu().contiguous() for k,v in net.state_dict().items()},str(file))
        spec=dict(architecture="visual_clause_roles",sha256=sha(file),parameters=sum(p.numel() for p in net.parameters()),
              rows=len(rows),updates=steps,presentations=exposures,fronted_presentations=front_count,
              seconds=monotonic()-start,energy=consumed(energy,snapshot()),peak_bytes=torch.cuda.max_memory_allocated(),loss=trace)
        dump(folder/"weights.json",spec);dump(PUBLIC/(arm+"_fit.json"),spec)
        print(arm,steps,trace[-1],flush=True)


if __name__=="__main__":
    with gpu_owner():
        run()

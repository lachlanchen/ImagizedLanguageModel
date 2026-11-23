"""Distil Chinese question-conditioned interval probabilities into a bilingual pixel reader."""

import argparse
import json
from time import monotonic

import numpy as np
import torch
from torch.nn.functional import kl_div
from safetensors.torch import load_file,save_file

from ilm.visual_lm.book_participant_reader import BookParticipantReader,InkBank,best_strip,group_spans,padded,positions
from ilm.visual_lm.book_person_proposals import PersonDetector,propose,select_reply
from scripts.prepare_book_span_distillation import LAST,PREVIOUS,PUBLIC,WORK
from scripts.train_book_participant_reader import dataset,batch,candidate_mass
from scripts.measure_writer_energy import consumed,snapshot
from scripts.train_causal_glyph_writer import sha,record
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES,Face


def fit(arm):
    spec=json.loads((PUBLIC/"corpus.json").read_text())
    file=WORK/"pairs.json"
    cache=WORK/"teacher_targets.npz"
    assert sha(file)==spec["sha256"]
    bank=InkBank(); face=Face(FACES[0]); rows=[]
    for i,row in enumerate(json.loads(file.read_text())):
        page=bank.encode(face(row["page"]))
        items=[]
        for lang in (("zh","en") if "en" in row["qs"][0] else ("zh",)):
            for j,q in enumerate(row["qs"]):
                y=positions(page,bank.encode(face(q["answer"])))
                assert y
                items.append((bank.encode(face(q[lang])),y,2*i+(j if arm=="matched" else 1-j)))
        rows.append((page,items))
    if arm != "human":
        assert sha(cache)==json.loads((PUBLIC/"teacher_learn.json").read_text())["sha256"]
        with np.load(cache) as f:
            index=torch.from_numpy(f["indices"].astype(np.int64)).cuda()
            weights=torch.from_numpy(f["prob"]).cuda()
            tail=torch.from_numpy(f["residual"]).cuda()
        assert len(index)==2*len(rows) and (weights.sum(-1)+tail-1).abs().max()<1e-5
    pixels=bank.raster("cuda")
    torch.manual_seed(241127); rng=torch.Generator().manual_seed(241128)
    net=BookParticipantReader().cuda();folder=WORK/arm;folder.mkdir(exist_ok=False)
    save_file(net.state_dict(),folder/"initial.safetensors")
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    updates=8*((len(rows)+11)//12)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,updates,eta_min=.0001)
    first=snapshot();t=monotonic();trace=[]
    for epoch in range(8):
        losses=[]
        for ids in torch.randperm(len(rows),generator=rng).split(12):
            batch_items=[(p,q,y,k) for b in ids.tolist() for p,items in [rows[b]] for q,y,k in items]
            p,q,human,k=zip(*batch_items,strict=True)
            pg=padded(p,"cuda")
            z=net(pixels,pg,padded(q,"cuda")).flatten(1)
            occurrences=padded([[i+1 for i in y] for y in human],"cuda")-1
            gold=z.gather(1,occurrences.clamp_min(0)).masked_fill(occurrences<0,-torch.inf).logsumexp(1)
            hard=(z.logsumexp(1)-gold).mean()
            if arm == "human":
                soft = hard.new_zeros(())
                loss = hard
            else:
                k=list(k); kept=index[k]; w=weights[k]; residual=tail[k]
                s=z/2
                remaining=s.scatter(1,kept,-1e4)
                bins=torch.cat([s.gather(1,kept),remaining.logsumexp(1,keepdim=True)],dim=1)
                target=torch.cat([w,residual[:,None]],dim=1)
                # A padded top-k entry has zero target mass and is excluded from the student normalizer.
                bins[:,:-1]=bins[:,:-1].masked_fill(w<=0,-1e4)
                soft=kl_div(bins.log_softmax(1),target,reduction="batchmean")
                loss=.5*(hard+4*soft)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();sched.step()
            losses.append([float(hard.detach()),float(soft.detach())])
        trace.append(np.mean(losses,axis=0).tolist())
        print(arm,epoch+1,trace[-1],flush=True)
        if monotonic()-t>1200 or torch.cuda.max_memory_allocated()>12*1024**3:raise RuntimeError("Compute ceiling")
    elapsed=monotonic()-t;energy=consumed(first,snapshot())
    weights_file=folder/"reader.safetensors";save_file(net.state_dict(),weights_file)
    record(PUBLIC/f"{arm}_fit.json",dict(initial=sha(folder/"initial.safetensors"),weights=sha(weights_file),
        epochs=8,updates=updates,question_presentations=8*sum(len(qs) for p,qs in rows),seconds=elapsed,
        parameters=sum(p.numel() for p in net.parameters()),peak_bytes=torch.cuda.max_memory_allocated(),energy=energy,loss=trace))


@torch.inference_mode()
def evaluate(arm):
    bank,rows=dataset("reserved");pixels=bank.raster("cuda")
    net=BookParticipantReader().cuda()
    file=WORK/arm/"reader.safetensors";assert sha(file)==json.loads((PUBLIC/f"{arm}_fit.json").read_text())["weights"]
    net.load_state_dict(load_file(str(file)))
    detector=PersonDetector().cuda();file=LAST/"person_detector.safetensors"
    assert sha(file)==json.loads((PREVIOUS/"person_fit.json").read_text())["weights"]
    detector.load_state_dict(load_file(str(file)))
    detail=[]
    for i in range(0,len(rows),8):
        group=rows[i:i+8]
        p,q,occ=batch(group,"cuda")
        z=net(pixels,p,q); prior=detector(pixels,p[::4]).cpu().numpy()
        logits=candidate_mass(z,occ)
        margins=(logits[:,0]-logits[:,1]).view(-1,4).cpu().numpy()
        z=z.cpu().numpy()
        for j,r in enumerate(group):
            groups=group_spans(r["page"]);names=propose(prior[j],groups)
            facts=[tuple(s) for s in r["names"]]
            ans=[]
            for h in range(4):
                s=z[4*j+h,:len(r["page"])]
                a,b=best_strip(s,groups),select_reply(s,groups,names)
                ans.append(dict(restricted=float(margins[j,h])*(1-2*(h%2))>0,open=a==facts[h%2],
                                proposed=b==facts[h%2],image=b"".join(bank.cells[n] for n in a).hex(),
                                proposed_image=b"".join(bank.cells[n] for n in b).hex()))
            detail.append(dict(key=r["key"],work=r["work"],replies=ans))
    file=WORK/f"{arm}_responses.json";record(file,detail)
    counts={}
    for w in sorted({r["work"] for r in detail})+["all"]:
        subset=[r for r in detail if w=="all" or r["work"]==w]
        for i,lang in enumerate(("zh","en")):
            c=dict(pages=len(subset),questions=2*len(subset))
            for mode in ("open","restricted","proposed"):
                x=np.array([[s[mode] for s in r["replies"][2*i:2*i+2]] for r in subset])
                c[mode+"_both"]=int(x.all(1).sum());c[mode+"_individual"]=int(x.sum())
            counts[w+"/"+lang]=c
    record(PUBLIC/f"{arm}_reserved.json",dict(sha256=sha(file),counts=counts))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("arm",choices=("matched","exchanged","human"));p.add_argument("phase",choices=("fit","evaluate"))
    a=p.parse_args()
    with gpu_owner():
        torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
        (fit if a.phase=="fit" else evaluate)(a.arm)

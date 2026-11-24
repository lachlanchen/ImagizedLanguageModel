"""Distil the teacher's varied question images with a fixed passage arc and writer."""

from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.gloss_focus import cell_features
from ilm.visual_lm.visual_clause_roles import VisualClauseRoles
from scripts.answer_visual_gloss import load_focus
from scripts.everyday_gloss_questions import TRAIN,BROAD
from scripts.prepare_visual_clause_roles import ROOT,WORK,PUBLIC,read,dump,sha
from scripts.train_visual_clause_roles import SEED
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face,FACES
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot,consumed


def run():
    review=read(PUBLIC/"teacher_review.json")
    assert sha(PUBLIC/"teacher_questions.json")==review["source"]
    offered=read(PUBLIC/"teacher_questions.json")["kept"]
    retained=[e["pair"] for i,e in enumerate(offered) if str(i) not in review["exclude"]]
    templates=TRAIN+BROAD+retained
    q,ql=packed_batch([q for pair in templates for q in pair],Face(FACES[0]),"cuda")
    q=cell_features(q)
    for arm in ("distilled","distilled_exchanged"):
        out=WORK/arm;out.mkdir(exist_ok=False)
        torch.manual_seed(SEED)
        initial=VisualClauseRoles().cuda()
        initial.load_state_dict(load_focus(ROOT/"models/everyday_gloss","cuda").state_dict(),strict=False)
        net=load_focus(WORK/"balanced","cuda").train()
        net.requires_grad_(False)
        layers=("q_input","q_sequence","query","role")
        params=[]
        for n in layers:
            m=getattr(net,n);m.load_state_dict(getattr(initial,n).state_dict());m.requires_grad_(True)
            params.extend(m.parameters())
        opt=torch.optim.AdamW(params,lr=.0006)
        rng=np.random.default_rng(SEED+9); corrupt=np.random.default_rng(SEED+10)
        energy=snapshot();start=monotonic()
        for step in range(160):
            ids=np.r_[rng.choice(40,16,replace=False),rng.choice(len(retained),24,replace=False)+40]
            ix=(2*ids[:,None]+np.arange(2)).flatten()
            target=np.tile(np.arange(2),(len(ids),1))
            if arm.endswith("exchanged"):target^=corrupt.integers(0,2,(len(ids),1))
            logits=net.role(net.read_questions(q[ix],ql[ix]))
            loss=F.cross_entropy(logits,torch.tensor(target.flatten(),device="cuda"))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        file=out/"focus.safetensors"
        save_file({k:v.cpu().contiguous() for k,v in net.state_dict().items()},str(file))
        spec=dict(architecture="visual_clause_roles",sha256=sha(file),parameters=sum(p.numel() for p in net.parameters()),
              trained=sum(p.numel() for p in params),parent=sha(WORK/"balanced/focus.safetensors"),updates=160,
              image_presentations=160*80,teacher_pairs=len(retained),energy=consumed(energy,snapshot()),
              seconds=monotonic()-start,teacher_data=review["source"],review=sha(PUBLIC/"teacher_review.json"))
        dump(out/"weights.json",spec);dump(PUBLIC/(arm+"_fit.json"),spec)
        print(arm,spec["seconds"],flush=True)

if __name__=="__main__":
    with gpu_owner():run()

"""Prepare real human questions and a frozen optical teacher's interval probabilities."""

import argparse
from collections import Counter
from itertools import combinations
import json
from time import monotonic

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ilm.visual_lm.book_participant_reader import InkBank, WIDTH, best_strip, end_values, group_spans
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from scripts.fit_font_invariant_reader import WORK as TEACHER
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_book_participant_questions import HELD
from scripts.train_bilingual_novel_writer import WORK as BOOK, PUBLIC as BOOK_PUBLIC
from scripts.train_book_participant_reader import WORK as LAST, PUBLIC as PREVIOUS
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face,FACES

WORK=ROOT/"artifacts/book_span_distillation_v1"
PUBLIC=ROOT/"references/evidence/book_span_distillation_20261124"


def prepare():
    WORK.mkdir(exist_ok=True); PUBLIC.mkdir(exist_ok=True,parents=True)
    face=Face(FACES[0]); seen={r["page"] for r in json.loads((LAST/"reserved.json").read_text())}
    def renderable(text):
        return bool(text) and all(c.isspace() or ord(c) in face.coverage for c in text)
    rows=[]
    source_counts=Counter()
    hashes={}
    def add(source,page,questions,limit):
        if page in seen or len(page)>506 or not renderable(page) or any(
            s in page.lower() for s in ("哈利","三体","三體","harry potter","three-body")
        ): return
        qs=[q for q in questions if renderable(q["zh"]) and len(q["zh"])+len(page)+3<=512
            and (not q.get("en") or renderable(q["en"])) and 0<len(q["answer"])<=WIDTH
            and q["answer"] in page and q["answer"].strip()]
        used=set()
        for a,b in combinations(qs,2):
            if a["id"] in used or b["id"] in used or a["answer"] in b["answer"] or b["answer"] in a["answer"]:continue
            rows.append(dict(source=source,page=page,qs=[a,b]))
            used.update([a["id"],b["id"]])
            if len(used)==2*limit: break
        if used:seen.add(page);source_counts[source]+=1
    path=BOOK/"train.jsonl"
    hashes["novels"]=sha(path)
    assert hashes["novels"]==json.loads((BOOK_PUBLIC/"corpus.json").read_text())["splits"]["train"]["sha256"]
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if any(w in q["id"] for w in HELD for q in r["questions"]): continue
        add("novels",r["page"],[dict(id=q["id"],zh=q["zh"],en=q["en"],answer=q["refs"][0]) for q in r["questions"]],2)
    for name,file in (("cmrc","cmrc2018/cmrc2018_train.json"),("drcd","drcd/DRCD_training.json")):
        path=ROOT/"data/raw"/file
        hashes[name]=sha(path)
        for doc in json.loads(path.read_text())["data"]:
            for r in doc["paragraphs"]:
                if source_counts[name]>=4096: break
                add(name,r["context"],[dict(id=q["id"],zh=q["question"],answer=q["answers"][0]["text"]) for q in r["qas"] if q["answers"]],1)
    p=WORK/"pairs.json"
    record(p,rows)
    record(PUBLIC/"corpus.json",dict(files=hashes,pages=source_counts,question_pairs=Counter(r["source"] for r in rows),
                                     sha256=sha(p),bilingual_pairs=sum("en" in r["qs"][0] for r in rows)))


class OpticalTeacher:
    def __init__(self,bank):
        receipt=json.loads((TEACHER/"fit.json").read_text())["weights"]
        assert all(sha(TEACHER/n)==v for n,v in receipt.items())
        self.net=FontInvariantReader.restore(TEACHER,"cuda").eval()
        self.emb=torch.cat([self.net.embed(b"".join(bank.cells[i:i+128])) for i in range(0,len(bank.cells),128)])
        self.source=receipt

    def score(self, pairs):
        windows=[]
        outputs=[torch.full((len(p),WIDTH),-torch.inf,device="cuda") for p,q in pairs]
        for i,(p,q) in enumerate(pairs):
            cap=512-len(q)-3
            if cap<WIDTH:raise ValueError("Question exceeds optical teacher window")
            for j in range(0,len(p),cap-WIDTH+1):
                windows.append((i,j,p[j:j+cap],q))
                if j+cap>=len(p):break
        for k in range(0,len(windows),12):
            group=windows[k:k+12];streams=[];roles=[]
            for _,_,p,q in group:
                m=self.net.markers
                streams.append(torch.cat([m[:1],self.emb[q],m[1:2],self.emb[p],m[1:2]]))
                roles.append(torch.tensor([0]*(len(q)+2)+[1]*(len(p)+1),device="cuda"))
            x=pad_sequence(streams,batch_first=True)
            length=torch.tensor([len(s) for s in streams],device="cuda")
            valid=torch.arange(x.shape[1],device="cuda")[None,:]<length[:,None]
            y=self.net.encoder(inputs_embeds=x,attention_mask=valid,token_type_ids=pad_sequence(roles,batch_first=True))
            for z,(i,j,p,q) in enumerate(group):
                lo=len(q)+2; hi=lo+len(p)
                a,b=y.start_logits[z,lo:hi],y.end_logits[z,lo:hi]
                right=torch.nn.functional.pad(b,(0,WIDTH-1),value=-torch.inf).unfold(0,WIDTH,1)
                local=a[:,None]+right
                old=outputs[i][j:j+len(p)]
                old.copy_(torch.maximum(old,local))
        return outputs


@torch.inference_mode()
def teach(split):
    face,bank=Face(FACES[0]),InkBank()
    path=WORK/"pairs.json" if split=="learn" else LAST/"reserved.json"
    spec=json.loads((PUBLIC/"corpus.json").read_text()) if split=="learn" else json.loads((PREVIOUS/"sources.json").read_text())["reserved"]
    assert sha(path)==spec["sha256"]
    rows=json.loads(path.read_text())
    prompts=[];refs=[]
    for r in rows:
        p=bank.encode(face(r["page"]))
        qs=r["qs"] if split=="learn" else r["questions"]
        for lang in (["zh"] if split=="learn" else ["zh","en"]):
            for q in qs:
                prompts.append((p,bank.encode(face(q[lang]))))
                refs.append(bank.encode(face(q["answer"] if split=="learn" else q["refs"][0])))
    net=OpticalTeacher(bank)
    first=snapshot();t=monotonic();indices=[];probs=[];other=[];flags=[];replies=[]
    for b in range(0,len(prompts),24):
        z=net.score(prompts[b:b+24])
        for j,score in enumerate(z):
            i=b+j
            if split=="learn":
                mass=(score.double().flatten()/2.).softmax(0)
                k=min(32,int((mass>0).sum()))
                v,idx=mass.topk(k)
                residual=float((1-v.sum()).clamp_min(0))
                indices.append(np.pad(idx.cpu().numpy(),(0,32-k)))
                probs.append(np.pad(v.float().cpu().numpy(),(0,32-k)))
                other.append(residual)
            else:
                best=best_strip(score.cpu().numpy(),group_spans(prompts[i][0]))
                flags.append(best==tuple(refs[i]))
                replies.append(b"".join(bank.cells[n] for n in best).hex())
    elapsed=monotonic()-t;energy=consumed(first,snapshot())
    if split=="learn":
        file=WORK/"teacher_targets.npz"
        np.savez_compressed(file,indices=np.asarray(indices,np.int32),prob=np.asarray(probs,np.float32),residual=np.asarray(other,np.float32))
        meta=dict(top32_mass=float(np.mean(1-np.array(other))))
    else:
        file=WORK/"teacher_reserved.json"
        record(file,replies)
        flags=np.array(flags).reshape(-1,2,2)
        meta=dict(individual=flags.sum(axis=(0,2)).tolist(),both=flags.all(2).sum(0).tolist())
    record(PUBLIC/f"teacher_{split}.json",dict(**meta,questions=len(prompts),weights=net.source,
                    seconds=elapsed,energy=energy,sha256=sha(file)))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("phase",choices=("prepare","learn","reserved"))
    phase=ap.parse_args().phase
    if phase=="prepare":prepare()
    else:
        with gpu_owner():
            torch.set_num_threads(2)
            teach(phase)

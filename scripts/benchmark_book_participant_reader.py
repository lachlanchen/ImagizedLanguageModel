"""Compare autonomous answers across two novels with no supplied answer candidates."""

import argparse
import json
from collections import Counter
from statistics import median
from time import monotonic
import torch

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.local_translation_teacher import local_teacher
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_book_participant_questions import PUBLIC, WORK
from scripts.read_book_participant import answer, restore
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def samples():
    f=WORK/"reserved.json"
    assert sha(f)==json.loads((PUBLIC/"sources.json").read_text())["reserved"]["sha256"]
    for r in json.loads(f.read_text()):
        for lang in ("zh","en"):
            for q in r["questions"]:
                yield r,q,lang


def run(arm):
    rows=list(samples())
    face=Face(FACES[0])
    answers=[]
    times=[]
    model_sha=None
    if arm=="text":
        model="qwen3:8b-q8_0"
        with local_teacher(model,WORK) as (request,model_sha,_):
            system="根据摘录回答问题。只输出原文中指代这个人的最短完整词语，保留原文汉字，不要解释或加标点。"
            def ask(p,q):
                return request(system,dict(role="user",content=p+"\n"+q),
                               think=False,options=dict(temperature=0,seed=241127,num_ctx=2048,num_predict=30))
            ask("王伟借了书。","谁借了书？")
            energy=snapshot()
            for r,q,lang in rows:
                t=monotonic()
                result=ask(r["page"],q[lang])
                times.append(monotonic()-t)
                answers.append((result["text"],result["stop"]=="stop"))
            joules=consumed(energy,snapshot())
    else:
        prompts=[(face(r["page"]),face(q[lang])) for r,q,lang in rows]
        energy=None
        if arm=="previous":
            net=PreparedGlyphResponder(ROOT/"artifacts/action_binding_v1/clause_aligned/page","cuda")
            for i in range(0,len(rows),16):
                ys=net.answer_many([(tensor(p),tensor(q)) for p,q in prompts[i:i+16]])
                answers.extend((im.numpy().tobytes(),end) for im,end in ys)
            joules=None
        else:
            path=WORK/"aligned/reader.safetensors"
            model_sha=sha(path)
            net=restore(path,"cuda")
            answer(net,*prompts[0])
            energy=snapshot()
            for p,q in prompts:
                t=monotonic()
                answers.append((answer(net,p,q),True))
                times.append(monotonic()-t)
            joules=consumed(energy,snapshot())
    detail=[]
    counts={}
    for (r,q,lang),(y,stop) in zip(rows,answers,strict=True):
        gold=q["refs"][0]
        exact=bool(stop and (y==gold if arm=="text" else y==face(gold)))
        detail.append(dict(page=r["page_sha"],work=r["work"],lang=lang,id=q["id"],exact=exact,
                           answer=y if arm=="text" else y.hex(),ended=stop))
    for work in sorted({r["work"] for r in detail})+["all"]:
        for lang in ("zh","en"):
            subset=[r for r in detail if r["lang"]==lang and (work=="all" or r["work"]==work)]
            both=0
            for a,b in zip(subset[::2],subset[1::2],strict=True):
                assert a["page"]==b["page"];both+=a["exact"] and b["exact"]
            counts[work+"/"+lang]=dict(pages=len(subset)//2,questions=len(subset),both=both,
                                       exact=sum(r["exact"] for r in subset))
    path=WORK/f"benchmark_{arm}.json"
    record(path,detail)
    record(PUBLIC/f"benchmark_{arm}.json",dict(counts=counts,seconds_sum=sum(times) if times else None,
           median_seconds=median(times) if times else None,energy=joules,weights_sha=model_sha,sha256=sha(path)))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm",choices=("previous","vision","text"))
    a=ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        # Match the published CPU and interactive interface precision.
        run(a.arm)

"""Decode original and changed passages and preserve the emitted answer pixels."""

import argparse
from collections import defaultdict
from statistics import median
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.book_participant_reader import InkBank, group_spans, best_strip, padded
from scripts.local_translation_teacher import local_teacher
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_role_question_binding import LAST,PUBLIC,WORK,WEIGHTS,read
from scripts.prepare_book_participant_questions import PUBLIC as NOVEL_RECEIPTS
from scripts.read_book_participant import restore
from scripts.train_causal_glyph_writer import sha,record
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face,FACES


def populations():
    for name,folder,filename,digest in [
        ("classical",WORK,"classical.json",read(PUBLIC/"corpus.json")["classical_sha"]),
        ("novels",LAST,"reserved.json",read(NOVEL_RECEIPTS/"sources.json")["reserved"]["sha256"]),
        ("train",WORK,"learn.json",read(PUBLIC/"corpus.json")["learn_sha"]),
    ]:
        path=folder/filename
        assert sha(path)==digest
        rows=read(path)
        if name=="train":
            rows=[r for k in ("combined","natural") for r in [x for x in rows if x["kind"]==k][:24]]
        for i,r in enumerate(rows):
            if name=="novels":
                pages=[r["page"]];key=r["page_sha"]
                qs=[dict(**q,answer=a) for q,a in zip(r["questions"],r["names"],strict=True)]
            elif name=="classical":
                pages=r["pages"];key=r["key"];qs=r["qs"]
            else:
                pages=["。".join(r["parts"])];key=f"{r['kind']}-{i}";qs=r["qs"]
            for w,p in enumerate(pages):
                for lang in ("zh","en"):
                    for h in (0,1):
                        yield dict(group=name,key=key,world=w,lang=lang,h=h,page=p,question=qs[h][lang],
                                   answer=qs[h^w]["answer"],other=qs[1^h^w]["answer"])


def scores(detail):
    result={}
    for group in dict.fromkeys(r["group"] for r in detail):
        for l in ("zh","en"):
            rows=[r for r in detail if r["group"]==group and r["lang"]==l]
            scenes=defaultdict(lambda:defaultdict(list))
            for r in rows:scenes[r["key"]][r["world"]].append(r)
            pairs=[pair for scene in scenes.values() for pair in scene.values()]
            assert all(len(pair)==2 for pair in pairs)
            result[group+"/"+l]=dict(questions=len(rows),individual=sum(r["correct"] for r in rows),
                pairs=len(pairs),both=sum(all(r["correct"] for r in pair) for pair in pairs),
                scenes=len(scenes),whole=sum(all(r["correct"] for pair in scene.values() for r in pair) for scene in scenes.values()),
                repeats=sum(pair[0]["reply"]==pair[1]["reply"] for pair in pairs),rival=sum(r["rival"] for r in rows))
    return result


@torch.inference_mode()
def decode(arm,items,work,file=None,expected=None,measure_margin=False):
    face=Face(FACES[0]);answers=[];times=[];margins=[]
    if arm=="text":
        # Use the same text model and decoding prompt as the earlier novel reference.
        with local_teacher("qwen3:8b-q8_0",work) as (request,weights,_):
            def ask(p,q):
                return request("根据摘录回答问题。只输出原文中指代这个人的最短完整词语，保留原文汉字，不要解释或加标点。",
                    dict(role="user",content=p+"\n"+q),think=False,options=dict(temperature=0,seed=241127,num_ctx=2048,num_predict=30))
            ask("王伟借了书。","谁借了书？")
            first=snapshot()
            for item in items:
                t=monotonic();out=ask(item["page"],item["question"]);times.append(monotonic()-t)
                answers.append((out["text"],out["stop"]=="stop"))
            energy=consumed(first,snapshot())
    else:
        bank=InkBank()
        weights=sha(file)
        assert weights==expected
        net=restore(file,"cuda")
        pages=[bank.encode(face(r["page"])) for r in items]
        qs=[bank.encode(face(("誰？" if r["lang"]=="zh" else "Who?") if arm=="without_question" else r["question"])) for r in items]
        pixels=bank.raster("cuda")
        first=snapshot()
        last_page=groups=None
        for i in range(0,len(items),24):
            t=monotonic();ps=pages[i:i+24];z=net(pixels,padded(ps,"cuda"),padded(qs[i:i+24],"cuda")).cpu().numpy()
            for j,(p,logits) in enumerate(zip(ps,z,strict=True)):
                if last_page!=p:
                    groups=group_spans(p);last_page=p
                emitted=best_strip(logits[:len(p)],groups)
                if measure_margin:
                    item=items[i+j]
                    flat=logits[:len(p)].flatten().astype(np.float64)
                    values=[np.logaddexp.reduce(flat[groups[tuple(bank.encode(face(item[k]))) ]]) for k in ("answer","other")]
                    margins.append(float(values[0]-values[1]))
                answers.append((b"".join(bank.cells[k] for k in emitted).hex(),True))
            times.append(monotonic()-t)
        energy=consumed(first,snapshot())
    detail=[]
    for item,(reply,ended) in zip(items,answers,strict=True):
        physical=arm!="text"
        def label(a):return face(a).hex() if physical else a
        detail.append(dict(**{k:item[k] for k in ("group","key","world","lang","h")},reply=reply,ended=ended,
                           correct=ended and reply==label(item["answer"]),rival=ended and reply==label(item["other"])))
    if margins:
        for r,m in zip(detail,margins,strict=True):
            r["source_margin"]=m
    cost=dict(weights=weights,energy=energy,elapsed_batches=sum(times),median_batch_seconds=median(times),maximum_batch=1 if arm=="text" else 24)
    return detail,cost


def evaluate(arm):
    items=list(populations())
    if arm=="text":items=[x for x in items if x["group"]=="classical"]
    model_arm="aligned" if arm=="without_question" else arm
    file=WEIGHTS if arm=="original" else WORK/model_arm/"reader.safetensors"
    expected=None if arm=="text" else (read(PUBLIC/"corpus.json")["parent_weights"] if arm=="original" else read(PUBLIC/(model_arm+"_fit.json"))["weights"])
    detail,cost=decode(arm,items,WORK,file,expected)
    path=WORK/(arm+"_responses.json");record(path,detail)
    record(PUBLIC/(arm+"_scores.json"),dict(counts=scores(detail),response_sha=sha(path),**cost))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("arm",choices=("aligned","human","original","without_question","text"))
    args=ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        evaluate(args.arm)

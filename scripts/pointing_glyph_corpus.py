"""Natural contrast pairs and pictured historical captions for causal writing."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from scripts.fit_font_invariant_reader import Face, FONTS, ROOT, sha
from scripts.ordered_image_corpus import PAIRS, PLATES, STAGES
from scripts.pictured_participant_scenes import fresh_cases
from scripts.read_paired_passage_questions import PairedPassages
from scripts.train_causal_glyph_writer import examples, OPEN


@dataclass
class Row:
    task: str
    key: str
    p: bytes
    q: bytes
    y: bytes
    foil: bytes


def natural_pair(task, key, p, questions, answers):
    return [Row(task, f"{key}/{i}", p, questions[i], answers[i], questions[1-i]) for i in range(2)]


def eligible(p, qs, ys):
    return (len(p)//128 + 2*max(map(len,qs))//128 <= 506 and
            all(0 < len(y)//128 <= 48 for y in ys) and qs[0]!=qs[1] and
            ys[0] not in ys[1] and ys[1] not in ys[0])


def cmrc(split):
    data=PairedPassages(PAIRS,split)
    pairs=data.pairs[::2]
    rng=np.random.default_rng(15321)
    seen=set()
    rows=[]
    for j in rng.permutation(len(pairs)):
        g,a,b=map(int,pairs[j])
        if g in seen: continue
        p=data.passages[g]
        qs=[data.questions[a],data.questions[b]]
        ys=[data.answers[a],data.answers[b]]
        if eligible(p,qs,ys):
            rows.extend(natural_pair("cmrc",str(g),p,qs,ys))
            seen.add(g)
        if split!="train" and len(seen)==128: break
    if split!="train" and len(seen)!=128:raise ValueError("Insufficient distinct CMRC passages")
    return rows


def drcd(face, used):
    path=ROOT/"data/raw/drcd/DRCD_test.json"
    articles=json.loads(path.read_text())["data"]
    pars=[p for art in articles for p in art["paragraphs"]]
    rows=[]
    for i in np.random.default_rng(15322).permutation(len(pars)):
        par=pars[i]
        if len(par["context"])>500:continue
        try:p=face(par["context"])
        except ValueError:continue
        if p in used:continue
        qs=[];ys=[]
        for question in par["qas"]:
            answer=question["answers"][0]["text"]
            if answer not in par["context"]:continue
            try:q,y=face(question["question"]),face(answer)
            except ValueError:continue
            if qs and not eligible(p,[qs[0],q],[ys[0],y]):continue
            qs.append(q);ys.append(y)
            if len(qs)==2:break
        if len(qs)==2 and eligible(p,qs,ys):
            used.add(p)
            rows.extend(natural_pair("drcd",str(par["id"]),p,qs,ys))
        if len(rows)==256:break
    assert len(rows)==256
    return rows


def plates(split, face):
    receipt=json.loads((PLATES/"report.json").read_text())
    assert not any(receipt["overlap"].values())
    src=PLATES/f"{split}_cards.json"
    assert sha(src)==receipt["splits"][split]["cards_sha256"] and sha(PLATES/"pixels.npy")==receipt["pixels_sha256"]
    cards=json.loads(src.read_text())
    bank=np.load(PLATES/"pixels.npy")
    held=split!="train"
    chosen=list(np.random.default_rng(15323+int(held)).permutation(len(cards))[:48 if held else 512])
    if held:chosen.append("yan")
    rows=[]
    for k in chosen:
        card=json.loads((PLATES/"yan/source.json").read_text())["rows"] if k=="yan" else cards[k]
        mapping={(r['char'],r['stage']):bytes(bank[r['tile']]) for r in card}
        names=sorted({c for c,s in mapping})
        stages=sorted({s for c,s in mapping},key=list(STAGES).index)
        if len(names)!=2 or len(stages)!=2:raise ValueError("Expected two names and two stages")
        p=b""
        for c in names:
            p+=face(c+"：")
            for s in stages:p+=face(STAGES[s])+mapping[c,s]
            p+=face("。")
        for reword in ([False,True] if held and k!="yan" else [False]):
            task=("yan" if k=="yan" else ("historical_rephrased" if reword else "historical"))
            qs=[];ys=[]
            for c in names:
                for order in (stages,stages[::-1]):
                    s,t=map(STAGES.get,order)
                    q=(f"先列{s}，再列{t}。要查的是「{c}」，各寫出名稱和字形。"
                       if reword else f"請介紹「{c}」的字形，依{s}、{t}的順序作答。")
                    qs.append(face(q))
                    ys.append(face(f"{c}的{s}為")+mapping[c,order[0]]+face(f"，{t}為")+mapping[c,order[1]]+face("。"))
            for j in range(4):
                # Three cyclic offsets across cards prevent a globally invertible foil convention.
                offset=1+(int(k) if k!="yan" else 0)%3
                rows.append(Row(task,f"{k}/{j}",p,qs[j],ys[j],qs[(j+offset)%4]))
    return rows


def load():
    face=Face((FONTS[0],2))
    train=cmrc("train")+plates("train",face)
    dev=cmrc("development")+plates("development",face)
    kinds=json.loads((OPEN/"development/sources.json").read_text())
    for split,rows in (("train",train),("development",dev)):
        for i,(p,q,y) in enumerate(examples(split)):
            kind="open" if split=="train" else kinds[i]["kind"]
            rows.append(Row(kind,str(i),p,q,y,q))
    used={r.p for r in train}
    assert not used.intersection(r.p for r in dev if r.task!="yan")
    dev+=drcd(face,used|{r.p for r in dev})
    for i,(p,q,y) in enumerate(fresh_cases()):
        dev.append(Row("role",str(i),face(p),face(q),face(y),face(q)))
    return train,dev


def digest(rows):
    h=hashlib.sha256()
    for r in rows:
        for field in (r.task.encode(),r.key.encode(),r.p,r.q,r.y,r.foil):
            h.update(len(field).to_bytes(8,"big"));h.update(field)
    return h.hexdigest()

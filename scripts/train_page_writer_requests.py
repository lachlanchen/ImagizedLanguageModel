"""Transfer detail writing across pictured question formulations with fixed answer supervision."""
import argparse
import json
from pathlib import Path
import random

import torch
from scripts.grounded_proposition_corpus import ARCHIVE, ARCHIVE_HASH, corpora, noun
from scripts.page_evidence_corpus import DEFAULT, SEED, dual_cases, fits, save_split, single_cases, unique_year
from scripts.prepare_visual_facts import Face,FONTS,ROOT,WORK
from scripts.span_reservation import normalized,reservation
from scripts.train_page_evidence_writer import BASE,fit
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_span_choice import gpu_owner

VARIED = ROOT/"artifacts/page_writer_requests_v1"
TRAIN = (
    "先说明{}，再写同段提到的年份。",
    "请用两句话介绍{}和这段资料中的年份。",
    "回答{}是谁。接着交代相关段落里写了哪一年。",
    "关于{}，请写明其身份，再告诉我同一段提及的年份。",
    "除了介绍{}，还要写明该段记载的是哪一年。",
    "先说{}，然后指出有关文字里的年份。",
    "找出{}和那一段所提的年份，并用完整句子回答。",
    "这段资料给出了{}和哪一年的记载？请分别说明。",
    "请说明{}是谁，并补充在相应段落看到的年份。",
    "写下{}的名字，再指出那段文字提到的是哪一年。",
    "能否把{}与同段出现的年份都交代清楚？",
    "请用一句话介绍{}，再用一句话写出相关年份。",
)
NEW = (
    "读完材料，请交代{}以及相应段落中的年份。",
    "关于{}，资料里怎么说？还请注明那段文字提及的年份。",
    "我想知道{}是谁，同一段写到的是哪一年？请完整作答。",
)


def load_records(folder,name):
    receipt=json.loads((folder/"corpus.json").read_text())[name]
    f=folder/(name+".jsonl")
    if sha(f)!=receipt["sha256"]:
        raise ValueError("Source records changed")
    return list(map(json.loads,f.read_text().splitlines()))


def fresh_cases(face):
    guard=reservation()
    names,pages=set(),set()
    for split in ("training","held"):
        for line in (WORK/(split+".jsonl")).read_text().splitlines():
            v=json.loads(line)
            pages.add(normalized(v["p"]))
            names.add(normalized(v["s"]))
    old=json.loads((ROOT/"artifacts/grounded_proposition_v1/boards.json").read_text())
    for boards in old.values():
        for b in boards:
            pages.update(normalized(p) for p in b["p"].split("　"))
            names.update(normalized(f[0]) for f in b["facts"])
    windows={p[i:i+64] for p in pages for i in range(len(p)-63)}
    if sha(ARCHIVE)!=ARCHIVE_HASH:
        raise ValueError("DuIE source changed")
    ds=corpora("dev")
    random.Random(SEED+89).shuffle(ds)
    candidates=[]
    for key,p,triples in ds:
        norm=normalized(p)
        if (p.count("年")==0 or norm in pages or any(norm[i:i+64] in windows for i in range(len(norm)-63))
            or guard.overlaps(p) or any(not c.isspace() and ord(c) not in face.coverage for c in p)):
            continue
        for s,r,o in triples:
            v=dict(group="duie",key=str(key),p=p,q="",s=s,r=r,o=o)
            if normalized(s) in names or not unique_year(v) or p.count(s)!=1:
                continue
            if not all(fits(dict(p=p,q=q.format(noun(s,r)),y="")) for q in NEW):
                continue
            candidates.append(v)
            pages.add(norm)
            windows.update(norm[i:i+64] for i in range(len(norm)-63))
            names.add(normalized(s))
            break
        if len(candidates)>=260:
            break
    result=[]
    while candidates and len(result)<128:
        v=candidates.pop()
        g=single_cases(v,True)
        if g and all(map(fits,g)):
            result.extend(g)
    dual=[]
    while len(candidates)>1 and len(dual)<192:
        a=candidates.pop()
        for i in range(len(candidates)-1,-1,-1):
            g=dual_cases(a,candidates[i],True)
            if g and all(fits(dict(x,q=q.format(noun(x["s"],x["r"])))) for x in g for q in NEW):
                dual.extend(g)
                candidates.pop(i)
                break
    if len(result)!=128 or len(dual)!=192:
        raise ValueError("Insufficient new single- or two-paragraph cases")
    return result+dual


def train(parent,folder):
    torch.set_num_threads(2)
    probe=json.loads((parent/"familiar/results.json").read_text())["groups"]["single"]
    if 2*probe["complete"]<probe["boards"]:
        raise RuntimeError("The familiar instruction criterion failed")
    folder.mkdir(exist_ok=False,parents=True)
    original=load_records(parent,"training")
    held=load_records(parent,"held")
    rng=random.Random(SEED+1)
    data=[]
    for x in original:
        q=x["q"]
        if x["task"] in ("single","dual"):
            q=rng.choice(TRAIN).format(noun(x["s"],x["r"]))
        data.append(dict(x,q=q))
    assert not set(TRAIN)&set(NEW)
    assert all(fits(x) for x in data)
    face=Face((FONTS[0],2))
    fresh=fresh_cases(face)
    corpus=json.loads((parent/"corpus.json").read_text())
    if any(sha(BASE/n)!=h for n,h in corpus["base"].items()):
        raise ValueError("Changed base writer")
    corpus.update(training=save_split(folder,"training",data),held=save_split(folder,"held",held),
                  fresh=save_split(folder,"fresh",fresh), original_training=sha(parent/"training.jsonl"))
    record(folder/"corpus.json",corpus)
    print(corpus["fresh"],flush=True)
    fit(folder,"page",data,face,order_records=original)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run",type=Path,default=DEFAULT)
    ap.add_argument("--run-dir",type=Path,default=VARIED)
    args=ap.parse_args()
    with gpu_owner():
        train(args.base_run,args.run_dir)

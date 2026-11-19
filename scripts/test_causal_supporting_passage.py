"""Compare question-only passage selection with inserting known supporting evidence."""
from collections import Counter, defaultdict
import fcntl
import json
import math
import os
from zipfile import ZipFile

import torch
from scripts.evaluate_causal_glyph_writer import evaluate
from scripts.prepare_composed_chinese_answers import ARCHIVE, normalized
from scripts.train_causal_glyph_writer import CausalGlyphWriter, Face, FONTS, OPEN, PUBLIC, ROOT, WORK, examples, record


def terms(s):
    s=normalized(s)
    return [*s,*(s[i:i+2] for i in range(len(s)-1))]


def select_window(paragraphs, question):
    parts=list(dict.fromkeys(p[j:j+224] for p in paragraphs for j in range(0,len(p),160) if p[j:j+224]))
    docs=[Counter(terms(s)) for s in parts]
    q=set(terms(question))
    freq=Counter(t for d in docs for t in q if t in d)
    sizes=[sum(d.values()) for d in docs]
    mean=max(1,sum(sizes)/len(docs))
    scores=[]
    for d,length in zip(docs,sizes,strict=True):
        k=1.2*(.25+.75*length/mean)
        scores.append(sum((2 if len(t)==2 else 1)*math.log1p((len(docs)-freq[t]+.5)/(freq[t]+.5))
                          *d[t]*2.2/(d[t]+k) for t in q if t in d))
    return parts[max(range(len(parts)),key=scores.__getitem__)]


def compare():
    readings=json.loads((WORK/"private_readings.json").read_text())
    entries=json.loads((OPEN/"development/sources.json").read_text())
    base=examples("development")
    lookup=defaultdict(dict)
    for i,r in enumerate(entries):
        if r["kind"]=="absent_content_ink":
            lookup[r["member"]][r["line"]]=i
    selected={}
    with ZipFile(ARCHIVE) as z:
        for member,items in lookup.items():
            with z.open(member) as f:
                for k,raw in enumerate(f):
                    if k>max(items):break
                    if k not in items:continue
                    i=items[k]; r=readings[i]; item=json.loads(raw)
                    assert str(item["question_id"])==entries[i]["question_id"]
                    docs=[p for d in item["documents"] for p in d["paragraphs"] if p]
                    target=r["reference"]
                    supporting=next((p for p in docs if target in p),None)
                    if supporting is None:continue
                    center=supporting.index(target)+len(target)//2
                    start=min(max(0,len(supporting)-224),max(0,center-112))
                    oracle=supporting[start:start+224]
                    assert target in oracle
                    selected[i]=(select_window(docs,r["question"]),oracle)
    assert len(selected)==51
    source_ids=sorted(selected)
    w=CausalGlyphWriter.restore(WORK/"matched","cuda")
    render=Face((FONTS[0],2))
    original=json.loads((WORK/"matched_open.json").read_text())
    stats={"cases":len(selected),"original_complete":sum(original["rows"][i]["exact"] for i in source_ids)}
    for column,key in enumerate(("bm25","reference_assisted")):
        records=[]; fallback=visible=0
        for i in source_ids:
            p,q,y=base[i]
            try:new=render(selected[i][column])
            except ValueError:new=p;fallback+=1
            visible+=any(new[k:k+len(y)]==y for k in range(0,len(new),128))
            records.append((new,q,y))
        stem="support_"+key
        measured=evaluate(w,records,["support"]*len(records),stem)
        outputs=json.loads((WORK/(stem+".json")).read_text())
        changed=sum((original["outputs"][i],original["rows"][i]["stopped"])!=(o,r["stopped"])
                    for i,o,r in zip(source_ids,outputs["outputs"],outputs["rows"],strict=True))
        stats[key]={"reference_visible":visible,"font_fallback":fallback,"changed":changed,"result":measured}
    record(PUBLIC/"supporting_passages.json",stats)
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        compare()

"""Independently recount the held scores using hashed per-stream public receipts."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import string

from scripts.dialogue_ink_data import read_data
from scripts.prepare_dialogue_ink import dump, sha


def normalize(s):
    words=re.sub(r"\b(?:a|an|the)\b"," ",s.lower().translate(str.maketrans("","",string.punctuation))).split()
    return words


def recount(run,public):
    source=read_data()
    records=json.loads(gzip.decompress((run/"decisions.json.gz").read_bytes()))
    groups={g:[r for r in source["rows"] if r[0]==g] for g in ("dev","reserved")}
    seen={tuple(normalize(r[5])) for r in source["rows"] if r[0]=="train"}
    totals=Counter()
    compact=[]
    for r in records:
        j,g=r["index"],r["group"]
        row=groups[g][j]
        answer=row[5]
        words=normalize(answer)
        assert (row[1],row[2],row[3],row[4],answer)==(r["story"],r["turn"],r["question"],r["history"],r["answer"])
        fields=(source["pages"][row[1]],row[3],row[4])
        missing=len(answer.split())>=2 and not any(answer.casefold() in s.casefold() for s in fields)
        strict=missing and tuple(words) not in seen
        assert (r["strict"],r["absent_multiword"])==(strict,missing)
        assert groups[g][r["passage_control"]][1]!=row[1] and groups[g][r["question_control"]][1]==row[1]
        arms={}
        for arm,d in r["out"].items():
            prediction=normalize(d["text"])
            px=d["stopped"] and d["text"]==answer
            hit=d["stopped"] and prediction==words
            overlap=sum((Counter(prediction)&Counter(words)).values())
            f1=2*overlap/(len(prediction)+len(words)) if d["stopped"] and overlap else 0.
            assert (px,hit)==(d["pixels"],d["exact"]) and abs(f1-d["f1"])<1e-9
            for key,valid in (("all",True),("strict",strict),("first",row[2]==0),("binary",r["binary"]),("absent_multiword",missing),("strict_exchangeable",strict and r["question_control"]!=j)):
                if valid:
                    for metric,x in (("exact",hit),("pixels",px),("stopped",d["stopped"]),("f1",f1),("n",1)):
                        totals[g,key,arm,metric]+=x
            arms[arm]=[hashlib.sha256(d["text"].encode()).hexdigest(),d["stopped"],px,hit,len(d["actions"])]
        compact.append([g,j,row[1],row[2],strict,r["question_control"],r["passage_control"],arms])
    scores=json.loads((run/"scores.json").read_text())
    for g,stats in scores.items():
        for key,v in stats.items():
            for arm,e in v["arms"].items():
                for metric in ("exact","pixels","stopped"):
                    assert e[metric]==totals[g,key,arm,metric]
                assert v["n"]==totals[g,key,arm,"n"]
                assert abs(e["mean_f1"]-totals[g,key,arm,"f1"]/v["n"])<1e-9
    assert len(records)==len({(r["group"],r["index"]) for r in records})==sum(map(len,groups.values()))
    public.mkdir(exist_ok=True,parents=True)
    receipts=public/"decisions.json.gz"
    receipts.write_bytes(gzip.compress(json.dumps(compact,separators=(",",":")).encode(),mtime=0))
    dump(public/"audit.json",dict(verified_questions=len(records),verified_outputs=4*len(records),
                                 raw_sha256=sha(run/"decisions.json.gz"),receipts_sha256=sha(receipts)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("public",type=Path)
    a=p.parse_args();recount(a.run,a.public)

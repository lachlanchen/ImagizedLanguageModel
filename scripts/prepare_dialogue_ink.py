"""Reserve complete CoQA passages and locate training-only supporting evidence."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from scripts.prepare_optical_passage_qa import ROOT, sha, dump, printable

WORK = ROOT / "artifacts/dialogue_ink_v1"
PUB = ROOT / "references/evidence/dialogue_ink_v1"
RELEASE = ROOT / "models/dialogue_ink"
SOURCES = {
    "train":"4272ee344ef27112c50237a1ae3b1c90e65e11a101d6874b2f57e4c08d147135",
    "validation":"d66f4811dd137818923a3db7a629ce3cf9e73977b5e2541471311c1566bf349f",
}


def clean(s):
    return " ".join(s.split())


def prepare():
    pages, rows = {}, []
    raw, reject = Counter(), Counter()
    seen = set()
    for split, digest in SOURCES.items():
        path = ROOT / "data/raw/coqa" / (split+".parquet")
        if sha(path)!=digest:
            raise ValueError("Unexpected CoQA release")
        for sample in pq.read_table(path).to_pylist():
            story = sample["story"]
            page = clean(story)
            key = hashlib.sha256(page.encode()).hexdigest()
            group = "dev" if split=="validation" else ("reserved" if int(key,16)%17==0 else "train")
            qa = sample["answers"]
            words = list(re.finditer(r"\S+",story))
            ok = printable(page) and 1<=len(words)<=320
            last = ""
            for turn,(q,a,first,end) in enumerate(zip(sample["questions"],qa["input_text"],qa["answer_start"],qa["answer_end"],strict=True)):
                raw[group]+=1
                question, answer = clean(q),clean(a)
                history = " ".join(last.split()[-48:])
                last = question+" "+answer
                if not (ok and printable(history) and printable(question) and printable(answer)
                        and 1<=len(question.split())<=32 and answer):
                    reject[group+"_geometry"]+=1
                    continue
                if group=="train" and (len(answer.split())>16 or max(map(len,answer.split()))>24):
                    reject["train_answer"]+=1
                    continue
                identity=(group,key,turn,question,history,answer)
                if identity in seen:
                    reject[group+"_duplicate"]+=1
                    continue
                seen.add(identity)
                cover=[i for i,w in enumerate(words) if 0<=first<end<=len(story) and w.start()<end and first<w.end()]
                rows.append([group,key,turn,question,history,answer,
                             [cover[0],cover[-1]] if cover else [-1,-1],sample["source"]])
                pages[key]=page
    train={r[1] for r in rows if r[0]=="train"}
    keep=[]
    for r in rows:
        if r[0]!="train" and r[1] in train:
            reject[r[0]+"_overlap"]+=1
        else:
            keep.append(r)
    WORK.mkdir(exist_ok=True,parents=True)
    file=WORK/"corpus.json"
    file.write_text(json.dumps(dict(pages=pages,rows=keep),separators=(",",":")))
    record=dict(raw=raw,excluded=reject,eligible=Counter(r[0] for r in keep),
                passages={g:len({r[1] for r in keep if r[0]==g}) for g in raw},
                sha256=sha(file),source=SOURCES)
    dump(PUB/"split.json",record)
    print(record)


if __name__=="__main__":
    prepare()

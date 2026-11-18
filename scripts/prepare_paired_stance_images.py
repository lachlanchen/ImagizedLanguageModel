"""Prepare local real questions with two documented, opposing answers in pictured source passages."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

import numpy as np

from scripts.measure_natural_stance_evidence import ARCHIVE, PUBLIC, WORK
from scripts.prepare_composed_chinese_answers import normalized
from scripts.prepare_pictured_entry_answers import Rasterizer, save_sequences, sha, write_json
from scripts.train_english_request_writer import ROOT

DEST=ROOT/"data/paired_stance_images_v1"


def source_window(paragraph,answer):
    index=paragraph.find(answer)
    if index<0:return None
    start=max(0,min(index-(192-len(answer))//2,len(paragraph)-192))
    return paragraph[start:start+192], start


def prepare():
    report=json.loads((PUBLIC/"natural_stance_feasibility.json").read_text())
    index=WORK/"candidates.jsonl"
    if report["candidate_sha256"]!=sha(index) or report["source_sha256"]!=sha(ARCHIVE):raise ValueError("Changed source")
    if DEST.exists():raise FileExistsError(DEST)
    rows=[json.loads(l) for l in index.read_text().splitlines()]
    raster=Rasterizer()
    reserved_q,reserved_runs=set(),set()
    totals={}
    with TemporaryDirectory(dir=DEST.parent) as temp,ZipFile(ARCHIVE) as z:
        out=Path(temp)/"corpus";out.mkdir()
        for split,tag in (("development","devset"),("train","trainset")):
            dest=out/split;dest.mkdir()
            bank={}
            collection={key:[] for key in ("questions","passages","answers")}
            detail=[]
            counts=Counter()
            def encode(s):
                return [bank.setdefault(raster.tile(c),len(bank)) for c in s]
            for member in sorted({r["member"] for r in rows if tag in r["member"]}):
                lookup={r["line"]:r for r in rows if r["member"]==member}
                with z.open(member) as f:
                    for i,line in enumerate(f):
                        if i not in lookup:continue
                        r=json.loads(line);idx=lookup[i]
                        q=r["question"].strip()
                        if sha256(normalized(q).encode()).hexdigest()!=idx["question_hash"]:raise ValueError("Wrong question")
                        if normalized(q) in reserved_q:continue
                        examples=[]
                        for side,key,prefix in (("Yes","yes","肯定。"),("No","no","否定。")):
                            d,k,j=idx[key]
                            a=r["answers"][j].strip()
                            assert r["yesno_answers"][j]==side and r["documents"][d]["is_selected"]
                            loc=source_window(r["documents"][d]["paragraphs"][k],a)
                            if not loc or len(prefix+a)>64:break
                            p,start=loc
                            flat=normalized(p)
                            if any(flat[t:t+32] in reserved_runs for t in range(len(flat)-31)):break
                            examples.append(dict(q=q,p=p,a=prefix+a,body=a,doc=d,paragraph=k,annotation=j,start=start,side=side))
                        if len(examples)!=2:continue
                        a,b=examples
                        if normalized(a["body"]) in normalized(b["p"]) or normalized(b["body"]) in normalized(a["p"]):
                            continue
                        try:
                            all_inputs=[(encode(x["q"]),encode(x["p"]),encode(x["a"])) for x in examples]
                        except ValueError:
                            continue
                        for x,streams in zip(examples,all_inputs):
                            qids,pids,yids=streams
                            for name,seq in zip(collection,streams,strict=True):
                                collection[name].append(seq)
                            counts["absent_stance_content"]+=int(any(g not in set(qids+pids) for g in yids[:2]))
                            detail.append(dict(member=member,line=i,id=r["question_id"],**{k:x[k] for k in ("doc","paragraph","annotation","start","side")}))
                            if split=="development":
                                s=normalized(x["p"])
                                reserved_runs.update(s[t:t+32] for t in range(len(s)-31))
                        reserved_q.add(normalized(q))
            for name,seqs in collection.items():
                save_sequences(dest/(name+".npz"),seqs)
            assert len(detail)==len(collection["answers"]) and len(detail)%2==0
            np.save(dest/"pixels.npy",np.frombuffer(b"".join(bank),np.uint8).reshape(-1,128))
            write_json(dest/"rows.json",detail)
            totals[split]=dict(pairs=len(detail)//2,answers=len(detail),counts=counts,files={p.name:sha(p) for p in dest.iterdir()})
        for name in ("README.md","License.docx"):(out/name).write_bytes(z.read("raw/"+name))
        result=dict(splits=totals,archive_sha256=report["source_sha256"],source_sha256=sha(__file__),
                    selection="Selected-document paragraphs containing the original human answer literally; different passages for opposite stances. Windows are reference-assisted and fixed offline.")
        write_json(out/"preparation.json",result)
        out.rename(DEST)
    write_json(PUBLIC/"natural_stance_preparation.json",result)
    print(totals)


if __name__=="__main__":
    prepare()

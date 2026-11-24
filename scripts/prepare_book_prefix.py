"""Reserve classical Chinese works, render glyphs and keep chapter boundaries."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from time import monotonic
import numpy as np
from scripts.prepare_dialogue_ink import ROOT,sha,dump
from scripts.fit_continuous_page_reader import Renderer
from ilm.visual_lm.portable_writer import DEFAULT_FONT

WORK=ROOT/"artifacts/book_prefix_v1"
PUB=ROOT/"references/evidence/book_prefix_v1"
MODEL=ROOT/"models/book_prefix"
SOURCE=ROOT/"data/raw/ud_classical_kyoto_r216"
TEST={"KR1h0004","KR4a0001"}
DEV={"KR1h0001"}


def prepare():
    clock=monotonic()
    chapters=defaultdict(list)
    seen=set()
    owners={}
    hashes={}
    for f in sorted(SOURCE.glob("*.conllu")):
        hashes[f.name]=sha(f)
        for line in f.open(encoding="utf8"):
            if line.startswith("# sent_id = "):
                sid=line.strip().split(" = ",1)[1]
                doc="_".join(sid.split("_")[:2])
            elif line.startswith("# text = "):
                if sid in seen or owners.setdefault(doc,f.name)!=f.name:
                    raise ValueError(f"Ambiguous ordering of {sid}")
                seen.add(sid)
                if not sid.endswith("_title"):
                    chapters[doc].append(line.rstrip("\r\n").split(" = ",1)[1])
    face=Renderer(DEFAULT_FONT)
    chars=sorted(set("".join("".join(v) for v in chapters.values())))
    rendered={c:face.cell(c) for c in chars if ord(c) in face.coverage and not c.isspace()}
    rendered={c:p for c,p in rendered.items() if any(p)}
    trainchars={c for d in chapters if d.split("_")[0] not in TEST|DEV for c in "".join(chapters[d]) if c in rendered}
    train=sorted({rendered[c] for c in trainchars})
    rest=sorted(set(rendered.values())-set(train))
    alphabet=[bytes(128)]+train+rest
    lookup={p:i for i,p in enumerate(alphabet)}
    rep={}
    for c in sorted(rendered,key=lambda c:(c not in trainchars,c)):
        rep.setdefault(lookup[rendered[c]],c)
    records=[]
    raw=[]
    missing=Counter()
    volumes=Counter()
    for doc,lines in sorted(chapters.items()):
        work=doc.split("_")[0]
        split="test" if work in TEST else "dev" if work in DEV else "train"
        fold=int.from_bytes(hashlib.sha256(doc.encode()).digest()[:4],"big")%4
        start=len(raw)
        for c in "".join(lines):
            if c in rendered:
                raw.append(lookup[rendered[c]])
            else:
                missing[c]+=1
                if len(raw)>start:
                    records.append([split,work,doc,fold,start,len(raw)])
                start=len(raw)
        if len(raw)>start:
            records.append([split,work,doc,fold,start,len(raw)])
        volumes[work]+=1
    if len(alphabet)>=65535:
        raise ValueError("Too many physical shapes for packed count keys")
    codes=np.zeros((len(alphabet),4),np.uint8)
    for i,c in rep.items():
        b=c.encode()
        codes[i,:len(b)]=list(b)
    WORK.mkdir(exist_ok=True,parents=True)
    PUB.mkdir(exist_ok=True,parents=True)
    packed=WORK/"corpus.npz"
    np.savez_compressed(packed,tiles=np.frombuffer(b"".join(alphabet),np.uint8).reshape(-1,128),
                        stream=np.array(raw,np.uint16),bytes=codes)
    meta=dict(source_hashes=hashes,font_sha256=sha(DEFAULT_FONT),license=sha(SOURCE/"LICENSE.txt"),
              counts={s:sum(b-a for t,_,_,_,a,b in records if t==s) for s in ("train","dev","test")},
              train_shapes=len(train),test_only_or_dev_shapes=len(rest),physical_aliases=len(rendered)-len(alphabet)+1,
              removed=dict(missing),volumes=volumes,held_work_ids=sorted(TEST),validation_work_ids=sorted(DEV),records=records,
              corpus=sha(packed))
    dest=PUB/"corpus.json"
    if dest.exists() and json.loads(dest.read_text())!=meta:
        raise ValueError("Prepared corpus differs")
    dump(dest,meta)
    dump(WORK/"representatives.json",{i:c for i,c in rep.items()})
    print(meta["counts"],"shapes",len(train),len(rest),"aliases",meta["physical_aliases"],"omitted",sum(missing.values()),"seconds",monotonic()-clock,flush=True)


def corpus():
    m=json.loads((PUB/"corpus.json").read_text())
    f=WORK/"corpus.npz"
    if sha(f)!=m["corpus"]:
        raise ValueError("Wrong prepared corpus")
    with np.load(f,allow_pickle=False) as src:
        return m,{k:src[k] for k in src}


if __name__=="__main__":
    prepare()

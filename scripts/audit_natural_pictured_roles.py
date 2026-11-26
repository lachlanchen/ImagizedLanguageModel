"""Verify unseen UD annotations, exact source-image overlap and fixed-geometry reversals."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.prepare_natural_pictured_roles import RAW,RUN,PUB,MODELS,ROOT,REVS,read,dump,sha
from scripts.read_natural_roles import choose,load,strip_png
from scripts.test_pictured_order_voice import PARTICIPLE


def check_gold(rows):
    maps={}
    for corpus in REVS:
        path=RAW/corpus/f"en_{corpus.lower()}-ud-test.conllu"
        for block in path.read_text().split("\n\n"):
            sent=next((s[12:] for s in block.splitlines() if s.startswith("# sent_id = ")),None)
            fields=[s.split("\t") for s in block.splitlines()]
            words={int(f[0]):f for f in fields if len(f)==10 and f[0].isdigit()}
            maps[corpus,sent]=words
    matched=0
    for r in rows:
        words=maps[r["corpus"],r["sentence"]]
        verb=r["text"][slice(*r["verb"])]
        pair=[r["text"][slice(*s)] for s in r["candidates"]]
        answer=pair[r["label"]],pair[1-r["label"]]
        grammatical=("nsubj","obj") if r["voice"]=="active" else ("obl:agent","nsubj:pass")
        found=[]
        for j,v in words.items():
            if v[1]!=verb or v[3]!="VERB":continue
            args=[[w[1] for w in words.values() if w[6]==str(j) and w[7]==rel] for rel in grammatical]
            if all(len(items)==1 for items in args):found.append(tuple(items[0] for items in args))
        assert answer in found
        matched+=1
    return matched


def main():
    torch.set_num_threads(2)
    codes=read(RUN/"codes.json");pal=np.load(MODELS/"palette.npy")
    rows=read(RUN/"test.json")
    def physical(sentence):
        return np.asarray([codes[c] for c in sentence],dtype="<u2").tobytes()
    assert len({img.tobytes() for img in pal[1:]})==len(pal)-1
    training={physical(r["text"]) for r in read(RUN/"train.json")}
    overlap=[i for i,r in enumerate(rows) if physical(r["text"]) in training]
    official=check_gold(rows)
    cases=read(ROOT/"references/evidence/independent_optical_actions_v1/role-reversal.json")["cases"]
    voice=ROOT/"publication/ilm-nature/evidence/strip_restoration_voice.zip"
    with ZipFile(voice) as z: permitted={r["i"] for r in json.loads(z.read("records.json"))}
    models={k:load(k) for k in ("layout","context")}
    counts=defaultdict(Counter);records=[]
    raw=ROOT/"publication/ilm-nature/evidence/natural_roles_constructions.zip"
    tmp=raw.with_suffix(".tmp")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for i in sorted(permitted):
            participle=PARTICIPLE[cases[i]["verb"]]
            for j,r in enumerate(cases[i:i+2]):
                a,finite,b=r["text"][0].rstrip(".").split()
                designs=[
                    ("aligned_active",f"{a} has {participle}    {b}.",participle,0),
                    ("aligned_passive",f"{a} was {participle} by {b}.",participle,1),
                    ("active",f"{a} has {participle} {b}.",participle,0),
                    ("subject_cleft",f"It is {a} who {finite} {b}.",finite,0),
                    ("object_cleft",f"It is {a} whom {b} {finite}.",finite,1),
                    ("by_distractor",f"By chance, {a} {finite} {b}.",finite,0)]
                aligned={}
                for name,text,verb,target in designs:
                    spans=[(k,k+len(w)) for w in (a,b) for k in [text.index(w)]]
                    vs=(text.index(verb),text.index(verb)+len(verb))
                    ink=pal[[codes[c] for c in text]]
                    chosen={}; scores={}
                    prefix=f"{i:03d}/{j}/{name}/"
                    z.writestr(prefix+"source.png",strip_png(ink))
                    for arm,model in models.items():
                        q,score=choose(model,ink,vs,spans,arm)
                        chosen[arm]=q;scores[arm]=score
                        counts[name].update({arm:int(q==target)})
                    counts[name]["n"]+=1
                    if name.startswith("aligned"):
                        aligned[name]=(len(text),vs,spans,scores["layout"])
                    for field,k in (("actor",chosen["context"]),("undergoer",1-chosen["context"])):
                        z.writestr(prefix+field+".png",strip_png(ink[slice(*spans[k])]))
                    records.append(dict(i=i,j=j,task=name,text=text,target=target,choices=chosen,verb=vs,
                                        candidates=spans,member=prefix+"source.png"))
                assert aligned["aligned_active"]==aligned["aligned_passive"]
        z.writestr("records.json",json.dumps(records))
    tmp.replace(raw)
    dump(PUB/"audit.json",dict(official_gold_matches=official,training_picture_overlap=overlap,counts=counts,
        geometry_matched=len(permitted)*2,archive=sha(raw),protocol=sha(ROOT/"references/natural_pictured_roles_audit.md")))
    print(official,len(overlap),dict(counts),flush=True)


if __name__=="__main__":main()

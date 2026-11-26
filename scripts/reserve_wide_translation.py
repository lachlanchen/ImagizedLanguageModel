"""Reserve one unseen FLORES devtest sentence per article before the capacity comparison."""
from functools import lru_cache
import numpy as np
from scripts.reserve_visual_prefix import BASE, ROOT, BOOK, PREVIOUS, WORK as PREFIX, Ink, read, dump, sha, key, rank, flores, FIRST, OLD, published, opus_source
from scripts.prepare_visual_rollins import training_rows

WORK=ROOT/"artifacts/wide_translation_v1"
PUB=ROOT/"references/evidence/wide_translation_v1"

def reserve():
    WORK.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    old=read(PREFIX/"test.json")+read(PREVIOUS/"fresh.json")+read(BOOK/"test.json")+read(BOOK/"dev.json")
    docs={r["doc"] for r in old}
    ink=Ink(np.load(BASE/"palette.npy"))
    vocab=[set(v) for v in read(BASE/"inventories.json")]
    norm=lru_cache(65536)(key)
    candidates=[]
    for split,domain,doc,pair in flores():
        if split!="test" or doc in docs or not all(2<=len(s)<=96 for s in pair):
            continue
        ids=[ink.encode(s) for s in pair]
        if all(set(g)<=v and not ink.bad.intersection(s) for g,v,s in zip(ids,vocab,pair,strict=True)):
            candidates.append(dict(domain=domain,doc=doc,text=pair,cells=ids))
    wanted=[{norm(r["text"][t]) for r in candidates} for t in (0,1)]
    seen=[set(),set()]
    def scan(t,s):
        if len(s)<=96 and (k:=norm(s)) in wanted[t]:
            seen[t].add(k)
    for r in old:
        for t,s in enumerate(r["text"]): scan(t,s)
    for name in ("train","dev","test","news-train","news-held"):
        for row in read(OLD/(name+".json")):
            for t,s in enumerate(row["text"]): scan(t,s)
    for pair in opus_source("train"):
        for t,s in enumerate(pair): scan(t,s)
    lookup=read(BASE/"readback.json")
    def decoded(t, seq):
        scan(t,"".join(lookup[str(int(j))] for j in seq if j))
    for t,lang in enumerate(("en","zh")):
        for row in published(lang,"train").values():
            for label in ("sentence1","sentence2"): scan(t,row[label])
    for r in read(FIRST/"contrasts.json"):
        for pair in r["cells"]:
            for t,seq in enumerate(pair): decoded(t,seq)
    for r in read(ROOT/"artifacts/translation_paraphrases_v1/examples.json"):
        for t in (0,1):
            for seq in [r["target"][t],*r["variants"][t]]: decoded(t,seq)
    for bank in [np.load(BOOK/"cells.npy",mmap_mode="r"),np.load(PREVIOUS/"replacement.npz")["cells"]]:
        for t in (0,1):
            for seq in bank[:,t]: decoded(t,seq[:np.count_nonzero(seq)])
    held=[]
    for r in sorted(candidates,key=lambda r:rank("width/"+repr(r["text"]))):
        ks=[norm(s) for s in r["text"]]
        if r["doc"] in docs or any(k in seen[t] for t,k in enumerate(ks)): continue
        docs.add(r["doc"]); held.append(r)
        for t,k in enumerate(ks): seen[t].add(k)
        if len(held)==32: break
    if len(held)<16: raise ValueError(("Too few unseen articles",len(held)))
    dump(WORK/"fresh.json",held)
    dump(PUB/"reservation.json",dict(articles=len(held),directions=2*len(held),sha256=sha(WORK/"fresh.json"),
        program=sha(ROOT/"scripts/reserve_wide_translation.py")))
if __name__=="__main__": reserve()

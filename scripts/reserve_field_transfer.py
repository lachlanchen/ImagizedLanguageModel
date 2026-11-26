"""Reserve bilingual passages beyond both previous literary panels and training exposures."""
from pathlib import Path
import numpy as np
from scripts.reserve_readout_prefixes import (
    BASE, WORK, PARENT, ROLE, PUB as PRIOR_PUB, RUN as PRIOR_RUN, ROOT, TITLES,
    Ink, candidates, dump, key, library, pool, previous, read, sha, word,
)

RUN=ROOT/"artifacts/field_to_glyph_v1"
PUB=ROOT/"references/evidence/field_to_glyph_v1"


def main():
    RUN.mkdir(exist_ok=True)
    PUB.mkdir(exist_ok=True,parents=True)
    target=RUN/"panel.json"
    receipt=PUB/"panel.json"
    if receipt.exists():
        if sha(target)!=read(receipt)["sha256"]: raise ValueError("Changed panel")
        return
    first=PARENT/"unseen-books.json"
    second=PRIOR_RUN/"fresh.json"
    if sha(first)!=read(ROLE/first.name)["rows_sha256"] or sha(second)!=read(PRIOR_PUB/second.name)["sha256"]:
        raise ValueError("Prior panel fingerprint")
    used=previous()+read(WORK/"held.json")+read(first)+read(second)
    for split in ("train","dev","test"): used.extend(library(split))
    norms=[{key(r["text"][t]) for r in used} for t in (0,1)]
    shapes=[{word(r["cells"][t]) for r in used} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            shapes[1-t].add(x)
            shapes[t].add(y)
    prior_counts=list(map(len,shapes))
    inventory=list(map(set,read(BASE/"inventories.json")))
    ink=Ink(np.load(BASE/"palette.npy"))
    picked=[]
    for title in TITLES:
        group=[]
        ordered,edition=candidates(title)
        if edition!=read(ROLE/first.name)["source_hashes"][title]: raise ValueError("Edition changed")
        for _,en,zh,chunk,para,u in ordered:
            text=[en,zh]
            keys=list(map(key,text))
            if any(not k or k in norms[t] for t,k in enumerate(keys)): continue
            cells=list(map(ink.encode,text))
            if any(not set(c)<=inventory[t] or ink.bad.intersection(text[t]) for t,c in enumerate(cells)): continue
            images=list(map(word,cells))
            if any(im in shapes[t] for t,im in enumerate(images)): continue
            for t in (0,1):
                norms[t].add(keys[t])
                shapes[t].add(images[t])
            group.append(dict(doc=title,domain="book",chunk=chunk,paragraph=para,unit=u,text=text,cells=cells))
            if len(group)==24: break
        if len(group)!=24: raise ValueError("Short book panel")
        picked.extend(group)
    dump(target,picked)
    dump(receipt,dict(sha256=sha(target),earlier=[sha(first),sha(second)],titles=TITLES,pairs=len(picked),
                      prior_bitmap_sides=prior_counts,program=sha(Path(__file__)),protocol=sha(ROOT/"references/field_to_glyph_protocol.md")))


if __name__=="__main__":
    main()

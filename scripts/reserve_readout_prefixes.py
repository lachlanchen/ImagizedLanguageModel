"""Reserve previously unseen bilingual requests for comparing visual state objectives."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.audit_teacher_neighbors import pool, word
from scripts.prepare_teacher_translation import BASE, WORK, Ink, key, previous
from scripts.prepare_library_roles import PUB as ROLE, RUN as PARENT
from scripts.reserve_library_role_books import TITLES, candidates
from scripts.sample_visual_library import ROOT, library, read, sha
from scripts.prepare_teacher_translation import dump

PUB = ROOT/"references/evidence/readout_objective_v1"
RUN = ROOT/"artifacts/readout_objective_v1"


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    target=RUN/"fresh.json"
    receipt=PUB/"fresh.json"
    if receipt.exists():
        if sha(target)!=read(receipt)["sha256"]: raise ValueError("Changed fresh panel")
        return
    prior=PARENT/"unseen-books.json"
    if sha(prior)!=read(ROLE/prior.name)["rows_sha256"]: raise ValueError("Earlier reservation differs")
    used=previous()+read(WORK/"held.json")+read(prior)
    for s in ("train","dev","test"):
        used.extend(library(s))
    texts=[{key(r["text"][t]) for r in used} for t in (0,1)]
    images=[{word(r["cells"][t]) for r in used} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            images[1-t].add(x)
            images[t].add(y)
    exclude=list(map(len,images))
    ink=Ink(np.load(BASE/"palette.npy"))
    allowed=list(map(set,read(BASE/"inventories.json")))
    rows=[]
    for title in TITLES:
        ordered, source_hash=candidates(title)
        if source_hash!=read(ROLE/prior.name)["source_hashes"][title]: raise ValueError("Changed edition")
        count=0
        for _,en,zh,ch,para,u in ordered:
            raw=[en,zh]
            norms=list(map(key,raw))
            if any(not n or n in texts[t] for t,n in enumerate(norms)): continue
            cells=list(map(ink.encode,raw))
            if any(not set(c)<=allowed[t] or ink.bad.intersection(raw[t]) for t,c in enumerate(cells)): continue
            sides=list(map(word,cells))
            if any(w in images[t] for t,w in enumerate(sides)): continue
            for t in (0,1):
                texts[t].add(norms[t])
                images[t].add(sides[t])
            rows.append(dict(doc=title,domain="book",chunk=ch,paragraph=para,unit=u,text=raw,cells=cells))
            count+=1
            if count==24: break
        if count!=24: raise ValueError("Insufficient held passages")
    with TemporaryDirectory(dir=RUN) as temp:
        staged=Path(temp)/"fresh.json"
        dump(staged,rows);staged.replace(target)
    out=dict(pairs=len(rows),directions=2*len(rows),titles=TITLES,pairs_per_work=24,prior_bitmap_sides=exclude,
             previous_panel=sha(prior),program=sha(Path(__file__)),sha256=sha(target))
    dump(receipt,out)
    print(out)


if __name__=="__main__":
    main()

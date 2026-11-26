"""Reserve disjoint pictured book requests for a paired print-transfer evaluation."""
from pathlib import Path
import numpy as np
from scripts.reserve_readout_prefixes import (
    BASE, WORK, PARENT, ROLE, PUB as PRIOR_PUB, RUN as PRIOR_RUN, ROOT,
    Ink, candidates, dump, key, library, pool, previous, read, sha, word,
)

from scripts.reserve_field_transfer import RUN as FIELD_RUN, PUB as FIELD_PUB

from scripts.reserve_target_glyph_book_panel import RUN as TARGET_RUN, PUB as TARGET_PUB
from scripts.prepare_visual_library import RECEIPT

TITLES = ("the-count-of-monte-cristo", "resurrection", "sanshiro", "siddhartha")
RUN=ROOT/"artifacts/print_translation_v1"
PUB=ROOT/"references/evidence/print_translation_v1"


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
    third=FIELD_RUN/"panel.json"
    if sha(third)!=read(FIELD_PUB/"panel.json")["sha256"]: raise ValueError("Prior field panel fingerprint")
    fourth=TARGET_RUN/"panel.json"
    if sha(fourth)!=read(TARGET_PUB/"panel.json")["sha256"]: raise ValueError("Prior target panel fingerprint")
    used=read(fourth)+read(third)+previous()+read(WORK/"held.json")+read(first)+read(second)
    excluded_titles=set(read(RECEIPT)["works"]) | set(read(ROOT/"references/evidence/book_breadth_v1/corpus.json")["local_book_sources"]) | {r["doc"] for r in used}
    if excluded_titles.intersection(TITLES): raise ValueError("A selected title occurs in named earlier corpora")
    editions = {}
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
        editions[title]=edition
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
    dump(receipt,dict(sha256=sha(target),earlier=[sha(first),sha(second),sha(third),sha(fourth)],editions=editions,titles=TITLES,pairs=len(picked),
                      prior_bitmap_sides=prior_counts,program=sha(Path(__file__)),protocol=sha(ROOT/"references/print_translation_study.md")))


if __name__=="__main__":
    main()

"""Reserve unseen editions before inducing the covered bilingual phrase model."""
from pathlib import Path
import numpy as np
from scripts import prepare_ordered_alignment as prior
from scripts.prepare_ordered_alignment import ROOT,base,past,candidates,pool,key,word

WORK=ROOT/"artifacts/covered_phrases_v1"
PUB=ROOT/"references/evidence/covered_phrases_v1"
MODEL=ROOT/"models/covered_phrases"
RULE=ROOT/"references/covered_phrases_protocol.md"


def main():
    WORK.mkdir(exist_ok=True,parents=True)
    PUB.mkdir(exist_ok=True,parents=True)
    if (PUB/"corpus.json").exists():
        assert base.read(PUB/"corpus.json")["panel"]==base.sha(WORK/"panel.json")
        return
    first=past.previous
    rows=first.previous()+base.read(first.WORK/"held.json")
    for path in (first.PARENT/"unseen-books.json",first.PRIOR_RUN/"fresh.json",
                 *(folder/"panel.json" for folder in (first.FIELD_RUN,first.TARGET_RUN,first.RUN,
                           past.span.RUN,past.pictured.RUN,past.drafts.RUN,past.actions.RUN,past.preceding.RUN,
                           past.RUN,base.FIRST,base.WORK,prior.WORK))):
        rows.extend(base.read(path))
    used=set(base.read(first.RECEIPT)["works"])|set(base.read(ROOT/"references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    for split in ("train","dev","test"):
        library=first.library(split)
        rows.extend(library)
        used.update(r["work"] for r in library)
    used.update(r["doc"] for r in rows if "doc" in r)
    normalized=[{key(r["text"][t]) for r in rows} for t in (0,1)]
    excluded=[{word(r["cells"][t]) for r in rows} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            excluded[1-t].add(x)
            excluded[t].add(y)
    pairs=prior.training_pairs()
    trained=[{r.tobytes() for r in pairs[:,t]} for t in (0,1)]
    face=base.Ink(np.load(base.BASE/"palette.npy"))
    inv=list(map(set,base.read(base.BASE/"inventories.json")))
    panel,editions=[],{}
    for title in past.TITLES:
        if any(k==title or k.startswith((title+"-spark",title+"-gpt")) for k in used):
            continue
        examples,digest=candidates(title)
        group=[]
        for _,en,zh,ch,para,unit in examples:
            txt=[en,zh]
            keys=list(map(key,txt))
            if any(not k or k in normalized[t] for t,k in enumerate(keys)):
                continue
            cells=list(map(face.encode,txt))
            if any(not 1<=len(v)<=96 or not set(v)<=inv[t] or face.bad.intersection(txt[t])
                   or word(v) in excluded[t] or base.compact(v) in trained[t] for t,v in enumerate(cells)):
                continue
            group.append(dict(doc=title,text=txt,cells=cells,chunk=ch,paragraph=para,unit=unit))
            for t in (0,1):
                normalized[t].add(keys[t])
                excluded[t].add(word(cells[t]))
            if len(group)==12:
                break
        if len(group)==12:
            panel.extend(group)
            editions[title]=digest
        if len(editions)==4:
            break
    assert len(panel)==48
    base.dump(WORK/"panel.json",panel)
    previous=base.read(prior.PUB/"corpus.json")
    assert previous["strips"]==[base.sha(prior.WORK/f"strips-{t}.npy") for t in (0,1)]
    base.dump(PUB/"corpus.json",dict(editions=editions,pairs=len(pairs),panel=base.sha(WORK/"panel.json"),
              strips=previous["strips"],previous=base.sha(prior.PUB/"corpus.json"),protocol=base.sha(RULE),source=base.sha(Path(__file__))))
    print(editions,flush=True)


if __name__=="__main__":
    main()

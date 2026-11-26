"""Reserve disjoint books and recover the exact earlier bilingual strip corpus."""
from pathlib import Path
import numpy as np
from scripts import fit_optical_lexicon as base
from scripts.prepare_full_image_roles import past
from scripts.reserve_print_translation import candidates, key, pool, word

ROOT = base.ROOT
WORK = ROOT / "artifacts/ordered_alignment_v1"
PUB = ROOT / "references/evidence/ordered_alignment_v1"
MODEL = ROOT / "models/ordered_alignment"
RULE = ROOT / "references/ordered_alignment_protocol.md"


def training_pairs():
    first = np.load(base.BOOK / "cells.npy")
    with np.load(base.REHEARSAL) as replacements:
        first[replacements["indices"]] = replacements["cells"]
    joined = np.concatenate((first, np.load(base.OPTICAL)))
    _, first_index = np.unique(joined.reshape(len(joined), -1).view("V384").ravel(), return_index=True)
    joined = joined[np.sort(first_index)]
    previous = base.read(base.FIRST / "panel.json") + base.read(base.WORK / "dev.json")
    forbidden = [{base.compact(row["cells"][t]) for row in previous} for t in (0, 1)]
    keep = [all(p[t].tobytes() not in forbidden[t] for t in (0, 1)) for p in joined]
    return joined[keep]


def reserve(pairs):
    prior = past.previous
    seen = prior.previous() + base.read(prior.WORK / "held.json")
    seen.extend(base.read(prior.PARENT / "unseen-books.json"))
    seen.extend(base.read(prior.PRIOR_RUN / "fresh.json"))
    for folder in (prior.FIELD_RUN, prior.TARGET_RUN, prior.RUN, past.span.RUN, past.pictured.RUN,
                   past.drafts.RUN, past.actions.RUN, past.preceding.RUN, past.RUN, base.FIRST, base.WORK):
        seen.extend(base.read(folder / "panel.json"))
    used = set(base.read(prior.RECEIPT)["works"]) | set(base.read(base.ROOT / "references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    for split in ("train", "dev", "test"):
        items = prior.library(split)
        used.update(r["work"] for r in items)
        seen.extend(items)
    used.update(r["doc"] for r in seen if "doc" in r)
    norm = [{key(r["text"][t]) for r in seen} for t in (0, 1)]
    blocked = [{word(r["cells"][t]) for r in seen} for t in (0, 1)]
    train_side = [{r.tobytes() for r in pairs[:,t]} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            blocked[1-t].add(x)
            blocked[t].add(y)
    face = base.Ink(np.load(base.BASE / "palette.npy"))
    inventory = list(map(set,base.read(base.BASE / "inventories.json")))
    panel, works = [], {}
    for title in past.TITLES:
        if any(k == title or k.startswith((title+"-spark", title+"-gpt")) for k in used):
            continue
        ordered, digest = candidates(title)
        selected = []
        for _,en,zh,ch,paragraph,unit in ordered:
            txt = [en, zh]
            keys = list(map(key,txt))
            if any(not k or k in norm[t] for t,k in enumerate(keys)):
                continue
            cells = list(map(face.encode,txt))
            if any(not 1<=len(v)<=96 or not set(v)<=inventory[t] or face.bad.intersection(txt[t])
                   or word(v) in blocked[t] or base.compact(v) in train_side[t] for t,v in enumerate(cells)):
                continue
            selected.append(dict(doc=title,text=txt,cells=cells,chunk=ch,paragraph=paragraph,unit=unit))
            for t in (0,1):
                norm[t].add(keys[t])
                blocked[t].add(word(cells[t]))
            if len(selected)==12:
                break
        if len(selected)==12:
            panel.extend(selected)
            works[title]=digest
        if len(works)==4:
            break
    assert len(panel)==48
    base.dump(WORK / "panel.json",panel)
    return works


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    pairs = training_pairs()
    assert len(pairs)==base.read(base.PUB / "corpus.json")["pairs"]==497775
    works = reserve(pairs)
    paths=[]
    for t,table in enumerate(base.read(base.TABLE)["strip"]):
        path=WORK / f"strips-{t}.npy"
        np.save(path,base.segment(np.ascontiguousarray(pairs[:,t]),*base.trie(table)))
        paths.append(base.sha(path))
    base.dump(PUB / "corpus.json",dict(pairs=len(pairs),editions=works,panel=base.sha(WORK/"panel.json"),
              strips=paths,previous=base.sha(base.PUB/"corpus.json"),protocol=base.sha(RULE),code=base.sha(Path(__file__))))
    print(list(works),len(pairs),flush=True)


if __name__=="__main__":
    main()

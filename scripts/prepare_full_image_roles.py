"""Assemble bilingual role masks, reserve new works, and prepare polarity training pairs."""
from collections import defaultdict, Counter
from pathlib import Path
import re
import numpy as np
from opencc import OpenCC
from scripts.prepare_visual_referents import WORK as ENGLISH, ROOT, dump, read, sha
from scripts.prepare_region_writer import Ink
from scripts.prepare_translation_paraphrases import alternatives
from scripts import reserve_visual_strip_restoration as past
from scripts.reserve_print_translation import BASE, candidates, key, word, pool
from scripts.prepare_natural_pictured_roles import clauses

WORK = ROOT/"artifacts/full_image_roles_v1"
PUB = ROOT/"references/evidence/full_image_roles_v1"
MODELS = ROOT/"models/full_image_roles"
BOOK = ROOT/"artifacts/book_breadth_v1"
REHEARSAL = ROOT/"artifacts/book_rehearsal_v1/replacement.npz"
RULE = ROOT/"references/full_image_roles_protocol.md"
CHINESE = ROOT/"data/raw/ud_chinese_gsd_r216"
LIMIT = 96


def reserve(ink):
    prior = past.previous
    used = prior.previous()+read(prior.WORK/"held.json")
    for p in (prior.PARENT/"unseen-books.json",prior.PRIOR_RUN/"fresh.json",
              *(r/"panel.json" for r in (prior.FIELD_RUN,prior.TARGET_RUN,prior.RUN,past.span.RUN,past.pictured.RUN,
                                          past.drafts.RUN,past.actions.RUN,past.preceding.RUN,past.RUN))):
        used.extend(read(p))
    titles = {r["doc"] for r in used} | set(read(prior.RECEIPT)["works"])
    titles.update(read(ROOT/"references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    for split in ("train","dev","test"):
        group = prior.library(split)
        titles.update(r["work"] for r in group)
        used.extend(group)
    keys = [{key(r["text"][t]) for r in used} for t in (0,1)]
    shapes = [{word(r["cells"][t]) for r in used} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            shapes[1-t].add(x)
            shapes[t].add(y)
    inventory = list(map(set,read(BASE/"inventories.json")))
    rows,editions = [],{}
    for title in past.TITLES:
        if any(k==title or k.startswith((title+"-spark",title+"-gpt")) for k in titles):
            continue
        ordered,digest = candidates(title)
        group = []
        for _,en,zh,ch,p,u in ordered:
            txt = [en,zh]
            norms = list(map(key,txt))
            if any(not v or v in keys[t] for t,v in enumerate(norms)):
                continue
            ids = list(map(ink.encode,txt))
            if any(len(v)>LIMIT or not set(v)<=inventory[t] or word(v) in shapes[t] for t,v in enumerate(ids)):
                continue
            for t in (0,1):
                keys[t].add(norms[t])
                shapes[t].add(word(ids[t]))
            group.append(dict(doc=title,domain="book",chunk=ch,paragraph=p,unit=u,text=txt,cells=ids))
            if len(group)==12:
                break
        if len(group)==12:
            rows.extend(group)
            editions[title]=digest
        if len(editions)==4:
            break
    assert len(rows)==48
    dump(WORK/"panel.json",rows)
    return editions


def mask_rows(rows,ink):
    # Keep all three annotated participants inside the same 96-cell source window.
    grouped = {}
    for r in rows:
        spans = [r["verb"],*r["target"]]
        lo,hi = min(a for a,b in spans),max(b for a,b in spans)
        if hi-lo>LIMIT:
            continue
        left = max(0,min(len(r["text"])-LIMIT,(hi+lo-LIMIT)//2))
        txt = r["text"][left:left+LIMIT]
        cells = tuple(ink.encode(txt))
        if not cells or ink.bad.intersection(txt):
            continue
        labels = grouped.setdefault(cells,np.zeros((LIMIT,3),np.uint8))
        for j,(a,b) in enumerate(spans):
            labels[a-left:b-left,j]=1
    return grouped


def zh_rows(path):
    convert = OpenCC("t2s")
    for block in path.read_text().split("\n\n"):
        for row in clauses(block,Counter(),"GSD"):
            text = convert.convert(row["text"])
            if len(text)==len(row["text"]):
                yield dict(text=text,verb=row["verb"],target=[row["candidates"][i] for i in (row["label"],1-row["label"])])


def event_examples(ink):
    inv = list(map(set,read(BASE/"inventories.json")))
    pairs = set()
    for kind,(en,zh),_ in alternatives():
        if kind!="event":
            continue
        # Training names follow the immutable earlier split by unordered participant pair.
        m = re.fullmatch(r"(.+?) (did not \w+|helped|visited|thanked|called|invited|followed|saw|found|hugged|praised) (.+)\.",en)
        if m is None:
            raise ValueError(en)
        a,v,b = m.groups()
        if a.startswith("The "):
            a = "the "+a[4:]
        for src in (en,f"It was {a} who {v} {b}.",f"It was {b} whom {a} {v}."):
            ids,targets = ink.encode(src),ink.encode(zh)
            if set(ids)<=inv[0] and set(targets)<=inv[1] and len(ids)<=LIMIT:
                pairs.add((tuple(ids),tuple(targets)))
    a = np.zeros((len(pairs),2,LIMIT),np.uint16)
    for i,pair in enumerate(sorted(pairs)):
        for t,side in enumerate(pair):
            a[i,t,:len(side)]=side
    np.save(WORK/"events.npy",a)
    return len(a)


def main():
    WORK.mkdir(exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    if (PUB/"corpus.json").exists():
        return
    ink = Ink(np.load(BASE/"palette.npy"))
    editions = reserve(ink)
    events = event_examples(ink)
    seen = set()
    counts = {}
    sources = {}
    for split in ("test","train"):
        structs = []
        for lang in ("en","zh"):
            file = ENGLISH/f"{split}.json" if lang=="en" else CHINESE/f"zh_gsd-ud-{split}.conllu"
            sources[str(file.relative_to(ROOT))]=sha(file)
            masks = mask_rows(read(file) if lang=="en" else zh_rows(file),ink)
            allowed = [k for k in masks if k not in seen]
            counts[f"{split}/{lang}"] = len(allowed)
            structs.extend((k,masks[k],lang=="zh") for k in allowed)
            seen.update(allowed)
        x = np.zeros((len(structs),LIMIT),np.int32)
        for i,(cells,_,_) in enumerate(structs):
            x[i,:len(cells)]=cells
        np.savez(WORK/f"{split}-roles.npz",cells=x,labels=np.stack([r[1] for r in structs]),lang=[r[2] for r in structs])
    np.save(WORK/"marker-palette.npy",ink.array())
    files=[WORK/n for n in ("panel.json","events.npy","test-roles.npz","train-roles.npz","marker-palette.npy")]
    dump(PUB/"corpus.json",dict(editions=editions,sources=sources,counts=counts,events=events,
         hashes={p.name:sha(p) for p in files},book=sha(BOOK/"cells.npy"),rehearsal=sha(REHEARSAL),
         program=sha(Path(__file__)),protocol=sha(RULE)))
    print(editions,counts,events,flush=True)


if __name__=="__main__":
    main()

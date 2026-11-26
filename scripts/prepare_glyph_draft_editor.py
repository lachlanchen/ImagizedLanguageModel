"""Reserve fresh books and build a reproducible corpus of actually emitted phrase drafts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from time import monotonic
import numpy as np
from scripts import prepare_covered_phrases as parent
from scripts.prepare_covered_phrases import ROOT,prior,base,candidates,past,key,word,pool
from scripts.write_covered_phrases import CoveredWriter

WORK=ROOT/"artifacts/glyph_draft_editor_v1"
PUB=ROOT/"references/evidence/glyph_draft_editor_v1"
MODEL=ROOT/"models/glyph_draft_editor"
TITLES=("sahara-cultural-history","nature-of-space-time","grand-design","brief-answers-big-questions")
RULE=ROOT/"references/glyph_draft_editor_protocol.md"
SEED=934107
CHUNK=256
ENGINES=None
PAIRS=None
STRENGTH=None


def reserve(pairs,events):
    first=past.previous
    old=first.previous()+base.read(first.WORK/"held.json")
    for path in (first.PARENT/"unseen-books.json",first.PRIOR_RUN/"fresh.json",
                 *(folder/"panel.json" for folder in (first.FIELD_RUN,first.TARGET_RUN,first.RUN,past.span.RUN,
                   past.pictured.RUN,past.drafts.RUN,past.actions.RUN,past.preceding.RUN,past.RUN,base.FIRST,
                   base.WORK,prior.WORK,parent.WORK))):
        old.extend(base.read(path))
    titles=set(base.read(first.RECEIPT)["works"]) | set(base.read(ROOT/"references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    for split in ("train","dev","test"):
        lib=first.library(split)
        titles.update(r["work"] for r in lib)
        old.extend(lib)
    titles.update(r["doc"] for r in old if "doc" in r)
    train=[{v.tobytes() for v in pairs[:,t]}|{v.tobytes() for v in events[:,t]} for t in (0,1)]
    norm=[{key(r["text"][t]) for r in old} for t in (0,1)]
    blocked=[{word(r["cells"][t]) for r in old} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            blocked[1-t].add(x)
            blocked[t].add(y)
    ink=base.Ink(np.load(base.BASE/"palette.npy"))
    inv=list(map(set,base.read(base.BASE/"inventories.json")))
    rows,editions=[],{}
    for title in TITLES:
        assert not any(k==title or k.startswith((title+"-spark",title+"-gpt")) for k in titles)
        candidates_for_work,digest=candidates(title)
        selected=[]
        for _,en,zh,ch,para,unit in candidates_for_work:
            txt=[en,zh]
            keys=list(map(key,txt))
            if any(not k or k in norm[t] for t,k in enumerate(keys)):
                continue
            cells=list(map(ink.encode,txt))
            if any(not 1<=len(v)<=96 or not set(v)<=inv[t] or ink.bad.intersection(txt[t]) or
                   base.compact(v) in train[t] or word(v) in blocked[t] for t,v in enumerate(cells)):
                continue
            selected.append(dict(doc=title,text=txt,cells=cells,chunk=ch,paragraph=para,unit=unit))
            for t in (0,1):
                norm[t].add(keys[t])
                blocked[t].add(word(cells[t]))
            if len(selected)==12:
                break
        assert len(selected)==12,title
        rows.extend(selected)
        editions[title]=digest
    base.dump(WORK/"panel.json",rows)
    return editions


def select():
    WORK.mkdir(exist_ok=True,parents=True)
    PUB.mkdir(exist_ok=True,parents=True)
    if (PUB/"selection.json").exists():
        old=base.read(PUB/"selection.json")
        assert all(base.sha(WORK/f)==h for f,h in old["hashes"].items())
        return
    allpairs=prior.training_pairs()
    events=np.load(base.FIRST/"events.npy")
    editions=reserve(allpairs,events)
    rng=np.random.default_rng(SEED)
    ids=np.sort(rng.choice(len(allpairs),32768,replace=False))
    chosen=np.concatenate((allpairs[ids],events))
    from scripts.evaluate_full_image_roles import rows
    face=base.Ink(np.load(base.BASE/"palette.npy"))
    held={base.compact(face.encode(r["source"])) for r in rows() if r["group"]=="object"}
    keep=np.asarray([r[0].tobytes() not in held for r in chosen])
    np.save(WORK/"pairs.npy",chosen[keep])
    np.save(WORK/"real_indices.npy",ids[keep[:len(ids)]])
    base.dump(PUB/"selection.json",dict(editions=editions,seed=SEED,real=int(keep[:len(ids)].sum()),
           events=int(keep[len(ids):].sum()),excluded_controlled_sources=int((~keep).sum()),
           protocol=base.sha(RULE),phrase_model=base.sha(parent.MODEL/"fit.json"),calibration=base.sha(parent.MODEL/"calibration.json"),
           hashes={f:base.sha(WORK/f) for f in ("panel.json","pairs.npy","real_indices.npy")}))
    print(editions,chosen[keep].shape,flush=True)


def shard(job):
    t,start=job
    path=WORK/f"draft-{t}-{start:06d}.npz"
    if path.exists():
        with np.load(path) as z:
            assert len(z["offsets"])==min(CHUNK,len(PAIRS)-start)+1 and z["offsets"][-1]==len(z["ids"])
        return path.name
    engine=ENGINES[t]
    outputs=[]
    for row in PAIRS[start:start+CHUNK,1-t]:
        cells=row[row!=0].tolist()
        strips=engine.input.source_strips(cells)[1:]
        assert [v for k in strips for v in engine.input_table[k]]==cells
        target,_,_=engine.inventory.decode(strips,*STRENGTH[t])
        outputs.append(np.array([g for k in target for g in engine.output[k]],np.uint16))
    offsets=np.cumsum([0]+list(map(len,outputs)),dtype=np.int32)
    partial=path.with_suffix(".partial.npz")
    np.savez_compressed(partial,offsets=offsets,ids=np.concatenate(outputs))
    partial.replace(path)
    return path.name


def prepare():
    global ENGINES,PAIRS,STRENGTH
    PAIRS=np.load(WORK/"pairs.npy",mmap_mode="r")
    ENGINES=[CoveredWriter(t) for t in (0,1)]
    STRENGTH=base.read(parent.MODEL/"calibration.json")["chosen"]
    jobs=[(t,i) for i in range(0,len(PAIRS),CHUNK) for t in (0,1)]
    start=monotonic()
    completed=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context("fork")) as workers:
        for name in workers.map(shard,jobs,chunksize=1):
            completed.append(name)
            if len(completed)%40==0:
                print("completed",len(completed),"/",len(jobs),"shards",flush=True)
            if monotonic()-start>7200:
                raise RuntimeError("Registered draft preparation ceiling exceeded")
    base.dump(PUB/"drafts.json",dict(selection=base.sha(PUB/"selection.json"),seconds=monotonic()-start,
               shards={f:base.sha(WORK/f) for f in completed}))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reserve-only",action="store_true")
    args=cli.parse_args()
    select()
    if not args.reserve_only:
        prepare()

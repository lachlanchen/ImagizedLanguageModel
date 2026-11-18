"""Evaluate unfamiliar entry families, independent requests, English questions and modern QA."""
import json
from time import monotonic
import numpy as np
import torch

from scripts.evaluate_same_passage_writer import generate, grade, image_archive, reference_sets
from scripts.prepare_pictured_entry_answers import write_json, sha
from scripts.train_general_entry_writer import (AUTHORED, DEVICE, Entries, PREVIOUS, WORK, PairCorpus,
            Rasterizer, memory_bank, physical_tables, small, load_writer)


def english_request(meta):
    fact = "original meaning" if meta["field"]=="本义" else "pronunciation"
    script = {"oracle":"oracle-bone", "bronze":"bronze", "seal":"seal","liushutong":"Liushutong"}[meta["stage"]]
    return f"Give the {fact} of {meta['head']} and its {script} form."


def pictured(bank,raster,text):
    return bank.merge([raster.tile(c) for c in text])


@torch.inference_mode()
def evaluate():
    start=monotonic()
    bank=memory_bank()
    data=Entries(bank,"development")
    ds=PairCorpus(bank,"development")
    du,du_groups=small(bank,"development")
    ids=np.unique(ds.rows[:,1:3])
    pos={i:k for k,i in enumerate(ids)}
    ix=np.array([[pos[i],pos[j]] for i,j in ds.rows[:,1:3]])
    q=[ds.questions[i] for i in ids]
    p=[ds.passages[ds.groups[i]] for i in ids]
    dr_refs=reference_sets(ds,ids,bank)
    du_refs=[{b"".join(bank.tiles[j] for j in y)} for y in du.y]
    authored=[json.loads(line) for line in AUTHORED.read_text().splitlines()]
    raster=Rasterizer()
    eng=[pictured(bank,raster,english_request(m)) for m in data.meta]
    independent=[pictured(bank,raster,r["text"]) for r in authored]
    n=len(data.rows)
    panels=[("simple",data.q,data.rows),("paraphrase",data.paraphrases,data.rows),
            ("independent",independent,np.array([r["row"] for r in authored])),
            ("english",eng,data.rows),("erased",[np.array([bank.lookup[bytes(128)]])]*n,data.rows)]
    features,mem,lookup=physical_tables(bank,DEVICE)
    report={}
    for arm in ("parent","paired","scrambled"):
        directory=WORK/arm
        if arm=="parent": directory.mkdir()
        net=load_writer(PREVIOUS/"contrast" if arm=="parent" else directory,DEVICE)
        ink=net.ink(features)
        keys=net.write_key(ink[mem])
        results={}
        for name,qs,rows in panels:
            sources=[data.cards[data.card_of[r]] for r in rows]
            images,stop=generate(net,ink,mem,keys,sources,qs,bank)
            image_archive(directory/(name+".npz"),images,stop)
            gold=[b"".join(bank.tiles[k] for k in data.y[r]) for r in rows]
            exact=[s and a==b for a,b,s in zip(images,gold,stop,strict=True)]
            targets=[b[-256:-128] for b in gold]
            assert all(data.y[r][-2] not in range(1,len(mem)+1) and data.y[r][-2] in src
                       for r,src in zip(rows,sources,strict=True))
            results[name]=dict(records=len(rows),exact=sum(exact),stopped=sum(stop),
                terminal_form=sum(s and len(a)>=256 and a[-256:-128]==t for a,s,t in zip(images,stop,targets)),
                all_in_group=int(np.asarray(exact).reshape(-1,4 if name=="independent" else 8).all(1).sum()))
        for name,ps,qs,refs,groups in (("drcd",p,q,dr_refs,None),("dureader",du.p,du.q,du_refs,du_groups)):
            images,stop=generate(net,ink,mem,keys,ps,qs,bank)
            image_archive(directory/(name+".npz"),images,stop)
            stats,ok=grade(images,stop,refs,groups)
            if name=="drcd": stats["both"]=int(ok[ix].all(1).sum())
            results[name]=stats
        report[arm]=results
        print(arm,results,flush=True)
    write_json(WORK/"result.json",dict(arms=report,seconds=monotonic()-start,
        fits={arm:json.loads((WORK/arm/"fit.json").read_text()) for arm in ("paired","scrambled")},
        archives={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob("*.npz")}))

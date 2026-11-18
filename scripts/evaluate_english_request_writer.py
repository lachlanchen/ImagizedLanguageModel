"""Generate whole images from held-out English and Chinese questions for all fixed checkpoints."""
from collections import Counter
import json
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.evaluate_general_entry_writer import english_request, pictured
from scripts.evaluate_same_passage_writer import generate, grade, image_archive, reference_sets
from scripts.train_general_entry_writer import AUTHORED, PairCorpus
from scripts.train_english_request_writer import (CROSS, DEVICE, Entries, FACT, FORM, ORIGINAL, PUBLIC,
          Rasterizer, WORK, load_writer, memory_bank, physical_tables, sha, small, write_json)


def prompts(bank,d):
    r=Rasterizer()
    q_english=[pictured(bank,r,english_request(m)) for m in d.meta]
    crossed=[]
    for i,m in enumerate(d.meta):
        a=0 if m["field"]=="本义" else 1
        text=CROSS.format(h=m["head"],f=FACT[a][0],n=FACT[1-a][0],s=FORM[m["stage"]][0],
                          u=FORM[d.meta[i^1]["stage"]][0])
        crossed.append(pictured(bank,r,text))
    for i in range(0,len(crossed),4):
        assert all(np.array_equal(np.sort(crossed[i]),np.sort(q)) for q in crossed[i+1:i+4])
    authored={}
    for lang,path in (("english",PUBLIC/"english.jsonl"),("chinese",AUTHORED)):
        records=[json.loads(l) for l in path.read_text().splitlines()]
        authored[lang]=([pictured(bank,r,rec["text"]) for rec in records],[rec["row"] for rec in records])
    return {"english":(q_english,d.rows),"crossed":(crossed,d.rows),"english_free":authored["english"],
            "chinese":(d.q,d.rows),"chinese_free":authored["chinese"],
            "white":([np.array([bank.lookup[bytes(128)]])]*len(d.rows),d.rows)}


@torch.inference_mode()
def evaluate():
    start=monotonic()
    bank=memory_bank()
    d=Entries(bank,"development")
    dr=PairCorpus(bank,"development")
    du,du_groups=small(bank,"development")
    panels=prompts(bank,d)
    renderer=Rasterizer()
    bounds=[bank.lookup[renderer.tile(c)] for c in "【】"]
    ids=np.unique(dr.rows[:,1:3])
    dp=[dr.passages[dr.groups[i]] for i in ids]
    dq=[dr.questions[i] for i in ids]
    dr_ref=reference_sets(dr,ids,bank)
    du_ref=[{b"".join(bank.tiles[k] for k in y)} for y in du.y]
    pairs=np.searchsorted(ids,dr.rows[:,1:3])
    targets=[b"".join(bank.tiles[k] for k in y) for y in d.y]
    features,mem,_=physical_tables(bank,DEVICE)
    report={}
    for arm in ("parent","matched","scrambled"):
        directory=WORK/arm
        if arm=="parent":
            directory.mkdir()
        net=load_writer(ORIGINAL/"paired" if arm=="parent" else directory,DEVICE)
        ink=net.ink(features)
        writing=net.write_key(ink[mem])
        results={}
        for name,(qs,rows) in panels.items():
            cards=[d.cards[d.card_of[i]] for i in rows]
            locations=[select_visible_record(p,q,*bounds) for p,q in zip(cards,qs,strict=True)]
            for layout in ("located","full") if name in ("english","english_free","chinese") else ("located",):
                sources=cards if layout=="full" else [s for s,_ in locations]
                key=name+"_"+layout
                images,end=generate(net,ink,mem,writing,sources,qs,bank)
                refs=[targets[i] for i in rows]
                ok=np.array([s and x==y for s,x,y in zip(end,images,refs,strict=True)])
                per_head=ok.reshape(-1,4).all(1)
                results[key]=dict(records=len(rows),exact=int(ok.sum()),stopped=sum(end),four=int(per_head.sum()),
                    terminal_form=sum(e and x[-256:-128]==y[-256:-128] for e,x,y in zip(end,images,refs,strict=True)),
                    matched_head=sum(flag=="located" for _,flag in locations))
                image_archive(directory/(key+".npz"),images,end)
        for key,p,q,refs,group in (("drcd",dp,dq,dr_ref,None),("dureader",du.p,du.q,du_ref,du_groups)):
            images,end=generate(net,ink,mem,writing,p,q,bank)
            results[key],ok=grade(images,end,refs,group)
            if key=="drcd": results[key]["both"]=int(ok[pairs].all(1).sum())
            image_archive(directory/(key+".npz"),images,end)
        report[arm]=results
        print(arm,{k:v["exact"] if "exact" in v else 0 for k,v in results.items()},flush=True)
    write_json(WORK/"result.json",dict(arms=report,seconds=monotonic()-start,
          fits={k:json.loads((WORK/k/"fit.json").read_text()) for k in ("matched","scrambled")},
          archives={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob("*.npz")}))


if __name__=="__main__":
    evaluate()

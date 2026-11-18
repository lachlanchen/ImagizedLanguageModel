"""Test complete visual answers when the same local glyph neighborhoods specify four intents."""
from time import monotonic
import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.evaluate_english_request_writer import prompts
from scripts.evaluate_same_passage_writer import generate, grade, image_archive, reference_sets
from scripts.load_scoped_writer import load_visual_writer
from scripts.prepare_long_scope_questions import HELD, UNSEEN, variants
from scripts.train_general_entry_writer import DEVICE, Entries, Rasterizer, PairCorpus, memory_bank, physical_tables, small
from scripts.train_scoped_request_writer import PREVIOUS, WORK, load, sha, write_json


@torch.inference_mode()
def evaluate():
    start=monotonic()
    bank=memory_bank()
    d=Entries(bank,"development")
    dr=PairCorpus(bank,"development")
    du,groups=small(bank,"development")
    r=Rasterizer()
    panels=prompts(bank,d)
    for name,frame in (("long",HELD),("new_cues",UNSEEN)):
        layouts=variants(d.meta,r,bank,frame)
        panels[name]=([q for layout in layouts for q in layout],np.tile(d.rows,len(layouts)))
    bounds=[bank.lookup[r.tile(c)] for c in "【】"]
    ids=np.unique(dr.rows[:,1:3])
    pairs=np.searchsorted(ids,dr.rows[:,1:3])
    refs=reference_sets(dr,ids,bank)
    du_ref=[{b"".join(bank.tiles[k] for k in y)} for y in du.y]
    targets=[b"".join(bank.tiles[k] for k in y) for y in d.y]
    features,mem,_=physical_tables(bank,DEVICE)
    report={}
    for arm in ("parent","ordered","unordered"):
        path=WORK/arm
        if arm=="parent":path.mkdir()
        net=load_visual_writer(PREVIOUS/"matched" if arm=="parent" else path,DEVICE)
        enc=net.ink(features)
        keys=net.write_key(enc[mem])
        stats={}
        for name,(questions,rows) in panels.items():
            full=[d.cards[d.card_of[i]] for i in rows]
            ps=[select_visible_record(p,q,*bounds)[0] for p,q in zip(full,questions,strict=True)]
            images,ends=generate(net,enc,mem,keys,ps,questions,bank)
            image_archive(path/(name+".npz"),images,ends)
            ys=[targets[i] for i in rows]
            ok=np.array([e and x==y for e,x,y in zip(ends,images,ys,strict=True)])
            blocks=ok.reshape(-1,4)
            stats[name]=dict(records=len(rows),exact=int(ok.sum()),stopped=sum(ends),four=int(blocks.all(1).sum()),
                 distinct_in_quartets=sum(len(set(images[i:i+4])) for i in range(0,len(images),4)),
                 identical_quartets=sum(len(set(images[i:i+4]))==1 for i in range(0,len(images),4)),
                 terminal_form=sum(e and x[-256:-128]==y[-256:-128] for e,x,y in zip(ends,images,ys,strict=True)))
        for name,p,q,ref,g in (("drcd",[dr.passages[dr.groups[i]] for i in ids],[dr.questions[i] for i in ids],refs,None),
                               ("dureader",du.p,du.q,du_ref,groups)):
            images,end=generate(net,enc,mem,keys,p,q,bank)
            image_archive(path/(name+".npz"),images,end)
            stats[name],ok=grade(images,end,ref,g)
            if name=="drcd":stats[name]["both"]=int(ok[pairs].all(1).sum())
        report[arm]=stats
        print(arm,{k:(v.get("exact",0),v.get("four")) for k,v in stats.items()},flush=True)
    write_json(WORK/"result.json",dict(arms=report,seconds=monotonic()-start,
             fits={a:load(WORK/a/"fit.json") for a in ("ordered","unordered")},
             archives={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob("*.npz")}))


if __name__=="__main__":
    evaluate()

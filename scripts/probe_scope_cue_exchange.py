"""Cross the slot wrappers, known-information cue and requested-information cue with frozen weights."""
from itertools import product
from time import monotonic
import fcntl
import os
import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.audit_traditional_passage_training import archived, clustered_delta
from scripts.evaluate_english_request_writer import prompts
from scripts.evaluate_same_passage_writer import generate, image_archive
from scripts.load_scoped_writer import load_visual_writer
from scripts.prepare_long_scope_questions import HELD, UNSEEN, variants
from scripts.train_general_entry_writer import DEVICE, Entries, PairCorpus, Rasterizer, memory_bank, physical_tables, small
from scripts.train_scoped_request_writer import PUBLIC, ROOT, WORK, load, sha, write_json


@torch.inference_mode()
def probe():
    start=monotonic()
    dest=WORK/"cue_exchange"
    dest.mkdir()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    bank=memory_bank()
    d=Entries(bank,"development")
    PairCorpus(bank,"development")
    small(bank,"development")
    prompts(bank,d)
    r=Rasterizer()
    for frame in (HELD,UNSEEN):
        # Preserve physical bank order and dimensions of the original generation.
        variants(d.meta,r,bank,frame)
    original_size=len(bank.tiles)
    features,mem,_=physical_tables(bank,DEVICE)
    net=load_visual_writer(WORK/"ordered",DEVICE)
    ink=net.ink(features)
    writing=net.write_key(ink[mem])
    corners=load(WORK/"result.json")["archives"]
    gold=[b"".join(bank.tiles[k] for k in y) for y in d.y]*4
    starts=[bank.lookup[r.tile(c)] for c in "【】"]
    rows=np.tile(d.rows,4)
    groups=np.array([d.meta[i]["card"] for i in rows])[::4]
    result,joints={},{}
    for a,b,c in product(range(2),repeat=3):
        wrap,known,asked=[(HELD,UNSEEN)[i] for i in (a,b,c)]
        frame=(wrap[0],wrap[1],known[2],asked[3],wrap[4])
        label=f"{a}{b}{c}"
        if a==b==c:
            key="ordered/"+("long" if a==0 else "new_cues")+".npz"
            digest=corners[key]
            imgs,end=archived(WORK/key,digest)
        else:
            qs=[q for layout in variants(d.meta,r,bank,frame) for q in layout]
            assert original_size==len(bank.tiles)
            ps=[select_visible_record(d.cards[d.card_of[i]],q,*starts)[0] for i,q in zip(rows,qs,strict=True)]
            imgs,end=generate(net,ink,mem,writing,ps,qs,bank)
            path=dest/(label+".npz")
            image_archive(path,imgs,end)
            digest=sha(path)
            imgs,end=archived(path,digest)
        ok=np.array([e and x==y for x,e,y in zip(imgs,end,gold,strict=True)])
        four=ok.reshape(-1,4).all(1)
        joints[label]=four
        result[label]=dict(exact=int(ok.sum()),four=int(four.sum()),layout_four=four.reshape(4,-1).sum(1).tolist(),
                           stopped=int(sum(end)),sha256=digest)
        print(label,result[label],flush=True)
    comparison={}
    for key,val in joints.items():
        for i,v in enumerate(key):
            if v=="0":
                flipped=key[:i]+"1"+key[i+1:]
                stats=clustered_delta(joints[flipped],val,groups)
                stats["cards"]=stats.pop("articles")
                comparison[key+"->"+flipped]=stats
    write_json(PUBLIC/"cue_exchange.json",dict(factors=["slot_wording","known_cue","requested_cue"],
              cells=result,paired_changes=comparison,seconds=monotonic()-start,source_sha256=sha(__file__)))


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("GPU 0 required")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        probe()

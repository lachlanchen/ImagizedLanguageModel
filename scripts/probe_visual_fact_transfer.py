"""Measure direct-answer transfer and physical equivariance without changing any weights."""
from collections import Counter
import argparse
import fcntl
import hashlib
import json
import os
import random
import numpy as np
import torch

from ilm.visual_lm.bound_fact_reader import BoundFactReader,best_bounds,features_batch,contextualize
from scripts.prepare_visual_facts import Face,FONTS,ROOT,WORK,PUBLIC,SEED,RELATIONS,statement,simplify,write_json
from scripts.train_visual_facts import rows,READER
from scripts.ordered_image_corpus import PLATES
from scripts.evaluate_grounded_propositions import NOTES
from scripts.train_pointing_causal_writer import sha


@torch.inference_mode()
def probe(archive_only=False):
    from scripts.span_reservation import reservation
    guard=reservation()
    model=BoundFactReader(READER,WORK/"genuine","cuda");face=Face((FONTS[0],2))
    exclude={simplify(x["p"]) for split in ("training","held") for x in rows(split)}
    stats={};ledger=[]
    sources=(("cmrc","data/raw/cmrc2018/cmrc2018_dev.json"),("drcd","data/raw/drcd/DRCD_dev.json"))
    for name,f in (() if archive_only else sources):
        rng=random.Random(SEED+777)
        candidates=[]
        for article in json.loads((ROOT/f).read_text())["data"]:
            for para in (guard.filter_development([article]) if name=="drcd" else article["paragraphs"]):
                p=para["context"][:416]
                if simplify(p) in exclude:continue
                for q in para["qas"]:
                    refs=[a["text"] for a in q["answers"] if a["text"] in p and 0<len(a["text"])<=48]
                    if refs and len(q["question"])+len(p)<=509:
                        candidates.append((p,q["question"],refs,q["id"]))
        rng.shuffle(candidates);cohort=[]
        for p,q,rs,key in candidates:
            if simplify(p) in exclude:continue
            try:pair=(face(p),face(q))
            except ValueError:continue
            exclude.add(simplify(p));cohort.append((pair,[face(r) for r in rs],key))
            if len(cohort)>=128:break
        c=Counter()
        for start in range(0,len(cohort),32):
            batch=cohort[start:start+32];pairs=[r[0] for r in batch]
            vals=features_batch(contextualize(model.reader,pairs),"cuda")
            _,logits=model.head(*vals)
            preds=best_bounds(logits[:,:,2:]).tolist()
            prior=best_bounds(vals[1][:,:,768:770].masked_fill(~vals[2][:,:,None],-1e4)).tolist()
            for ((p,_),refs,key),(a,b),(u,v) in zip(batch,preds,prior,strict=True):
                correct=p[128*a:128*b] in refs;baseline=p[128*u:128*v] in refs
                ledger.append(dict(group=name,key=key,adapted=correct,baseline=baseline));c.update(n=1,adapted=correct,baseline=baseline)
        stats[name]=dict(c)
    card=json.loads((PLATES/"yan/source.json").read_text())["rows"]
    assert sha(PLATES/"pixels.npy")==json.loads((PLATES/"report.json").read_text())["pixels_sha256"]
    bank=np.load(PLATES/"pixels.npy")
    tiles=[np.packbits(bank[r["tile"]].reshape(-1)).tobytes() if bank[r["tile"]].size==1024 else bank[r["tile"]].tobytes() for r in card]
    assert len(set(tiles))==4 and all(len(t)==128 for t in tiles)
    chars=set("".join(x["p"]+x["q"]+x["o"] for x in rows("training")))
    assert set(tiles).isdisjoint(face(c) for c in chars)
    cases=[];ref=[];original=[]
    for x in rows("held"):
        if x["group"]!="notes":continue
        _,_,a,_,b,*_=NOTES[int(x["key"])//4]
        src=face(x["p"]);q=face(x["q"])
        assert len(a)==len(b)==2
        def rename(raw):
            pieces=[raw[i:i+128] for i in range(0,len(raw),128)]
            for name,rep in ((a,tiles[:2]),(b,tiles[2:])):
                key=[face(c) for c in name]
                for i in range(len(pieces)-1):
                    if pieces[i:i+2]==key:pieces[i:i+2]=rep
            return b"".join(pieces)
        cases.append((rename(src),rename(q)));ref.append(rename(face(statement(x["s"],x["r"],x["o"]))))
        original.append((src,q))
    pred=model.answer_many(cases)
    initial=model.answer_many(original)
    # Freeze each page's own name mapping; transformations are distinct across programmes.
    equiv=[]
    for i,(plain,after) in enumerate(zip(initial,pred)):
        _,_,a,_,b,*_=NOTES[i//4];wanted=plain
        for name,rep in ((a,tiles[:2]),(b,tiles[2:])):
            glyphs=[wanted[j:j+128] for j in range(0,len(wanted),128)]
            for j in range(len(glyphs)-1):
                if glyphs[j:j+2]==[face(c) for c in name]:glyphs[j:j+2]=rep
            wanted=b"".join(glyphs)
        equiv.append(after==wanted)
    hits=[y==t for y,t in zip(pred,ref)]
    stats["name_substitution"]=dict(n=32,complete=sum(hits),all_four=sum(all(hits[i:i+4]) for i in range(0,32,4)),
        commutes=sum(equiv),tiles_sha256=[hashlib.sha256(t).hexdigest() for t in tiles])
    if archive_only:
        frozen=json.loads((PUBLIC/"transfer.json").read_text())["name_substitution"]
        if stats["name_substitution"]!=frozen:raise ValueError("Saved substitution scores changed")
    else:
        write_json(PUBLIC/"transfer.json",stats);write_json(WORK/"transfer_questions.json",ledger)
    from scripts.evaluate_grounded_propositions import store_streams
    from ilm.visual_lm.causal_glyph_writer import write_strip
    other=PUBLIC/"aliases";other.mkdir(exist_ok=True)
    for name,seqs in (("pages",[p for p,_ in cases]),("questions",[q for _,q in cases]),("outputs",pred),("reference",ref)):
        store_streams(WORK/("alias_"+name+".npz"),seqs,[True]*len(seqs))
        for i in (0,1,2,3):
            write_strip(np.frombuffer(seqs[i],np.uint8).reshape(-1,128),other/f"film_{i}_{name}.png")
    write_json(other/"evidence.json",dict(hits=hits,substitution_commutes=equiv,source_labels=[r["label"] for r in card],
        original_plate_sha256=sha(PLATES/"yan/source.json"),
        streams={n:sha(WORK/("alias_"+n+".npz")) for n in ("pages","questions","outputs","reference")}))
    print(stats,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-aliases",action="store_true")
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(3);probe(args.archive_aliases)

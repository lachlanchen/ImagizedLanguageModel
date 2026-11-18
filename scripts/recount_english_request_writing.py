"""Recount complete saved images, independent prompts and repeat-layout comparisons by card."""
import argparse
from collections import Counter
import json

import numpy as np

from scripts.audit_traditional_passage_training import archived, clustered_delta
from scripts.compare_traditional_passage_training import TraditionalPassages
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_english_request_writer import ENTRIES, PUBLIC, WORK
from scripts.train_general_entry_writer import AUTHORED
from scripts.train_grounded_glyph_writer import Split
from scripts.train_paired_passage import unpack_sequences
from scripts.train_same_passage_writer import PAIRS
from scripts import train_scoped_request_writer as scope


def contrast(a,b,cluster):
    out=clustered_delta(a,b,np.asarray(cluster))
    out["cards"]=out.pop("articles")
    return out


def recount(scoped):
    root,public=(scope.WORK,scope.PUBLIC) if scoped else (WORK,PUBLIC)
    public.mkdir(exist_ok=True)
    result=load(root/"result.json")
    folder=ENTRIES/"development"
    meta=load(folder/"index.json")
    pix=np.load(folder/"pixels.npy")
    targets=[pix[y].tobytes() for y in unpack_sequences(folder/"answers.npz")]
    options=[{} for _ in range(len(meta)//8)]
    for i,m in enumerate(meta):
        options[m["card"]][targets[i]]=i
    english=[json.loads(l)["row"] for l in (PUBLIC/"english.jsonl").read_text().splitlines()]
    chinese=[json.loads(l)["row"] for l in AUTHORED.read_text().splitlines()]
    reviewed=set(map(int,load(PUBLIC/"intent_review.json")["excluded_from_secondary_only"]))
    du=Split("development")
    ds=TraditionalPassages("development")
    pairs=np.load(PAIRS/"development.npy")[:,1:3]
    ids=np.unique(pairs)
    refs={i:{ds.bank[ds.passages[ds.groups[i]][a:b]].tobytes() for a,b in ds.intervals[i]} for i in ids}
    saved={}
    group_ids={}
    analysis={}
    for arm,conds in result["arms"].items():
        saved[arm]={}
        summary={}
        for mode,stats in conds.items():
            k=arm+"/"+mode+".npz"
            imgs,end=archived(root/k,result["archives"][k])
            assert len(imgs)==stats["records"] and sum(end)==stats["stopped"]
            if mode=="drcd":
                ok={i:bool(e and im in refs[i]) for i,im,e in zip(ids,imgs,end,strict=True)}
                assert sum(ok[i] and ok[j] for i,j in pairs)==stats["both"] and sum(ok.values())==stats["exact"]
                continue
            if mode=="dureader":
                ok=[bool(e and im==du.tiles[y].tobytes()) for im,e,y in zip(imgs,end,du.y,strict=True)]
                assert sum(ok)==stats["exact"]
                for group in set(du.groups):
                    assert sum(v for v,g in zip(ok,du.groups) if g==group)==stats[group+"_exact"]
                continue
            rows=english if mode.startswith("english_free") else chinese if mode.startswith("chinese_free") else np.tile(np.arange(len(meta)),len(imgs)//len(meta))
            ok=np.array([e and im==targets[i] for e,im,i in zip(end,imgs,rows,strict=True)])
            joint=ok.reshape(-1,4).all(1)
            assert int(ok.sum())==stats["exact"] and int(joint.sum())==stats["four"]
            cluster=np.array([meta[i]["card"] for i in rows])
            assert np.all(cluster.reshape(-1,4)==cluster[::4,None])
            saved[arm][mode]=(ok,joint)
            group_ids[mode]=cluster
            errors=Counter()
            for i,im,e in zip(rows,imgs,end,strict=True):
                j=options[meta[i]["card"]].get(im)
                if not e:
                    errors["unstopped"]+=1
                elif j is None:
                    errors["outside_card_answers"]+=1
                elif j==i:
                    errors["correct"]+=1
                else:
                    diff=[key for key in ("head","field","stage") if meta[i][key]!=meta[j][key]]
                    errors["wrong_"+"_and_".join(diff)]+=1
            summary[mode]=dict(errors=errors)
            if mode.startswith("english_free"):
                keep=np.array([i not in reviewed for i in range(len(imgs))])
                summary[mode].update(clear_intents=int(ok[keep].sum()),clear_total=int(keep.sum()))
            if mode in ("long","new_cues"):
                counts=ok.reshape(4,-1)
                summary[mode]["layout_exact"]=counts.sum(1).tolist()
                summary[mode]["layout_four"]=counts.reshape(4,-1,4).all(2).sum(1).tolist()
                summary[mode]["all_four_intents_in_all_four_layouts"]=int(counts.reshape(4,-1,4).all(2).all(0).sum())
                summary[mode]["unchanged_within_quartet"]=sum(len(set(imgs[i:i+4]))==1 for i in range(0,len(imgs),4))
        analysis[arm]=summary
    winner,control=("ordered","unordered") if scoped else ("matched","scrambled")
    deltas={}
    for mode,(a,g) in saved[winner].items():
        groups=group_ids[mode]
        deltas[mode]={}
        for other in (control,"parent"):
            b,h=saved[other][mode]
            deltas[mode][other]=dict(exact=contrast(a,b,groups),four=contrast(g,h,groups[::4]))
    write_json(public/"result.json",result)
    write_json(public/"audit.json",dict(archive_count=len(result["archives"]),breakdown=analysis,deltas=deltas,source_sha256=sha(__file__)))
    print({m:(v[control]["exact"]["ci95_pp"],v[control]["four"]["ci95_pp"]) for m,v in deltas.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scoped",action="store_true")
    recount(p.parse_args().scoped)

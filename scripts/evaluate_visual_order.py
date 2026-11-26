"""Seal and replay physical order interventions with the same field selections."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from ilm.visual_lm.visual_order import MODEL, VisualOrder
from ilm.visual_lm.origin_instruction import question
from scripts.evaluate_origin_instructions import ARCHIVE as PREVIOUS, cases as old_cases, counts as count_old
from scripts.evaluate_origin_value_reader import physical,board
from scripts.fit_visual_order import FRESH, PROTOCOL
from scripts.predict_ordered_values import predict_two
from scripts.prepare_region_writer import Ink
from scripts.train_origin_instruction import load
from scripts.train_origin_value_reader import ROOT, OUT, read, dump, sha, rows
from scripts.write_origin_values import decode,strip
from scripts.train_field_to_glyph import configure_cuda_cli,gpu_owner

PUB=ROOT/"references/evidence/visual_order_v1"
ARCHIVE=ROOT/"publication/ilm-nature/evidence/visual_order_answers.zip"


def cases():
    with ZipFile(PREVIOUS) as z:
        for r in json.loads(z.read("records.json")):
            def get(k): return z.read("images/"+r[k]+".png")
            yield {k:r[k] for k in ("i","kind","language","style","head","entry","stage","cells","geometry")},get("question"),get("page"),get("target")
    tiles=list(map(bytes,np.load(OUT/"tiles.npy")))
    ink=Ink()
    for i,r in enumerate(rows("sealed")):
        _,page=physical(r,tiles)
        s=r["supervision"]
        style=(i//4)%2
        for swapped in (False,True):
            text,target=question(r,FRESH[s["english"]][style],swapped)
            q=[ink.tiles[k] for k in ink.encode(text)]
            assert not ink.bad
            qp,pp,geom=board(q,page)
            a,b=target
            bits=[*page[a]["value"],bytes(128),bytes(128),*page[b]["value"]]
            yield dict(i=i,kind="new_swap" if swapped else "new",language="en" if s["english"] else "zh",
                       style=style,head=s["head"],entry=s["entry"],stage=s["form"],cells=len(bits),geometry=geom),qp,pp,strip(bits)


def summarize(records):
    groups={}
    paired={}
    def update(key,values):
        counter=groups.setdefault(key,Counter())
        counter.update(values)
        counter["n"]+=1
    for r in records:
        values={m:v["correct"] for m,v in r["answers"].items()}
        for key in (r["kind"],r["kind"]+"/"+r["language"]):
            update(key,values)
        paired.setdefault(r["i"],{})[r["kind"]]=r
    for fam in paired.values():
        for label,first,last in (("prior_pairs","fresh","order_swap"),("new_pairs","new","new_swap")):
            a,b=fam[first],fam[last]
            assert a["page"]==b["page"] and a["question"]!=b["question"] and a["target"]!=b["target"]
            values={m:a["answers"][m]["correct"] and b["answers"][m]["correct"] for m in ("varied","ordered")}
            targets={a["target"],b["target"]}
            values["available_both"]=all(r["answers"]["varied"]["stop"] and r["answers"]["varied"]["image"] in targets for r in (a,b))
            assert not values["ordered"] or values["available_both"]
            for key in (label,label+"/"+a["language"],label+"/style"+str(a["style"])):
                update(key,values)
    return groups


def execute(replay):
    reader=load("varied","cpu" if replay else "cuda")
    order=VisualOrder()
    device=next(reader.parameters()).device
    tmp=ARCHIVE.with_suffix(".tmp")
    with ZipFile(ARCHIVE if replay else tmp,"r" if replay else "w",compression=ZIP_DEFLATED) as z:
        seen=set()
        result=[]
        exact=Counter()
        differences=[]
        def image(blob):
            h=sha256(blob).hexdigest()
            if h not in seen:
                z.writestr("images/"+h+".png",blob)
                seen.add(h)
            return h
        def process(batch):
            items=[decode(q,p,r["geometry"]) for r,q,p,_ in batch]
            for (r,q,p,y),both in zip(batch,predict_two(reader,order,items,device),strict=True):
                if not replay: r.update(page=image(p),question=image(q),target=image(y),answers={})
                for arm,(png,n,stop,sel) in both.items():
                    if replay:
                        old=r["answers"][arm]
                        if (sha256(png).hexdigest(),n,stop,sel)==(old["image"],old["cells"],old["stop"],old["rows"]):
                            exact[arm]+=1
                        else: differences.append(dict(i=r["i"],kind=r["kind"],arm=arm,sha=sha256(png).hexdigest(),stop=stop,rows=sel))
                    else:
                        r["answers"][arm]=dict(image=image(png),cells=n,stop=stop,rows=sel,correct=(stop and n==r["cells"] and png==y))
                assert sorted(r["answers"]["ordered"]["rows"])==sorted(r["answers"]["varied"]["rows"])
                assert r["answers"]["ordered"]["stop"]==r["answers"]["varied"]["stop"]
                result.append(r)
        if replay:
            records=json.loads(z.read("records.json"))
            it=((r,*(z.read("images/"+r[k]+".png") for k in ("question","page","target"))) for r in records)
        else: it=cases()
        batch=[]
        for item in it:
            batch.append(item)
            if len(batch)==48:
                process(batch)
                batch=[]
        if batch: process(batch)
        stats=summarize(result)
        if not replay: z.writestr("records.json",json.dumps(result,ensure_ascii=False))
    if replay:
        assert stats==read(PUB/"evaluation.json")["counts"]
        dump(PUB/"replay.json",dict(archive=sha(ARCHIVE),same=exact,differences=differences))
        print(exact,len(differences),flush=True)
    else:
        tmp.replace(ARCHIVE)
        passed=all(stats[k]["ordered"]>stats[k]["varied"] for k in ("new_pairs/en","new_pairs/zh")) and stats["original"]["ordered"]>=.99*stats["original"]["n"]
        dump(PUB/"evaluation.json",dict(archive=sha(ARCHIVE),previous=sha(PREVIOUS),counts=stats,passed=passed,
             requests=len(result),unique_images=len(seen),fit=sha(MODEL/"fit.json"),protocol=sha(PROTOCOL),program=sha(Path(__file__))))
        print(stats["new_pairs"],stats["original"],passed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("generate","replay"))
    action=p.parse_args().action
    torch.set_num_threads(2)
    PUB.mkdir(exist_ok=True)
    if (PUB/"evaluation.json").exists():
        assert sha(ARCHIVE)==read(PUB/"evaluation.json")["archive"]
    if action=="replay": execute(True)
    elif not (PUB/"evaluation.json").exists():
        configure_cuda_cli()
        with gpu_owner(): execute(False)

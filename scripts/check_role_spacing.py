"""Audit optical spacing correction and equal-geometry changes of grammatical voice."""
from collections import Counter,defaultdict
from io import BytesIO
import json
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import torch
from ilm.visual_lm.paper_spacing import compact_paper
from scripts.prepare_natural_pictured_roles import ROOT,PUB,RUN,MODELS,read,dump,sha
from scripts.read_natural_roles import load,choose,load_strip,strip_png


def main():
    torch.set_num_threads(2)
    nets={arm:load(arm) for arm in ("context","layout")}
    pal=np.load(MODELS/"palette.npy");codes=read(RUN/"codes.json")
    original=ROOT/"publication/ilm-nature/evidence/natural_pictured_roles.zip"
    assert sha(original)==read(PUB/"assessment.json")["archive"]
    changes=[];official=0
    with ZipFile(original) as z:
        for r in json.loads(z.read("records.json")):
            if r["corpus"]=="role challenge":continue
            official+=1
            ink=load_strip(z.read(f"{r['i']:04d}/source.png"),r["cells"])
            packed,v,c=compact_paper(ink,r["verb"],r["candidates"])
            k,_=choose(nets["context"],packed,v,c)
            if k!=r["choices"]["context"]:changes.append(r["i"])
    source=ROOT/"publication/ilm-nature/evidence/natural_roles_constructions.zip"
    assert sha(source)==read(PUB/"audit.json")["archive"]
    counts=defaultdict(Counter)
    raw=ROOT/"publication/ilm-nature/evidence/natural_roles_spacing.zip"
    tmp=raw.with_suffix(".tmp");results=[];layouts={};joint=defaultdict(dict)
    with ZipFile(source) as old, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for r in json.loads(old.read("records.json")):
            if r["task"] not in ("aligned_active","aligned_passive"):continue
            designs=[(r["task"],r["text"])]
            if r["task"]=="aligned_active":
                designs.append(("honorific_active",r["text"].replace("    "," Dr ")))
            for name,text in designs:
                ink=pal[[codes[c] for c in text]]
                key=r["i"],r["j"]
                packed,v,c=compact_paper(ink,r["verb"],r["candidates"])
                if name!="aligned_active":
                    signature=(len(packed),tuple(v),tuple(tuple(x) for x in c))
                    if key in layouts:assert signature==layouts[key]
                    layouts[key]=signature
                prefix=f"{r['i']:03d}/{r['j']}/{name}/"
                z.writestr(prefix+"source.png",strip_png(ink));truth=r["target"];choices={}
                for arm in ("raw","normalized","layout"):
                    body=nets["layout" if arm=="layout" else "context"]
                    if arm=="raw":k,_=choose(body,ink,r["verb"],r["candidates"])
                    else:k,_=choose(body,packed,v,c,arm)
                    counts[name][arm]+=int(k==truth);choices[arm]=k
                    joint[key,arm][name]=k==truth
                counts[name]["n"]+=1
                span=r["candidates"][choices["normalized"]]
                z.writestr(prefix+"actor.png",strip_png(ink[slice(*span)]))
                results.append(dict(i=r["i"],j=r["j"],task=name,text=text,choices=choices,target=truth,member=prefix))
        z.writestr("records.json",json.dumps(results))
    tmp.replace(raw)
    both={a:sum(v["aligned_passive"] and v["honorific_active"] for (_,m),v in joint.items() if m==a) for a in ("layout","raw","normalized")}
    dump(PUB/"spacing.json",dict(archive=sha(raw),official=official,changed_official=changes,counts=counts,
          matched_ink_pairs=len(layouts),both_ink=both))
    print(official,changes,dict(counts),both,flush=True)


if __name__=="__main__":main()

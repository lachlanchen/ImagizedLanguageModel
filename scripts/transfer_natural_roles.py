"""Apply the frozen participant reader to a third, independently sourced treebank."""
from collections import Counter
import json
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
from PIL import ImageFont
import requests
import torch
from scripts.prepare_natural_pictured_roles import PUB,ROOT,RUN,RAW,FONT,bitmap,clauses,read,sha,dump
from scripts.read_natural_roles import load,choose,strip_png

REV="6e328e8195c04a297c1e375d21fe663be1ce6178"


def main():
    torch.set_num_threads(2)
    path=RAW/"PUD/en_pud-ud-test.conllu";path.parent.mkdir(exist_ok=True)
    if not path.exists():
        res=requests.get(f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/{REV}/{path.name}",timeout=60)
        res.raise_for_status();tmp=path.with_suffix(".part");tmp.write_bytes(res.content);tmp.replace(path)
    known={r["full"] for s in ("train","dev","test") for r in read(RUN/(s+".json"))}
    exclude=Counter();rows=[]
    for block in path.read_text().split("\n\n"):
        for r in clauses(block,exclude,"PUD"):
            if r["full"] not in known:rows.append(r)
    font=ImageFont.truetype(str(FONT),24)
    chars={c:bitmap(c,font) for r in rows for c in r["text"]}
    nets={a:load(a) for a in ("layout","context")};counts={a:Counter() for a in ("left","layout","context")}
    archive=ROOT/"publication/ilm-nature/evidence/natural_roles_pud.zip"
    tmp=archive.with_suffix(".tmp")
    records=[]
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for i,r in enumerate(rows):
            ink=np.stack([chars[c] for c in r["text"]])
            predictions={"left":0}
            for arm,model in nets.items():
                predictions[arm]=choose(model,ink,r["verb"],r["candidates"],arm)[0]
            for arm,k in predictions.items():
                counts[arm][r["voice"]+"_n"]+=1
                counts[arm][r["voice"]+"_correct"]+=int(k==r["label"])
            z.writestr(f"{i:04d}/source.png",strip_png(ink))
            actor=r["candidates"][predictions["context"]]
            z.writestr(f"{i:04d}/actor.png",strip_png(ink[slice(*actor)]))
            records.append(dict(i=i,sentence=r["sentence"],verb=r["verb"],candidates=r["candidates"],
                                voice=r["voice"],target=r["label"],choices=predictions))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    dump(PUB/"pud.json",dict(revision=REV,raw=sha(path),archive=sha(archive),counts=counts,
         eligible=len(rows),filters=exclude))
    print(counts,flush=True)


if __name__=="__main__":main()

"""Freeze original sources and physically selected heads on the official unseen clauses."""
from collections import Counter,defaultdict
import json
import re
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import torch

from ilm.visual_lm.natural_roles import tiles
from scripts.prepare_natural_pictured_roles import ROOT,MODELS,PUB,RUN,read,dump,sha
from scripts.read_natural_roles import strip_png,load_strip,load,choose
from scripts.train_natural_pictured_roles import arrays
from scripts.test_pictured_order_voice import PARTICIPLE


def template_rows(codes):
    source=ROOT/"publication/ilm-nature/evidence/strip_restoration_voice.zip"
    assert sha(source)==read(ROOT/"references/evidence/visual_strip_restoration_v1/voice.json")["archive"]
    with ZipFile(source) as z:
        tests=json.loads(z.read("records.json"))
    rows=[]
    for r in tests:
        text=r["source"]
        words=list(re.finditer(r"[A-Za-z]+",text))
        heads=[(w.start(),w.end()) for w in (words[0],words[-1])]
        verb=words[1 if r["voice"]=="active" else 2].span()
        assert all(c in codes for c in text)
        rows.append(dict(text=text,verb=verb,candidates=heads,label=int(r["voice"]=="passive"),
                         corpus="role challenge",voice=r["voice"],sentence=str((r["i"],r["j"]))))
    return rows


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    receipt=read(PUB/"corpus.json")
    assert sha(RUN/"test.json")==receipt["split_hashes"]["test"] and sha(MODELS/"palette.npy")==receipt["palette"]
    codes=read(RUN/"codes.json");pal=np.load(MODELS/"palette.npy")
    rows=read(RUN/"test.json")+template_rows(codes)
    raw=ROOT/"publication/ilm-nature/evidence/natural_pictured_roles.zip"
    tmp=raw.with_suffix(".tmp");models={a:load(a) for a in ("layout","context")}
    conditions=("left","layout","context","without_by")
    grades={k:defaultdict(Counter) for k in conditions}
    fororder={k:defaultdict(Counter) for k in conditions}
    records=[]
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for i,r in enumerate(rows):
            ink=pal[[codes[c] for c in r["text"]]]
            original=strip_png(ink)
            source=load_strip(original,len(ink))
            assert np.array_equal(source,ink)
            z.writestr(f"{i:04d}/source.png",original)
            bits=source.copy()
            for match in re.finditer(r"\bby\b",r["text"],flags=re.I):
                bits[match.start():match.end()]=0
            values={"left":0}
            for k in ("layout","context","without_by"):
                model=models["context" if k=="without_by" else k]
                values[k]=choose(model,bits if k=="without_by" else source,r["verb"],r["candidates"],k)[0]
            # Move the candidate labels only. Pixels, mark type, predicate and source order stay fixed.
            flipped,_=choose(models["context"],source,r["verb"],r["candidates"][::-1])
            assert flipped==1-values["context"]
            same_layout=r["candidates"][0][1]<=r["verb"][0]<r["verb"][1]<=r["candidates"][1][0]
            members={}
            for k,pos in values.items():
                key=r["corpus"]+"/"+r["voice"]
                grades[k][key].update(n=1,correct=int(pos==r["label"]))
                if same_layout:fororder[k][key].update(n=1,correct=int(pos==r["label"]))
                if k in models:
                    members[k]=[]
                    for field,span in (("actor",r["candidates"][pos]),("undergoer",r["candidates"][1-pos])):
                        actual=source[slice(*span)]
                        blob=strip_png(actual)
                        assert np.array_equal(load_strip(blob,len(actual)),actual)
                        member=f"{i:04d}/{k}-{field}.png"
                        z.writestr(member,blob);members[k].append(member)
            records.append(dict(i=i,corpus=r["corpus"],sentence=r["sentence"],voice=r["voice"],same_layout=same_layout,
                verb=r["verb"],cells=len(ink),candidates=r["candidates"],target=r["label"],choices=values,members=members))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(raw)
    balanced={m:{c:sum(v[c+"/"+voice]["correct"]/v[c+"/"+voice]["n"] for voice in ("active","passive"))/2
                 for c in ("EWT","GUM","role challenge")} for m,v in grades.items()}
    passed=all(balanced["context"][c]>max(balanced[k][c] for k in ("left","layout"))
         and grades["context"][c+"/passive"]["correct"]>grades["left"][c+"/passive"]["correct"]
         and (grades["left"][c+"/active"]["correct"]-grades["context"][c+"/active"]["correct"])/
         grades["left"][c+"/active"]["n"]<=.10 for c in ("EWT","GUM"))
    dump(PUB/"assessment.json",dict(archive=sha(raw),counts=grades,between_heads=fororder,balanced=balanced,
         passed=passed,independent_input_reversals=len(rows),test=len(read(RUN/"test.json")),template=len(template_rows(codes))))
    print(balanced,passed,flush=True)


if __name__=="__main__":main()

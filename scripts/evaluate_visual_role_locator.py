"""Score open visual head localization on two official tests, PUD and construction transfers."""
from collections import Counter,defaultdict
import json
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import torch
from PIL import ImageFont

from scripts.prepare_natural_pictured_roles import ROOT,RUN,RAW,PUB,FONT,read,dump,sha,bitmap,clauses
from scripts.evaluate_natural_pictured_roles import template_rows
from scripts.locate_pictured_roles import PicturedLocator
from scripts.read_natural_roles import strip_png,load_strip


def main():
    torch.set_num_threads(2)
    corpus=read(PUB/"corpus.json");assert sha(RUN/"test.json")==corpus["split_hashes"]["test"]
    rows=read(RUN/"test.json")
    all_known={r["full"] for s in ("train","dev","test") for r in read(RUN/(s+".json"))}
    pud=RAW/"PUD/en_pud-ud-test.conllu"
    assert sha(pud)==read(PUB/"pud.json")["raw"]
    for block in pud.read_text().split("\n\n"):
        rows.extend(r for r in clauses(block,Counter(),"PUD") if r["full"] not in all_known)
    rows+=template_rows(read(RUN/"codes.json"))
    raw=ROOT/"publication/ilm-nature/evidence/natural_roles_constructions.zip"
    with ZipFile(raw) as z: designs=json.loads(z.read("records.json"))
    for r in designs:
        rows.append(dict(corpus="construction",voice=r["task"],text=r["text"],sentence=str((r["i"],r["j"])),
                         verb=r["verb"],candidates=r["candidates"],label=r["target"]))
    font=ImageFont.truetype(str(FONT),24)
    chars={c:bitmap(c,font) for r in rows for c in r["text"]}
    writers={k:PicturedLocator(k) for k in ("context","position")}
    archive=ROOT/"publication/ilm-nature/evidence/visual_role_localization.zip"
    tmp=archive.with_suffix(".tmp")
    records=[];counts=defaultdict(lambda:defaultdict(Counter))
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for i,r in enumerate(rows):
            original=strip_png(np.stack([chars[c] for c in r["text"]]))
            ink=load_strip(original,len(r["text"]))
            z.writestr(f"{i:04d}/source.png",original)
            expected=tuple(tuple(r["candidates"][k]) for k in (r["label"],1-r["label"]))
            outputs={}
            for arm,writer in writers.items():
                pred=writer.locate(ink,r["verb"])
                pixels=[]
                for tag,span,true in zip(("actor","undergoer"),pred,expected,strict=True):
                    img=ink[slice(*span)]
                    blob=strip_png(img)
                    assert np.array_equal(load_strip(blob,len(img)),img)
                    z.writestr(f"{i:04d}/{arm}-{tag}.png",blob)
                    pixels.append(np.array_equal(img,ink[slice(*true)]))
                counter=counts[arm][r["corpus"]+"/"+r["voice"]]
                counter.update(n=1,exact_spans=int(pred==expected),exact_images=int(all(pixels)),
                               actor=int(pred[0]==expected[0]),undergoer=int(pred[1]==expected[1]))
                outputs[arm]=dict(spans=pred,exact=pred==expected,exact_images=all(pixels))
            records.append(dict(i=i,corpus=r["corpus"],sentence=r["sentence"],voice=r["voice"],cells=len(ink),
                                verb=r["verb"],target=expected,answers=outputs))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    def total(arm,groups):
        return sum(counts[arm][g]["exact_spans"] for g in groups)
    groups=[[c+"/"+v for v in ("active","passive")] for c in ("EWT","GUM")]
    groups += [[c+"/"+v for c in ("EWT","GUM")] for v in ("active","passive")]
    passed=all(total("context",g)>total("position",g) for g in groups)
    dump(PUB/"localization.json",dict(archive=sha(archive),counts=counts,passed=passed,n=len(rows),
        protocol=sha(ROOT/"references/visual_role_localization_protocol.md")))
    print({a:{g:v["exact_spans"] for g,v in vals.items()} for a,vals in counts.items()},passed,flush=True)


if __name__=="__main__":main()

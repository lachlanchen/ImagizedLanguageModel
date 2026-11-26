"""Independently replay physical responses and enumerate the shortest held search spaces."""
import ast
from collections import Counter
import json
from zipfile import ZipFile

import numpy as np
import torch
from ilm.visual_lm.natural_roles import tiles
from scripts.prepare_natural_pictured_roles import PUB,ROOT,read,dump,sha
from scripts.read_natural_roles import strip_png,load_strip
from scripts.locate_pictured_roles import PicturedLocator


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    archive=ROOT/"publication/ilm-nature/evidence/visual_role_localization.zip"
    assert sha(archive)==read(PUB/"localization.json")["archive"]
    models={a:PicturedLocator(a) for a in ("context","position")}
    counts=Counter();pronouns=Counter()
    with ZipFile(ROOT/"publication/ilm-nature/evidence/natural_roles_constructions.zip") as z:
        cases={(r["i"],r["j"],r["task"]):r for r in json.loads(z.read("records.json"))}
    with ZipFile(archive) as z:
        rows=json.loads(z.read("records.json"))
        shortest={r["i"] for r in sorted((r for r in rows if r["corpus"] in ("EWT","GUM","PUD")),
                                           key=lambda r:(r["cells"],r["i"]))[:12]}
        for r in rows:
            i=r["i"];verb=r["verb"]
            ink=load_strip(z.read(f"{i:04d}/source.png"),r["cells"])
            for arm,model in models.items():
                pred=model.locate(ink,verb)
                target=tuple(tuple(v) for v in r["answers"][arm]["spans"])
                assert pred==target
                for tag,span in zip(("actor","undergoer"),pred,strict=True):
                    assert strip_png(ink[slice(*span)])==z.read(f"{i:04d}/{arm}-{tag}.png")
                    counts["png_replayed"]+=1
                if i in shortest:
                    mask=np.any(ink,axis=(1,2));mask[slice(*verb)]=False
                    spans=[(a,b) for a in range(len(ink)) for b in range(a+1,len(ink)+1) if mask[a:b].all()]
                    v=torch.zeros((1,len(ink)),dtype=torch.bool);v[0,slice(*verb)]=True
                    scores=model.model.read(model.model.eye(tiles(torch.from_numpy(ink)))[None],v,torch.ones_like(v),arm=="position")[0].double().numpy()
                    def value(pair):
                        (a,b),(c,d)=pair
                        return scores[0,a]+scores[1,b-1]+scores[2,c]+scores[3,d-1]
                    optimum=max(value((a,b)) for a in spans for b in spans if a[1]<=b[0] or b[1]<=a[0])
                    assert abs(optimum-value(pred))<2e-5
                    counts["exhaustive_optima"]+=1
            if r["corpus"]=="construction" and r["voice"] in ("subject_cleft","object_cleft"):
                x,y=ast.literal_eval(r["sentence"]);text=cases[x,y,r["voice"]]["text"]
                a,b=[text[slice(*s)] for s in r["answers"]["context"]["spans"]]
                pronouns[r["voice"]]+=int(a=="who" if r["voice"]=="subject_cleft" else b=="whom")
    dump(PUB/"locator-replay.json",dict(archive=sha(archive),cases=len(rows),counts=counts,relative_pronouns=pronouns))
    print(counts,pronouns,flush=True)


if __name__=="__main__":main()

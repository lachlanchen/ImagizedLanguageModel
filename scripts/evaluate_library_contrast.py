"""Measure the fixed context-ratio diagnostic on unassisted visual translations."""

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from time import monotonic
from hashlib import sha256
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.background_contrast import write
from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.evaluate_visual_library import RUN,EVIDENCE,MODELS,ROOT,BASE,read,dump,sha,material,pack,stopped,semantic,blind_votes
from scripts.write_ordered_translation import restore
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


@torch.inference_mode()
def generate(rows, originals, gain):
    table=torch.as_tensor(np.load(MODELS/"broad/backgrounds.npy"),device="cuda",dtype=torch.long)
    net=restore("library-broad","cuda")
    palette=np.load(BASE/"palette.npy")
    ids={bytes(c):i for i,c in enumerate(palette) if i>=2}
    lookup=read(BASE/"readback.json")
    actual=[]
    for i,r in enumerate(rows):
        x=read_image(RUN/"pixels"/f"{i:04d}"/"input.png",len(r["input"]))
        actual.append([ids[bytes(c)] for c in x])
        if actual[-1]!=r["input"]:
            raise ValueError("Source image changed")
    features,invs,_=material("cuda")
    out=[{} for _ in rows]
    times={}
    for arm,a in (("ordinary",0.),("contrast",gain)):
        torch.cuda.synchronize()
        start=monotonic()
        for t in (0,1):
            group=[i for i,r in enumerate(rows) if r["t"]==t]
            for j in range(0,len(group),24):
                ix=group[j:j+24]
                seq=write(net,features,pack([actual[i] for i in ix],"cuda"),invs[t],t,table[t],a).cpu()
                for i,y in zip(ix,seq,strict=True):
                    cells,stop=stopped(y)
                    raw=palette[np.asarray(cells,dtype=int)].tobytes()
                    folder=RUN/"pixels"/f"{i:04d}"
                    if arm=="ordinary":
                        path=folder/"broad.png"
                        if cells!=originals[i]["broad"]["cells"] or stop!=originals[i]["broad"]["stop"] or sha(path)!=originals[i]["broad"]["png"]:
                            raise ValueError(f"Original output differs for {i}")
                    else:
                        path=folder/"contrast.png"
                        save_grid(raw,path)
                    out[i][arm]=dict(cells=cells,stop=stop,packed=sha256(raw).hexdigest(),png=sha(path),
                                     text="".join(lookup[str(v)] for v in cells),
                                     exact=stop and cells==rows[i]["target"],
                                     similarity=stop*Levenshtein.normalized_similarity(cells,rows[i]["target"]))
        torch.cuda.synchronize()
        times[arm]=monotonic()-start
    return out,times


def summarize(rows,answers,votes):
    works=defaultdict(Counter)
    records=[]
    for i,(r,ans) in enumerate(zip(rows,answers,strict=True)):
        ok={m:v["stop"] and votes[2*i][m] and votes[2*i+1][m] for m,v in ans.items()}
        works[r["doc"]].update(ok)
        works[r["doc"]]["n"]+=1
        records.append(ok)
    d=[(v["contrast"]-v["ordinary"])/v["n"] for v in works.values()]
    ci=np.quantile([np.mean(a) for a in product(d,repeat=4)],[.025,.975]).tolist()
    return dict(works=works,total=sum(works.values(),Counter()),paired_95pct=ci,records=records)


def main():
    configure_cuda_cli()
    source=RUN/"generated.json"
    original=read(source)
    rows=original["rows"]
    calibration=MODELS/"broad/contrast.json"
    selected=read(calibration)["selected"]
    gain=selected or .5
    raw=RUN/"contrast-generated.json"
    guided=RUN/"contrast-reference.json"
    blind=RUN/"contrast-source.json"
    panel=[240*j+i for j in range(4) for i in range(24,48)]
    assert not set(panel).intersection(original["panel"])
    binding=dict(source=sha(source),program=sha(Path(__file__)),calibration=sha(calibration),
                 algorithm=sha(ROOT/"ilm/visual_lm/background_contrast.py"))
    with gpu_owner():
        if not raw.exists():
            out,times=generate(rows,original["answers"],gain)
            dump(raw,dict(answers=out,wall_seconds=times,binding=binding))
        gen=read(raw)
        if gen["binding"]!=binding:
            raise ValueError("Changed diagnostic")
        rr=[rows[i] for i in panel]
        if not guided.exists():
            a=[dict(gen["answers"][i]) for i in panel]
            votes=semantic(rr,a,("ordinary","contrast"))["votes"]
            dump(guided,dict(source=sha(raw),answers=a,votes=votes))
        g=read(guided)
        if not blind.exists():
            dump(blind,dict(source=sha(guided),votes=blind_votes(rr,g["answers"])))
        b=read(blind)
        if g["source"]!=sha(raw) or b["source"]!=sha(guided):
            raise ValueError("Assessment changed")
    out=gen["answers"]
    stats={}
    for m in ("ordinary","contrast"):
        stats[m]=dict(exact=sum(r[m]["exact"] for r in out[:960]),stopped=sum(r[m]["stop"] for r in out[:960]),
                      similarity=float(np.mean([r[m]["similarity"] for r in out[:960]])),
                      role_families=sum(all(a[m]["exact"] for a in out[j:j+8]) for j in range(960,len(rows),8)))
    result=dict(binding=binding,selected_gain=selected,evaluated_gain=gain,panel=panel,
                optical=stats,wall_seconds=gen["wall_seconds"],
                changed=sum(r["ordinary"]["cells"]!=r["contrast"]["cells"] for r in out[:960]),
                source_only=summarize(rr,g["answers"],b["votes"]),reference_informed=summarize(rr,g["answers"],g["votes"]),
                files={p.name:sha(p) for p in (raw,guided,blind)})
    dump(EVIDENCE/"contrast.json",result)
    print(stats,result["source_only"]["total"],result["reference_informed"]["total"],flush=True)


if __name__=="__main__":
    main()

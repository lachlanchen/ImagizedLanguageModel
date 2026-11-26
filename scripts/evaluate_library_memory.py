"""Test associative optical writing and its bigram-preserving shuffled control."""

from collections import Counter,defaultdict
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import monotonic
import numpy as np
from rapidfuzz.distance import Levenshtein
import torch
from ilm.visual_lm.glyph_association import GlyphMemory,generate
from ilm.visual_lm.portable_writer import save_grid,read_image
from scripts.build_library_memory import ROOT,RUN,PARENT,PUB,read,sha,dump
from scripts.evaluate_visual_library import BASE,semantic,blind_votes,pack,stopped,material
from scripts.train_span_choice import gpu_owner
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli


@torch.inference_mode()
def write_all(rows,baseline,calibration):
    palette=np.load(BASE/"palette.npy")
    ids={bytes(v):i for i,v in enumerate(palette) if i>=2}
    lut=read(BASE/"readback.json")
    net=restore("library-broad","cuda")
    features,inv,_=material("cuda")
    mix=calibration["selected"]["mix"]
    temp=calibration["selected"]["temperature"] if mix else .05
    applied=mix or .25
    inputs=[]
    for i,r in enumerate(rows):
        file=PARENT/"pixels"/f"{i:04d}"/"input.png"
        src=[ids[bytes(c)] for c in read_image(file,len(r["input"]))]
        if src!=r["input"]:
            raise ValueError("Source grid differs")
        inputs.append(src)
    out=[dict(ordinary=v["broad"]) for v in baseline]
    elapsed=Counter()
    for t in (0,1):
        p=RUN/f"memory-{t}.npz"
        if sha(p)!=calibration["files"][p.name]:
            raise ValueError("Memory digest differs")
        with np.load(p) as archive:
            memory=GlyphMemory(archive,"cuda")
        indices=[i for i,r in enumerate(rows) if r["t"]==t]
        for m in ("associative","shuffled"):
            torch.cuda.synchronize()
            start=monotonic()
            for j in range(0,len(indices),24):
                ix=indices[j:j+24]
                seq=generate(net,features,pack([inputs[i] for i in ix],"cuda"),inv[t],t,
                             memory,temp,applied,shuffled=m=="shuffled").cpu()
                for i,y in zip(ix,seq,strict=True):
                    cells,stop=stopped(y)
                    ink=palette[np.asarray(cells,dtype=int)].tobytes()
                    folder=RUN/"pixels"/f"{i:04d}"
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/(m+".png")
                    save_grid(ink,dest)
                    out[i][m]=dict(cells=cells,stop=stop,packed=sha256(ink).hexdigest(),png=sha(dest),
                                   text="".join(lut[str(v)] for v in cells),exact=stop and cells==rows[i]["target"],
                                   similarity=stop*Levenshtein.normalized_similarity(cells,rows[i]["target"]))
            torch.cuda.synchronize()
            elapsed[m]+=monotonic()-start
        del memory
    return out,elapsed


def review(rows,ans,votes):
    docs=defaultdict(Counter)
    records=[]
    for i,(r,a) in enumerate(zip(rows,ans,strict=True)):
        ok={m:o["stop"] and votes[i*2][m] and votes[i*2+1][m] for m,o in a.items()}
        docs[r["doc"]].update(ok)
        docs[r["doc"]]["n"]+=1
        records.append(ok)
    intervals={}
    for m in ("ordinary","shuffled"):
        ds=[(v["associative"]-v[m])/v["n"] for v in docs.values()]
        intervals[m]=np.quantile([np.mean(a) for a in product(ds,repeat=4)],[.025,.975]).tolist()
    return dict(total=sum(docs.values(),Counter()),works=docs,records=records,paired_95pct=intervals)


def main():
    configure_cuda_cli()
    cal=PUB/"calibration.json"
    primary=PARENT/"generated.json"
    original=read(primary)
    rows=original["rows"]
    panel=[j*240+i for j in range(4) for i in range(48,72)]
    if set(panel)&(set(original["panel"])|set(read(ROOT/"references/evidence/library_diversity_v1/contrast.json")["panel"])):
        raise ValueError("Repeated semantic assessment")
    binding=dict(calibration=sha(cal),program=sha(Path(__file__)),baseline=sha(primary))
    raw,ref,source=[RUN/f for f in ("generated.json","reference.json","source.json")]
    with gpu_owner():
        if not raw.exists():
            out,times=write_all(rows,original["answers"],read(cal))
            dump(raw,dict(binding=binding,answers=out,elapsed=times))
        gen=read(raw)
        if gen["binding"]!=binding:
            raise ValueError("Generation changed")
        rr=[rows[i] for i in panel]
        if not ref.exists():
            candidates=[dict(gen["answers"][i]) for i in panel]
            torch.cuda.empty_cache()
            guided=semantic(rr,candidates,("ordinary","associative","shuffled"))
            dump(ref,dict(raw=sha(raw),answers=candidates,votes=guided["votes"]))
        r=read(ref)
        if r["raw"]!=sha(raw):
            raise ValueError("Wrong reference assessment")
        if not source.exists():
            dump(source,dict(ref=sha(ref),votes=blind_votes(rr,r["answers"])))
        b=read(source)
        if b["ref"]!=sha(ref):
            raise ValueError("Wrong source assessment")
    out=gen["answers"]
    optical={}
    for m in ("ordinary","associative","shuffled"):
        optical[m]=dict(exact=sum(a[m]["exact"] for a in out[:960]),stopped=sum(a[m]["stop"] for a in out[:960]),
                        similarity=float(np.mean([a[m]["similarity"] for a in out[:960]])),
                        role_families=sum(all(a[m]["exact"] for a in out[j:j+8]) for j in range(960,len(rows),8)))
    result=dict(binding=binding,files={f.name:sha(f) for f in (raw,ref,source)},panel=panel,optical=optical,
                elapsed=gen["elapsed"],source_only=review(rr,r["answers"],b["votes"]),
                reference_informed=review(rr,r["answers"],r["votes"]))
    dump(PUB/"evaluation.json",result)
    print(optical,result["source_only"]["total"],result["reference_informed"]["total"],flush=True)


if __name__=="__main__":
    main()

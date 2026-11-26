"""Measure whether a nearby writing state contains the next complete lexical run."""

from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.build_library_memory import RUN,PUB,ROOT,MODELS,read,sha,dump,library,hidden,material,pack,gpu_owner,restore,configure_cuda_cli
from scripts.prepare_teacher_translation import BASE
from ilm.visual_lm.glyph_association import GlyphMemory


def requests(cells,t,lookup):
    chars=[lookup[str(c)] for c in cells]
    if t:
        for j in range(len(cells)-3):
            if all("\u3400"<=c<="\u9fff" for c in chars[j:j+4]):
                yield j,4
    else:
        letters=[c.isascii() and c.isalpha() for c in chars]
        for j,letter in enumerate(letters):
            if letter and (j==0 or not letters[j-1]):
                k=j+1
                while k<len(cells) and letters[k]:
                    k+=1
                if k-j>=3:
                    yield j,k-j


@torch.inference_mode()
def probe(net,bank,t,dev,train,memory,lookup):
    rng=np.random.default_rng(48133+t)
    counts=Counter()
    cases=[]
    for j in range(0,len(dev),12):
        rr=dev[j:j+12]
        x=pack([r["cells"][1-t] for r in rr],"cuda")
        y=pack([r["cells"][t]+[1] for r in rr],"cuda")
        h,s=hidden(net,bank,x,y,t)
        selected=(s.float()@memory.sources.float().T).topk(32,dim=1).indices
        keys,_,_,valid=memory.passages(s)
        distances=(F.normalize(h,dim=-1).to(keys.dtype)@keys.transpose(1,2)).float()
        local=distances.masked_fill(~valid[:,None],-float("inf")).topk(16,dim=-1).indices.cpu().numpy()
        width=keys.size(1)//32
        for k,r in enumerate(rr):
            indices=selected[k].cpu().tolist()
            options=[train[i]["cells"][t] for i in indices]
            random=[train[i]["cells"][t] for i in rng.choice(len(train),32,replace=False)]
            # Two-byte packed glyphs are searched only on cell boundaries.
            anyruns={}
            target=r["cells"][t]
            row=Counter()
            for pos,n in requests(target,t,lookup):
                desired=tuple(target[pos:pos+n])
                if n not in anyruns:
                    anyruns[n]=[{tuple(a[z:z+n]) for a in seq for z in range(len(a)-n+1)} for seq in (options,random)]
                oracle=any(tuple(options[u//width][u%width:u%width+n])==desired for u in local[k,pos])
                in_pages=desired in anyruns[n][0]
                if oracle and not in_pages:
                    raise RuntimeError("Retrieved state exceeded page support")
                row.update(n=1,states=int(oracle),pages=int(in_pages),random=int(desired in anyruns[n][1]))
            if row["n"]:
                row["requests"]=1
                row["all_states"]=int(row["n"]==row["states"])
                row["all_pages"]=int(row["n"]==row["pages"])
            counts.update(row)
            cases.append(dict(index=j+k,**row))
    return dict(counts=counts,cases=cases)


def main():
    configure_cuda_cli()
    record=read(PUB/"calibration.json")
    train,dev=library("train"),library("dev")
    lookup=read(BASE/"readback.json")
    results={}
    with gpu_owner(),torch.inference_mode():
        net=restore("library-broad","cuda")
        bank=net.optical(material("cuda")[0])
        for t,name in enumerate(("english_word","four_han")):
            p=RUN/f"memory-{t}.npz"
            if sha(p)!=record["files"][p.name]:
                raise ValueError("Memory changed")
            with np.load(p) as archive:
                mem=GlyphMemory(archive,"cuda")
            results[name]=probe(net,bank,t,dev,train,mem,lookup)
            del mem
    dump(PUB/"phrase-support.json",dict(groups=results,calibration=sha(PUB/"calibration.json"),
         protocol=sha(ROOT/"references/library_memory_phrase_protocol.md"),program=sha(Path(__file__))))
    print({k:v["counts"] for k,v in results.items()})


if __name__=="__main__":
    main()

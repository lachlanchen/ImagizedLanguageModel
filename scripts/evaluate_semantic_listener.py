"""Compare likelihood and visual readback using exactly the same completed candidates."""
from pathlib import Path
from time import monotonic
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from scripts.evaluate_semantic_page import realization_patterns,claims
from scripts.train_semantic_page import WORK,PUB,MODEL,read,archive,palette,load,gpu_owner,snapshot,consumed,sha,dump
from scripts.train_semantic_listener import load_listener


def canonical(facts):
    equivalents={
        "customer rating":{"1 out of 5":"low","3 out of 5":"average","5 out of 5":"high"},
        "priceRange":{"less than £20":"cheap","£20-25":"moderate","more than £30":"high"},
    }
    return {(k,equivalents.get(k,{}).get(v,v)) for k,v in facts}


def metrics(lines, gold, patterns):
    claimed=[claims(s,patterns) for s in lines]
    return dict(count=len(lines),strict=sum(p==g for p,g in zip(claimed,gold)),
                equivalent=sum(canonical(p)==canonical(g) for p,g in zip(claimed,gold)),
                missing=sum(len(g-p) for p,g in zip(claimed,gold)),
                extra=sum(len(p-g) for p,g in zip(claimed,gold)))


def run():
    original=read(WORK/"generation.json")
    meta=read(PUB/"corpus.json")
    schema=read(WORK/"schema.json")
    corpus=archive(WORK/"test.npz",meta["archives"]["test"])
    rows=corpus["slots"][original["first"]]
    concept=[tuple(v) if v else None for v in schema["concepts"]]
    families=[concept.index(("familyFriendly",v)) for v in ("no","yes")]
    pairs=np.asarray(original["original"])
    flip=rows[pairs].copy()
    no,yes=flip==families[0],flip==families[1]
    flip[no],flip[yes]=families[1],families[0]
    inputs=np.concatenate((rows,flip))
    gold=[{concept[v] for v in row if v} for row in inputs]
    bank=archive(WORK/"bank.npz",meta["bank"])
    chars=read(MODEL/"readback.json")
    training=set(read(WORK/"train-refs.json"))
    patterns=realization_patterns()
    allstats={}
    n=len(rows)
    def score(lines):
        inferred=[claims(s,patterns) for s in lines]
        return dict(original=metrics(lines[:n],gold[:n],patterns),flipped=metrics(lines[n:],gold[n:],patterns),
            novel=sum(s not in training for s in lines[:n]),
            exchanged=sum(x==g for x,g in zip(inferred[:n],gold[1:n]+gold[:1])),
            both=sum(inferred[i]==gold[i] and inferred[n+j]==gold[n+j] for j,i in enumerate(pairs)),
            polarity_both=sum({v for k,v in inferred[i] if k=="familyFriendly"} ==
                              {v for k,v in gold[i] if k=="familyFriendly"} and
                              {v for k,v in inferred[n+j] if k=="familyFriendly"} ==
                              {v for k,v in gold[n+j] if k=="familyFriendly"} for j,i in enumerate(pairs)))
    with gpu_owner(), torch.inference_mode():
        glyphs=palette("cuda");listener=load_listener("cuda")
        for mode in ("ink","byte"):
            net=load(mode,"cuda")
            obs=torch.as_tensor(observations(bank,mode),device="cuda")
            features=obs[bank["surface"]]
            keys=net.encode_rows(features.flatten(0,1)).reshape(2,features.shape[1],-1)
            for lang,title in enumerate(("en","zh")):
                name=mode+"-"+title; path=WORK/(name+"-beam.json")
                if path.exists():
                    result=read(path)
                    if result["listener"]!=sha(MODEL/"listener.safetensors") or result["parent"]!=sha(MODEL/(mode+".safetensors")):
                        raise ValueError("Different stored candidate model")
                else:
                    start=monotonic();energy=snapshot()
                    result={"beam":[],"readback":[]}
                    for j in range(0,len(inputs),32):
                        batch=torch.as_tensor(inputs[j:j+32],device="cuda").long()
                        generated=beam_write(net,keys[lang,batch],batch!=0,glyphs,listener)
                        for key,(ids,ns) in zip(("beam","readback"),generated):
                            result[key].extend("".join(chars[c] for c in seq[:size])
                                               for seq,size in zip(ids.cpu().tolist(),ns.tolist()))
                    result.update(seconds=monotonic()-start,energy=consumed(energy,snapshot()),
                                  parent=sha(MODEL/(mode+".safetensors")),listener=sha(MODEL/"listener.safetensors"))
                    dump(path,result)
                allstats[name]={key:score(lines) for key,lines in (
                    ("greedy",original["saved"][name]["lines"]),("beam",result["beam"]),("readback",result["readback"]))}
                allstats[name]["generation"]={k:result[k] for k in ("seconds","energy")}
                print(name,allstats[name],flush=True)
    human=[{concept[v] for v in row if v} for row in corpus["slots"]]
    allstats["human"]=metrics(read(WORK/"test-refs.json"),human,patterns)
    dump(PUB/"listener-results.json",allstats)


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    run()

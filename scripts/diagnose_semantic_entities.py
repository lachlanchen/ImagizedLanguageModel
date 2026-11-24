"""Resolve the prior eight-candidate coverage ceiling into name and other facts."""
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from scripts.evaluate_semantic_listener import canonical,claims,realization_patterns
from scripts.train_semantic_page import WORK,PUB,MODEL,read,archive,palette,load,gpu_owner,dump
from scripts.train_semantic_listener import load_listener


def diagnose():
    meta=read(PUB/"corpus.json")
    bank=archive(WORK/"bank.npz",meta["bank"])
    rows=archive(WORK/"test.npz",meta["archives"]["test"])["slots"][read(WORK/"generation.json")["first"]]
    expected=read(WORK/"ink-zh-beam.json")["readback"]
    labels=read(WORK/"schema.json")["concepts"]
    chars=read(MODEL/"readback.json")
    patterns=realization_patterns()
    counts=dict(complete=0,names_only_barrier=0,other_only_barrier=0,both_barriers=0,incompatible_candidates=0)
    with gpu_owner(),torch.inference_mode():
        net=load("ink","cuda");listener=load_listener("cuda");out=palette("cuda")
        surface=torch.as_tensor(observations(bank,"ink"),device="cuda")[bank["surface"]]
        keys=net.encode_rows(surface.flatten(0,1)).reshape(2,surface.shape[1],-1)[1]
        for j in range(0,len(rows),32):
            batch=torch.as_tensor(rows[j:j+32],device="cuda").long()
            ids,ns,_,chosen=beam_write(net,keys[batch],batch!=0,out,listener,return_candidates=True)
            for i,(seq,length,pick) in enumerate(zip(ids.tolist(),ns.tolist(),chosen.tolist())):
                texts=["".join(chars[c] for c in s[:n]) for s,n in zip(seq,length)]
                if texts[pick]!=expected[j+i]:raise RuntimeError("Cannot reproduce the original selection")
                truth=canonical({tuple(labels[c]) for c in rows[j+i] if c})
                observed=[canonical(claims(t,patterns)) for t in texts]
                if truth in observed:
                    counts["complete"]+=1
                    continue
                def only(facts,names):
                    return {x for x in facts if (x[0] in {"name","near"})==names}
                name_ok=any(only(v,True)==only(truth,True) for v in observed)
                other_ok=any(only(v,False)==only(truth,False) for v in observed)
                label={(False,True):"names_only_barrier",(True,False):"other_only_barrier",
                       (False,False):"both_barriers",(True,True):"incompatible_candidates"}[name_ok,other_ok]
                counts[label]+=1
    if counts["complete"]!=1668 or sum(counts.values())!=1847:
        raise RuntimeError("Original coverage count changed")
    dump(PUB.parent/"variable_ink_v1"/"prior_coverage.json",counts)
    print(counts)


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    diagnose()

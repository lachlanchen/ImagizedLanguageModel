"""Score the same eight fixed Chinese-card completions with the listener switched off."""
import numpy as np
import torch
from scripts.evaluate_semantic_listener import claims,canonical,realization_patterns
from scripts.train_semantic_page import WORK,PUB,MODEL,archive,load,palette,gpu_owner,read,dump
from scripts.train_semantic_listener import load_listener
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write


def run():
    expected=read(WORK/"ink-zh-beam.json")
    meta=read(PUB/"corpus.json"); plan=read(WORK/"generation.json")
    rows=archive(WORK/"test.npz",meta["archives"]["test"])["slots"][plan["first"]]
    bank=archive(WORK/"bank.npz",meta["bank"])
    meanings=read(WORK/"schema.json")["concepts"]; chars=read(MODEL/"readback.json")
    patterns=realization_patterns()
    groups={key:[] for key in ("beam","mean_only","readback","any_of_eight")}
    with gpu_owner(),torch.inference_mode():
        reader=load("ink","cuda"); listener=load_listener("cuda"); out=palette("cuda")
        ink=torch.as_tensor(observations(bank,"ink"),device="cuda")
        surface=ink[bank["surface"]]
        keys=reader.encode_rows(surface.flatten(0,1)).reshape(2,surface.shape[1],-1)[1]
        for j in range(0,len(rows),32):
            facts=torch.as_tensor(rows[j:j+32],device="cuda").long()
            ids,n,mean,best=beam_write(reader,keys[facts],facts!=0,out,listener,return_candidates=True)
            for i,(cands,ns,plain,chosen) in enumerate(zip(ids.tolist(),n.tolist(),mean.argmax(1).tolist(),best.tolist())):
                gold={tuple(meanings[v]) for v in rows[j+i] if v}
                text=["".join(chars[c] for c in row[:length]) for row,length in zip(cands,ns)]
                if text[0]!=expected["beam"][j+i] or text[chosen]!=expected["readback"][j+i]:
                    raise ValueError("Original generation changed")
                claims_by=[claims(s,patterns) for s in text]
                hits=np.array([[p==gold,canonical(p)==canonical(gold)] for p in claims_by])
                for label,k in (("beam",0),("mean_only",plain),("readback",chosen)):
                    groups[label].append(hits[k])
                groups["any_of_eight"].append(hits.any(0))
    data={k:np.asarray(v) for k,v in groups.items()}
    results={k:dict(strict=int(v[:,0].sum()),equivalent=int(v[:,1].sum())) for k,v in data.items()}
    on=data["readback"];off=data["mean_only"]
    results["paired"]=dict(gain=(on&~off).sum(0).tolist(),loss=(off&~on).sum(0).tolist(),n=len(rows))
    dump(PUB/"same_beam.json",results)
    print(results)


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    run()

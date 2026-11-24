"""Re-score the precise subset where an absent field changes the action space."""
import json
import gzip
import numpy as np
import torch
from scripts.evaluate_variable_ink import Decoder,plan_text,summaries,realization_patterns,ROLES
from scripts.train_variable_ink import ORIGINAL,EVIDENCE,PUB,BASE
from scripts.train_semantic_page import archive,read,dump,gpu_owner


def main():
    data=read(ORIGINAL/"generation.json")
    label=read(ORIGINAL/"schema.json")["concepts"]
    corpus=archive(ORIGINAL/"test.npz",read(EVIDENCE/"corpus.json")["archives"]["test"])
    cards=corpus["slots"][data["first"]]; pair=data["original"]; edited=cards[pair].copy()
    y,n=[label.index(["familyFriendly",v]) for v in ("yes","no")]
    yes,no=edited==y,edited==n;edited[yes],edited[no]=n,y
    rows=np.concatenate((cards,edited))
    values=[dict(label[i] for i in row if i and label[i][0] in ROLES) for row in rows]
    index=np.array([i for i,v in enumerate(values) if len(v)<2])
    before=json.loads(gzip.decompress((PUB/"sentences.json.gz").read_bytes()))
    chars=read(BASE/"readback.json");patterns=realization_patterns();results={}
    outputs={}
    with gpu_owner(),torch.inference_mode():
        for mode in ("ink","byte"):
            model=Decoder("variable",mode)
            for lang,suffix in enumerate(("en","zh")):
                originals=before[f"variable-{mode}-{suffix}"];after=originals.copy()
                for i,ids in zip(index,model.decode(rows[index],lang,legal=True)):
                    after[i]=plan_text(ids,chars,values[i])
                if None in after:raise ValueError("Illegal field action survived")
                key=mode+"-"+suffix
                results[key]=summaries(after,rows,pair,label,patterns)|dict(
                    redecoded=len(index),changed_original=sum(s!=t for s,t in zip(originals[:len(cards)],after[:len(cards)])))
                outputs[key]=after
    dump(PUB/"legal.json",results)
    (PUB/"legal-sentences.json.gz").write_bytes(gzip.compress(json.dumps(outputs,ensure_ascii=False).encode(),mtime=0))
    print(results)


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()

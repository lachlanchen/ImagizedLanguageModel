"""Evaluate source-span generation and a frozen surrogate-name control on locked cards."""
import gzip
import json
import re
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from ilm.visual_lm.variable_ink import ROLES
from scripts.train_variable_ink import PUB,WORK,MODEL,ORIGINAL,EVIDENCE,BASE,load_writer,visual_palette
from scripts.train_semantic_page import read,archive,palette,load,sha,dump,gpu_owner,snapshot,consumed
from scripts.train_semantic_listener import load_listener
from scripts.evaluate_semantic_listener import canonical,realization_patterns,claims

SURROGATE={"name":"Blue Spice","near":"Café Rouge"}


def plan_text(ids,chars,values):
    tokens=[]
    for i in ids:
        if i<len(chars):
            tokens.append(chars[i])
        else:
            role=ROLES[i-len(chars)]
            if role not in values:return None
            tokens.append(values[role])
    return "".join(tokens)


def substitute(line,values):
    # A single pass prevents one replacement value from triggering the other.
    mapping={s.casefold():k for k,s in SURROGATE.items()}
    pattern=r"(?<!\w)(?:"+"|".join(map(re.escape,SURROGATE.values()))+r")(?!\w)"
    for m in re.finditer(pattern,line,re.I):
        if mapping[m.group().casefold()] not in values:return None
    return re.sub(pattern,lambda m:values[mapping[m.group().casefold()]],line,flags=re.I)


def summaries(lines,rows,pairs,concepts,patterns):
    gold=[{tuple(concepts[i]) for i in row if i} for row in rows]
    pred=[claims(t,patterns) if t is not None else set() for t in lines]
    truth=[canonical(g) for g in gold]
    found=[canonical(v) for v in pred]
    n=len(rows)-len(pairs)
    strict=np.asarray([s is not None and v==g for s,v,g in zip(lines,pred,gold)])
    merged=np.asarray([s is not None and v==g for s,v,g in zip(lines,found,truth)])
    entity=lambda v:{f for f in v if f[0] in ROLES}
    polar=lambda v:{f for f in v if f[0]=="familyFriendly"}
    return dict(n=n,strict=int(strict[:n].sum()),merged=int(merged[:n].sum()),invalid=lines[:n].count(None),
                entity=sum(entity(v)==entity(g) for v,g in zip(found[:n],truth[:n])),
                other=sum(v-entity(v)==g-entity(g) for v,g in zip(found[:n],truth[:n])),
                pair_count=len(pairs),both_full=int((merged[np.asarray(pairs)]&merged[n:]).sum()),
                polarity_both=sum(polar(pred[i])==polar(gold[i]) and polar(pred[n+j])==polar(gold[n+j])
                                  for j,i in enumerate(pairs)))


class Decoder:
    def __init__(self,owner,mode):
        self.bound=owner=="variable"
        base=WORK if self.bound else ORIGINAL
        meta=read((PUB if self.bound else EVIDENCE)/"corpus.json")
        bank=archive(base/"bank.npz",meta["bank"])
        self.writer=load_writer(mode,"cuda") if self.bound else load(mode,"cuda")
        self.listener=load_listener("cuda",PUB,MODEL) if self.bound else load_listener("cuda")
        self.out=visual_palette("cuda") if self.bound else palette("cuda")
        surface=torch.as_tensor(observations(bank,mode),device="cuda")[bank["surface"]]
        self.keys=self.writer.encode_rows(surface.flatten(0,1)).reshape(2,surface.shape[1],-1)
        schema=read(ORIGINAL/"schema.json")["concepts"]
        self.map=np.arange(len(schema))
        if self.bound:
            self.map=np.asarray(read(WORK/"schema.json")["mapping"])
        else:
            for i,c in enumerate(schema):
                if c and c[0] in ROLES:self.map[i]=schema.index([c[0],SURROGATE[c[0]]])
        self.chars=read(BASE/"readback.json")

    def decode(self,rows,lang,legal=False):
        mapped=self.map[rows]
        # Variable and surrogate writers are invariant to spelling. Avoid duplicate neural work.
        unique,reverse=np.unique(mapped,axis=0,return_inverse=True)
        outputs=[]
        for j in range(0,len(unique),32):
            batch=torch.as_tensor(unique[j:j+32],device="cuda")
            allowed=None
            if legal:
                allowed=torch.ones((len(batch),len(self.out)),dtype=torch.bool,device=batch.device)
                for k,r in enumerate(ROLES):
                    c=next(i for i,v in enumerate(read(WORK/"schema.json")["concepts"]) if v and v[0]==r)
                    allowed[:,-len(ROLES)+k]=(batch==c).any(1)
            _,(ids,ns)=beam_write(self.writer,self.keys[lang,batch],batch!=0,self.out,self.listener,allowed=allowed)
            outputs.extend([r[:n] for r,n in zip(ids.tolist(),ns.tolist())])
        return [outputs[i] for i in reverse]


def run():
    orig=read(ORIGINAL/"generation.json");meta=read(EVIDENCE/"corpus.json")
    concepts=read(ORIGINAL/"schema.json")["concepts"]
    corpus=archive(ORIGINAL/"test.npz",meta["archives"]["test"])
    base=corpus["slots"][orig["first"]];pairs=orig["original"]
    yes,no=[concepts.index(["familyFriendly",s]) for s in ("yes","no")]
    flips=base[pairs].copy()
    a,b=flips==yes,flips==no;flips[a],flips[b]=no,yes
    rows=np.concatenate((base,flips))
    values=[dict(concepts[i] for i in row if i and concepts[i][0] in ROLES) for row in rows]
    patterns=realization_patterns();results={};saved={}
    with gpu_owner(),torch.inference_mode():
        for owner,mode in (("variable","ink"),("variable","byte"),("surrogate","ink")):
            decoder=Decoder(owner,mode)
            for lg,lang in enumerate(("en","zh")):
                key="-".join((owner,mode,lang));start=monotonic();power=snapshot()
                acts=decoder.decode(rows,lg)
                lines=[plan_text(s,decoder.chars,v) if decoder.bound else substitute("".join(decoder.chars[i] for i in s),v)
                       for s,v in zip(acts,values)]
                cost=dict(seconds=monotonic()-start,energy=consumed(power,snapshot()))
                results[key]=summaries(lines,rows,pairs,concepts,patterns)|{"generation":cost};saved[key]=lines
                print(key,results[key],flush=True)
    dump(PUB/"results.json",results)
    (PUB/"sentences.json.gz").write_bytes(gzip.compress(json.dumps(saved,ensure_ascii=False).encode(),mtime=0))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    run()

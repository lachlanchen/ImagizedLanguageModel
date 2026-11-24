"""Generate complete descriptions and independently read back the expressed facts."""
import ast
from collections import Counter,defaultdict
import json
import re
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_semantic_page import WORK,PUB,MODEL,RAW,sha,dump
from scripts.train_semantic_page import read,archive,palette,load,gpu_owner,snapshot,consumed

NAMES={"eat_type":"eatType","price_range":"priceRange","rating":"customer rating","family_friendly":"familyFriendly"}


def realization_patterns():
    path=RAW/"slot_error.py"
    if sha(path)!=read(PUB/"corpus.json")["files"][path.name]:
        raise ValueError("Changed pattern source")
    assignment=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.Assign) and
                    isinstance(x.targets[0],ast.Name) and x.targets[0].id=="REALIZATIONS")
    rules=ast.literal_eval(assignment.value)
    patterns=[]
    for key,values in rules.items():
        if isinstance(values,list):
            values={v:[re.escape(v)] for v in values}
        for val,items in values.items():
            expression="|".join(("" if s.startswith("^") else r"\b")+"(?:"+s+")"+("" if s.endswith("$") else r"\b") for s in items)
            patterns.append((NAMES.get(key,key),val,re.compile(expression,re.IGNORECASE)))
    return patterns


def claims(line,patterns):
    matches=[(m.start(),m.end(),(key,v)) for key,v,p in patterns for m in p.finditer(line)]
    return {fact for a,b,fact in matches if not any(c<=a and b<=d and (a,b)!=(c,d) for c,d,_ in matches)}


@torch.inference_mode()
def write(net,mode,bank,slots,langs,output):
    obs=torch.as_tensor(observations(bank,mode),device="cuda")
    images=obs[bank["surface"]]
    keys=net.encode_rows(images.flatten(0,1)).reshape(2,images.shape[1],-1)
    results=[];stops=[]
    for j in range(0,len(slots),64):
        rows=torch.tensor(slots[j:j+64].astype(np.int64),device="cuda")
        lang=torch.as_tensor(langs[j:j+64],device="cuda")
        mem=keys[lang[:,None],rows]
        ids,length,stop=net.write(mem,rows!=0,output)
        results.extend([r[:n].tolist() for r,n in zip(ids.cpu().numpy(),length.cpu().numpy())])
        stops.extend(stop.tolist())
    return results,stops


def main():
    meta=read(PUB/"corpus.json");schema=read(WORK/"schema.json")
    data=archive(WORK/"test.npz",meta["archives"]["test"]);train=archive(WORK/"train.npz",meta["archives"]["train"])
    bank=archive(WORK/"bank.npz",meta["bank"])
    train_text=read(WORK/"train-refs.json");held_text=read(WORK/"test-refs.json")
    if len(train_text)!=len(train["slots"]) or len(held_text)!=len(data["slots"]):raise ValueError("Reference mismatch")
    concepts=[tuple(v) if v else None for v in schema["concepts"]]
    fitting=set(train_text)
    _,first=np.unique(data["slots"],axis=0,return_index=True)
    first=np.sort(first)
    rows=data["slots"][first]
    gold=[{concepts[i] for i in row if i} for row in rows]
    patterns=realization_patterns()
    output_chars=read(MODEL/"readback.json")
    family={v:concepts.index(("familyFriendly",v)) for v in ("yes","no")}
    original=np.flatnonzero(np.isin(rows,list(family.values())).sum(1)==1)
    flipped=rows[original].copy()
    yes=flipped==family["yes"];no=flipped==family["no"]
    flipped[yes]=family["no"];flipped[no]=family["yes"]
    altered=[{concepts[i] for i in row if i} for row in flipped]
    saved={};scores={};cost={}
    def score(text,source,stop):
        prediction=[claims(s,patterns) for s in text]
        exact=np.array([p==g and ok for p,g,ok in zip(prediction,source,stop)])
        return dict(n=len(text),exact=int(exact.sum()),stopped=int(sum(stop)),novel=sum(t not in fitting for t in text),
                    faithful_novel=sum(ok and t not in fitting for ok,t in zip(exact.tolist(),text)),
                    missing=sum(len(g-p) for p,g in zip(prediction,source)),added=sum(len(p-g) for p,g in zip(prediction,source)),
                    facts=sum(map(len,source))),exact
    with gpu_owner():
        output=palette("cuda")
        for mode in ("ink","byte"):
            net=load(mode,"cuda");clock=monotonic();energy=snapshot()
            for lang in (0,1):
                given=np.concatenate((rows,flipped))
                ids,stops=write(net,mode,bank,given,np.full(len(given),lang),output)
                lines=["".join(output_chars[c] for c in r) for r in ids]
                n=len(rows); key=mode+("-zh" if lang else "-en")
                stats,exact=score(lines[:n],gold,stops[:n])
                flipstat,flipexact=score(lines[n:],altered,stops[n:])
                stats["counterfactual"]=flipstat
                stats["both_counterfactual"]=int((exact[original]&flipexact).sum())
                stats["polarity_both"]=sum(
                    ("familyFriendly",gval) in claims(lines[idx],patterns) and ("familyFriendly",opposite) not in claims(lines[idx],patterns)
                    and ("familyFriendly",opposite) in claims(lines[n+j],patterns) and ("familyFriendly",gval) not in claims(lines[n+j],patterns)
                    for j,idx in enumerate(original) for gval in ("yes","no") if ("familyFriendly",gval) in gold[idx]
                    for opposite in (("no" if gval=="yes" else "yes"),))
                stats["source_exchange"]=score(lines[:n],gold[1:]+gold[:1],stops[:n])[0]["exact"]
                saved[key]=dict(lines=lines,stops=stops)
                scores[key]=stats
                print(key,stats,flush=True)
            cost[mode]=dict(seconds=monotonic()-clock,energy=consumed(energy,snapshot()))
            dump(WORK/(mode+"-generation.json"),{k:v for k,v in saved.items() if k.startswith(mode)})
    unique,which=np.unique(train["slots"],axis=0,return_index=True)
    features=np.zeros((len(unique),len(concepts)),np.int8)
    features[np.arange(len(unique))[:,None],unique]=1;features[:,0]=0
    nearest=[]
    for row in rows:
        cells=row[row!=0]
        shared=features[:,cells].sum(1)
        difference=features.sum(1)+len(cells)-2*shared
        nearest.append(train_text[which[difference.argmin()]])
    scores["nearest"],_=score(nearest,gold,[True]*len(gold))
    human=[{concepts[i] for i in row if i} for row in data["slots"]]
    scores["human"],_=score(held_text,human,[True]*len(human))
    dump(WORK/"generation.json",dict(first=first.tolist(),original=original.tolist(),saved=saved))
    ref=saved["ink-zh"]["lines"]
    before=saved["ink-zh"]["stops"]
    _,passed=score(ref[:len(rows)],gold,before[:len(rows)])
    wrong=np.flatnonzero(~passed)
    both=[j for j,i in enumerate(original) if passed[i] and claims(ref[len(rows)+j],patterns)==altered[j]]
    examples=dict(first=dict(index=0,facts=sorted(gold[0]),written=ref[0]),
                  first_counterfactual=dict(index=int(original[0]),facts=sorted(gold[original[0]]),written=ref[original[0]],edited=ref[len(rows)]))
    if len(wrong):
        i=int(wrong[0]);examples["first_miss"]=dict(index=i,facts=sorted(gold[i]),written=ref[i])
    if both:
        j=both[0];i=int(original[j]);examples["first_faithful_pair"]=dict(index=i,facts=sorted(gold[i]),written=ref[i],edited=ref[len(rows)+j])
    dump(PUB/"results.json",dict(scores=scores,cost=cost,examples=examples,outputs=sha(WORK/"generation.json"),
                               corpus=sha(PUB/"corpus.json")))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()

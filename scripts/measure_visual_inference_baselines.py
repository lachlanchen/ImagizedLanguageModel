"""Fit small visual baselines to the same OCNLI rows without decoding text at inference."""
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts.train_visual_inference_writer import Corpus, PRIVATE, PUBLIC, REG as FIT, WORK, load, memory_bank, sha, write_json

REG=PUBLIC/"baseline_binding.json"


def features(p,h,use_source):
    def runs(seq):
        return [str(int(x)) for x in seq]+[f"{a}:{b}" for a,b in zip(seq,seq[1:])]
    claim=runs(h)
    code=["h"+k for k in claim]
    if use_source:
        source=runs(p)
        code.extend("p"+k for k in source)
        shared=set(source)&set(claim)
        code.extend("b"+k for k in shared)
        code.extend("missing"+k for k in set(claim)-shared)
        code.append("overlap"+str(int(10*len(shared)/max(1,len(set(claim))))))
    return code


class Neighbors:
    def __init__(self,ds,rows):
        counts=[defaultdict(Counter) for _ in range(4)]
        for i in rows:
            h=ds.h[i]
            avail=[(n,pos) for n,pos in enumerate(ds.choices[i],1) if len(pos)]
            for n,pos in avail:
                w=1/(len(avail)*len(pos))
                for j in pos:
                    l=int(h[j-1]) if j else -1
                    r=int(h[j+n]) if j+n<len(h) else -1
                    target=tuple(h[j:j+n])
                    for counter,key in zip(counts,((l,r),l,r,0),strict=True):
                        counter[key][target]+=w
        self.best=[{k:min(c,key=lambda t:(-c[t],t)) for k,c in d.items()} for d in counts]

    def answer(self,ds,masked):
        stream=masked[len(ds.restore):]
        loc=np.flatnonzero(stream==ds.box)
        assert len(loc)==1
        j=loc[0]
        l=int(stream[j-1]) if j else -1
        r=int(stream[j+1]) if j+1<len(stream) else -1
        for table,key in zip(self.best,((l,r),l,r,0),strict=True):
            if key in table:
                return table[key]


def run():
    if (PUBLIC/"baseline.json").exists():raise FileExistsError("Existing baseline")
    bank=memory_bank()
    train,dev=[Corpus(bank,k) for k in ("train","development")]
    with np.load(PRIVATE) as s:
        rows=s["fit"]
        splits={k:(ds,s[k],s[k+"_triples"]) for k,ds in (("validation",train),("official",dev))}
    start=monotonic()
    ngram=Neighbors(train,rows)
    counts={}
    for n,(name,(ds,ix,_)) in enumerate(splits.items()):
        valid,questions,targets=ds.holes(ix,np.random.default_rng(73121+n))
        predicted=[ngram.answer(ds,q) for q in questions]
        ok=np.array([p==tuple(y) for p,y in zip(predicted,targets,strict=True)])
        absent=np.array([bool(set(y)-set(ds.p[i])-set(q)) for i,q,y in zip(valid,questions,targets,strict=True)])
        counts[name]=dict(restoration=dict(total=len(ok),exact=int(ok.sum()),absent_total=int(absent.sum()),
                                         absent=int(ok[absent].sum())))
    times=dict(context_count=monotonic()-start)
    hasher=FeatureHasher(n_features=65536,input_type="string",alternate_sign=True)
    for source in (False,True):
        tag="both" if source else "claim"
        start=monotonic()
        def design(ds,ix):
            return hasher.transform(features(ds.p[i],ds.h[i],source) for i in ix)
        learner=LogisticRegression(C=.5,solver="lbfgs",max_iter=500)
        learner.fit(design(train,rows),[train.meta[i]["label"] for i in rows])
        for name,(ds,ix,trips) in splits.items():
            out=learner.predict(design(ds,ix))
            y=np.array([ds.meta[i]["label"] for i in ix])
            ok=out==y
            tri=np.searchsorted(ix,trips)
            counts[name][tag]=dict(records=len(ok),exact=int(ok.sum()),triples=int(ok[tri].all(1).sum()),
                                   total_triples=len(tri))
        times[tag]=monotonic()-start
    write_json(PUBLIC/"baseline.json",dict(scores=counts,seconds=times,inputs_sha256=sha(PRIVATE),
            program_sha256=sha(Path(__file__)),fit_registration=sha(FIT)))
    print(counts)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--register",action="store_true")
    expected=dict(script_sha256=sha(Path(__file__)),neural_fit=sha(FIT),cohort=sha(PRIVATE),
                  decision="hashed visual unary and adjacent-pair features; C .5; 65536 dimensions",
                  restore="weighted most frequent hidden span with both neighbors; left, then right, then marginal fallback")
    if p.parse_args().register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,expected)
    else:
        assert expected==load(REG)
        with threadpool_limits(limits=2):
            run()

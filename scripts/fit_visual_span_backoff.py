"""Freeze a two-candidate visual writer on internal validation and evaluate official held-out masks."""
import argparse
from pathlib import Path
from time import monotonic
import numpy as np
from sklearn.linear_model import LogisticRegression

from ilm.visual_lm.span_backoff import SpanBackoff
from scripts.audit_traditional_passage_training import archived
from scripts.measure_visual_inference_baselines import Neighbors
from scripts.evaluate_visual_inference_writer import paired_interval
from scripts.evaluate_same_passage_writer import image_archive
from scripts.train_visual_inference_writer import Corpus, PRIVATE, PUBLIC, WORK, REG as PRIMARY, RESTORE, load, memory_bank, sha, write_json

REG=PUBLIC/"backoff_binding.json"
PATH=WORK/"span_backoff"


def binding():
    policies=[Path(__file__),Path(__file__).parents[1]/"ilm/visual_lm/span_backoff.py",
              PUBLIC/"result.json",PUBLIC/"baseline.json",PUBLIC/"audit.json",PRIMARY,PRIVATE,
              Path(__file__).parents[1]/"references/visual_span_backoff_protocol.md"]
    return {str(p.name):sha(p) for p in policies}


def compile_table(ds,rows,bank):
    learned=Neighbors(ds,rows)
    n=len(bank.tiles)
    entries={}
    for level,lookup in enumerate(learned.best):
        for key,span in lookup.items():
            if level==0:
                l,r=key;address=max(0,l)*n+max(0,r)
            elif level<3:
                address=n*n+(level-1)*n+max(0,key)
            else:
                address=n*n+2*n
            assert address not in entries
            entries[address]=span
    keys=np.sort(np.fromiter(entries,np.int64))
    spans=np.zeros((len(keys),3),np.int32)
    for i,k in enumerate(keys):
        s=entries[k];spans[i,:len(s)]=s
    np.savez_compressed(PATH/"neighbors.npz",keys=keys,spans=spans)
    np.save(PATH/"glyphs.npy",np.frombuffer(b"".join(bank.tiles),np.uint8).reshape(-1,128))


def run():
    assert binding()==load(REG)
    PATH.mkdir()
    started=monotonic()
    report=load(PUBLIC/"result.json")
    simple=load(PUBLIC/"baseline.json")["scores"]
    def read(path):
        return archived(WORK/path,report["archives"][path])
    bank=memory_bank()
    train=Corpus(bank,"train")
    with np.load(PRIVATE) as split:compile_table(train,split["fit"],bank)
    model=SpanBackoff(PATH)
    dev=Corpus(bank,"development")
    result={}
    for split,ds in (("validation",train),("official",dev)):
        ids=np.load(WORK/(split+"_cloze_rows.npy"))
        neural,stops=read("paired/"+split+"_restoration_intact.npz")
        masked,_=read(split+"_cloze_requests.npz")
        refs,_=read(split+"_cloze_targets.npz")
        xs=[];alts=[];absent=[];clusters=[]
        for i,raw,q,target,stop in zip(ids,neural,masked,refs,stops,strict=True):
            passage=b"".join(bank.tiles[t] for t in ds.p[i])
            claim=q[len(RESTORE)*128:]
            alt,x=model.features(passage,claim,ds.mask_tile(),raw,stop)
            xs.append(x);alts.append(alt)
            chunks=lambda b:{b[j:j+128] for j in range(0,len(b),128)}
            absent.append(bool(chunks(target)-chunks(passage)-chunks(q)))
            clusters.append(ds.meta[i]["premise"])
        xs=np.stack(xs);absent=np.asarray(absent);clusters=np.asarray(clusters)
        n_ok=np.asarray([s and y==v for s,y,v in zip(stops,neural,refs,strict=True)])
        a_ok=np.asarray([a==y for a,y in zip(alts,refs,strict=True)])
        assert a_ok.sum()==simple[split]["restoration"]["exact"] and a_ok[absent].sum()==simple[split]["restoration"]["absent"]
        if split=="validation":
            useful=n_ok!=a_ok
            fit=LogisticRegression(C=1.,fit_intercept=False,solver="lbfgs",max_iter=1000)
            fit.fit(xs[useful],a_ok[useful].astype(int))
            coeff=fit.coef_[0]
            np.save(PATH/"coefficients.npy",coeff)
        else:
            coeff=np.load(PATH/"coefficients.npy")
        choice=xs@coeff>0
        images=[a if c else n for a,n,c in zip(alts,neural,choice,strict=True)]
        ends=stops|choice
        ok=np.array([s and im==y for s,im,y in zip(ends,images,refs,strict=True)])
        assert np.array_equal(ok,np.where(choice,a_ok,n_ok))
        image_archive(PATH/(split+".npz"),images,ends)
        numbers=lambda sel:dict(total=int(sel.sum()),exact=int(ok[sel].sum()),neural=int(n_ok[sel].sum()),
                neighbors=int(a_ok[sel].sum()),either=int((a_ok|n_ok)[sel].sum()),took_neighbor=int(choice[sel].sum()))
        result[split]=dict(all=numbers(np.ones(len(ok),bool)),absent=numbers(absent),
            effect_neural=paired_interval(ok,n_ok,clusters),effect_neighbors=paired_interval(ok,a_ok,clusters),
            absent_effect_neural=paired_interval(ok[absent],n_ok[absent],clusters[absent]),
            absent_effect_neighbors=paired_interval(ok[absent],a_ok[absent],clusters[absent]),
            by_length={str(k):numbers(np.array([len(t)==128*k for t in refs])) for k in (1,2,3)})
    write_json(PUBLIC/"backoff.json",dict(partitions=result,seconds=monotonic()-started,trainable_gate=len(coeff),
            files={p.name:sha(p) for p in PATH.iterdir()},table_entries=len(model.keys),
            deployment_bytes=sum((PATH/k).stat().st_size for k in ("glyphs.npy","neighbors.npz","coefficients.npy"))))
    print(result["official"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--register",action="store_true")
    if p.parse_args().register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,binding())
    else:run()

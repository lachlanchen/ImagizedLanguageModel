"""Calibrate the bilingual visual lexicon on development and evaluate its free translations."""
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from torch.nn import functional as F
from scripts.fit_optical_lexicon import BASE,ROOT,PUB,WORK,MODEL,read,dump,sha
from scripts.evaluate_full_image_roles import png
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_visual_library import blind_votes
from scripts.write_optical_lexicon import LexicalWriter
from scripts.train_paired_ink import material
from scripts.train_ordered_translation import pack
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/"publication/ilm-nature/evidence/optical_lexicon.zip"
ARMS=("neural","lexical","exchange")


@torch.inference_mode()
def calibrate(w,features,inv,local):
    strengths=(0.,.25,.5,1.)
    totals=np.zeros((2,len(strengths)),dtype=np.float64)
    positions=[0,0]
    dev=read(WORK/"dev.json")
    for t in (0,1):
        for start in range(0,len(dev),16):
            rows=dev[start:start+16]
            sources=[r["cells"][1-t] for r in rows]
            targets=[r["cells"][t]+[1] for r in rows]
            y=pack(targets,"cuda")
            src=pack(sources,"cuda")
            logits=w.neural.writer(features,w.inputs(features,src),y,inv[t],t)
            states=w.laws(sources,t)
            for j in range(y.shape[1]):
                ratio=torch.tensor(w.correction(states),device="cuda",dtype=logits.dtype)
                ids=y[:,j]
                q=local[t][ids]
                for k,strength in enumerate(strengths):
                    sc=w.calibrated(logits[:,j],ratio,strength)
                    totals[t,k]+=float(F.cross_entropy(sc,q,ignore_index=-100,reduction="sum"))
                positions[t]+=int(q.ne(-100).sum())
                for i,(z,c) in enumerate(zip(ids.tolist(),q.tolist(),strict=True)):
                    if z>1:
                        for state in states[i]:
                            state.advance(c,z)
    chosen=strengths[int(totals.sum(0).argmin())]
    meta=dict(strength=chosen,strengths=strengths,positions=positions,
              mean_nll=(totals/np.asarray(positions)[:,None]).tolist(),dev=sha(WORK/"dev.json"))
    dump(PUB/"calibration.json",meta)
    return chosen


def generate():
    if (PUB/"generation.json").exists():
        assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
        return
    pairs=read(WORK/"panel.json")
    assert sha(WORK/"panel.json")==read(PUB/"corpus.json")["reserved"]
    palette=np.load(BASE/"palette.npy")
    names=read(BASE/"readback.json")
    writer=LexicalWriter("cuda")
    features,inv,local=material("cuda")
    strength=calibrate(writer,features,inv,local)
    applied=strength or .5
    panel=[]
    for i,r in enumerate(pairs):
        donor=(i+12)%len(pairs)
        assert pairs[donor]["doc"]!=r["doc"]
        for t in (0,1):
            panel.append(dict(doc=r["doc"],group="book",id=i,t=t,source=r["text"][1-t],ref=r["text"][t],
                              foil=pairs[donor]["text"][t],donor=donor,answers={}))
    tmp=ARCHIVE.with_suffix(".partial")
    lexical_scores=[]
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for t in (0,1):
            for start in range(0,len(pairs),12):
                group=pairs[start:start+12]
                sources=[r["cells"][1-t] for r in group]
                donors=[pairs[(start+j+12)%len(pairs)]["cells"][1-t] for j in range(len(group))]
                for j,r in enumerate(group):
                    i=(start+j)*2+t
                    z.writestr(f"{i:04d}/source.png",png(palette[sources[j]]))
                    target=writer.strips[t].source_strips(r["cells"][t])[1:]
                    table,_=writer.models[t]
                    scores=[]
                    for src in (sources[j],donors[j]):
                        pred=table[writer.strips[1-t].source_strips(src)].mean(0)
                        scores.append(float(np.log(pred[target]).mean()))
                    lexical_scores.append(dict(i=i,t=t,matched=scores[0],exchanged=scores[1]))
                for arm in ARMS:
                    outs=writer.write_ids(sources,t,features,0 if arm=="neural" else applied,donors if arm=="exchange" else None)
                    for j,(ids,stop) in enumerate(outs):
                        i=(start+j)*2+t
                        blob=png(palette[ids])
                        z.writestr(f"{i:04d}/{arm}.png",blob)
                        panel[i]["answers"][arm]=dict(text="".join(names[str(k)] for k in ids),stop=stop,cells=len(ids),
                                                       exact=bool(stop and ids==group[j]["cells"][t]),png=sha256(blob).hexdigest())
        z.writestr("records.json",json.dumps(panel,ensure_ascii=False))
    tmp.replace(ARCHIVE)
    dump(PUB/"generation.json",dict(archive=sha(ARCHIVE),applied=applied,selected=strength,lexical_scores=lexical_scores))
    print("generation",strength,sum(r["matched"]>r["exchanged"] for r in lexical_scores),flush=True)


def assess():
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
    answers=[dict(r["answers"]) for r in rows]
    cache=WORK/"comparator.json"
    if cache.exists():
        ctrl=read(cache)
        assert ctrl["archive"]==sha(ARCHIVE)
        for a,c in zip(answers,ctrl["outputs"],strict=True):
            a.update(c)
    else:
        trial=semantic(rows,answers,ARMS)
        ctrl=dict(archive=sha(ARCHIVE),outputs=[{k:a[k] for k in ("qwen","reference","foil")} for a in answers],
                  qwen=trial["qwen"],judge=trial["judge"],reference_assisted=trial["votes"])
        dump(cache,ctrl)
    votes=blind_votes(rows,answers)
    counts=defaultdict(Counter)
    grades=[]
    for i,(r,a) in enumerate(zip(rows,answers,strict=True)):
        g={k:bool(v["stop"] and votes[2*i][k] and votes[2*i+1][k]) for k,v in a.items()}
        grades.append(g)
        for name in (r["doc"],("English","Chinese")[r["t"]],"all"):
            counts[name].update({k:int(v) for k,v in g.items()})
            counts[name]["n"]+=1
    delta=[(v["lexical"]-v["neural"])/v["n"] for k,v in counts.items() if k not in ("all","English","Chinese")]
    ci=np.quantile([np.mean(boot) for boot in product(delta,repeat=len(delta))],[.025,.975]).tolist()
    dump(PUB/"assessment.json",dict(archive=sha(ARCHIVE),counts=counts,grades=grades,controls=ctrl,votes=votes,
         paired_work_interval=ci))
    print(dict(counts["all"]),ci,flush=True)


def main():
    configure_cuda_cli()
    with gpu_owner():
        generate()
        if not (PUB/"assessment.json").exists():
            assess()


if __name__=="__main__":
    main()

"""Generate, archive and assess matched ordered and unordered visual translations."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from ilm.visual_lm.ordered_alignment import movement,strip_log_likelihood
from scripts.prepare_ordered_alignment import PUB, WORK, ROOT, base
from scripts.write_ordered_alignment import OrderedWriter
from scripts.write_optical_lexicon import LexicalWriter
from scripts.evaluate_full_image_roles import png,rows as role_rows
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_visual_library import blind_votes
from scripts.search_optical_passages import PixelPassageSearch
from scripts.train_paired_ink import material
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/"publication/ilm-nature/evidence/ordered_alignment.zip"
ARMS=("neural","unordered","ordered")


def generate():
    digest=base.sha(WORK/"panel.json")
    assert digest==base.read(PUB/"corpus.json")["panel"]
    if (PUB/"generation.json").exists():
        assert base.sha(ARCHIVE)==base.read(PUB/"generation.json")["archive"]
        return
    pairs=base.read(WORK/"panel.json")
    panel=[]
    for i,r in enumerate(pairs):
        for t in (0,1):
            other=pairs[(i+12)%len(pairs)]
            assert other["doc"]!=r["doc"]
            panel.append(dict(doc=r["doc"],group="book",id=i,t=t,source=r["text"][1-t],ref=r["text"][t],foil=other["text"][t]))
    panel.extend(r for r in role_rows() if r["group"]=="object")
    ink=base.Ink(np.load(base.BASE/"palette.npy"))
    source=[ink.encode(r["source"]) for r in panel]
    ref=[ink.encode(r["ref"]) for r in panel]
    assert not ink.bad
    pixels=ink.array()
    names=base.read(base.BASE/"readback.json")
    features,*_=material("cuda")
    ordered=OrderedWriter("cuda")
    unordered=LexicalWriter("cuda")
    # Every held source uses the same physical palette as the earlier print writer.
    assert all(max(s)<len(ordered.neural.palette) for s in source)
    partial=ARCHIVE.with_suffix(".partial")
    with ZipFile(partial,"w",ZIP_DEFLATED) as z:
        for i,r in enumerate(panel):
            z.writestr(f"{i:04d}/source.png",png(pixels[source[i]]))
            r["answers"]={}
        for name in ARMS:
            writer=ordered if name=="ordered" else unordered
            for t in (0,1):
                index=[i for i,r in enumerate(panel) if r["t"]==t]
                for j in range(0,len(index),12):
                    batch=index[j:j+12]
                    outs=writer.write_ids([source[i] for i in batch],t,features,0. if name=="neural" else .5)
                    for i,(cells,stop) in zip(batch,outs,strict=True):
                        blob=png(pixels[cells])
                        z.writestr(f"{i:04d}/{name}.png",blob)
                        panel[i]["answers"][name]=dict(text="".join(names[str(k)] for k in cells),stop=stop,
                                     exact=bool(stop and cells==ref[i]),cells=len(cells),png=sha256(blob).hexdigest())
            print("emitted",name,flush=True)
        z.writestr("records.json",json.dumps(panel,ensure_ascii=False))
    partial.replace(ARCHIVE)
    base.dump(PUB/"generation.json",dict(archive=base.sha(ARCHIVE),source=digest,cases=len(panel),arms=ARMS))


def retrieval():
    writer=OrderedWriter()
    cases={}
    for label,folder in (("previous",base.WORK),("fresh",WORK)):
        passages=base.read(folder/"panel.json")
        results=[]
        for t,lang in enumerate(("en","zh")):
            search=PixelPassageSearch(lang)
            clean=[[np.asarray([k for k in writer.strips[d].source_strips(r["cells"][d]) if k in search.content[d]],dtype=int)
                    for r in passages] for d in (0,1)]
            laws=writer.models[t][0]
            for i,tgt in enumerate(clean[t]):
                if not len(tgt):
                    results.append(dict(i=i,t=t,ordered=False,unordered=False,no_query=True))
                    continue
                ordered,unordered=[],[]
                for src in clean[1-t]:
                    n=len(src)
                    e=(laws[np.ix_(src,tgt)].astype(np.float64)*n+laws[0,tgt])/(n+1) if n else laws[[0]][:,tgt]
                    trans=movement(len(e),writer.weights[t])
                    ordered.append(strip_log_likelihood(e,trans,range(len(tgt)))/len(tgt))
                    unordered.append(np.log(e.mean(0)).mean())
                record=dict(i=i,t=t)
                for name,scores in (("ordered",ordered),("unordered",unordered)):
                    record[name]=bool(scores[i]>max(scores[:i]+scores[i+1:]))
                results.append(record)
        cases[label]=dict(counts={m:sum(r[m] for r in results) for m in ("ordered","unordered")},total=len(results),rows=results)
    assert cases["previous"]["counts"]["unordered"]==93
    base.dump(PUB/"retrieval.json",cases)
    print("search",[(k,v["counts"]) for k,v in cases.items()],flush=True)


def assessment():
    if (PUB/"assessment.json").exists():
        assert base.read(PUB/"assessment.json")["archive"]==base.sha(ARCHIVE)
        return
    with ZipFile(ARCHIVE) as z:
        panel=json.loads(z.read("records.json"))
    book=[r for r in panel if r["group"]=="book"]
    responses=[r["answers"].copy() for r in book]
    cache=WORK/"comparison.json"
    if cache.exists():
        ctrl=base.read(cache)
        assert ctrl["archive"]==base.sha(ARCHIVE)
        for r,c in zip(responses,ctrl["controls"],strict=True):
            r.update(c)
    else:
        reference_run=semantic(book,responses,ARMS)
        ctrl=dict(archive=base.sha(ARCHIVE),controls=[{k:v[k] for k in ("qwen","foil","reference")} for v in responses],
                  qwen=reference_run["qwen"],judge=reference_run["judge"])
        base.dump(cache,ctrl)
    votes=blind_votes(book,responses)
    counts=defaultdict(Counter)
    grades=[]
    for i,(r,data) in enumerate(zip(book,responses,strict=True)):
        grades.append({m:bool(v["stop"] and votes[2*i][m] and votes[2*i+1][m]) for m,v in data.items()})
        for group in ("all",r["doc"],("en","zh")[r["t"]]):
            counts[group].update({m:int(v) for m,v in grades[-1].items()})
            counts[group]["n"]+=1
    face=base.Ink(np.load(base.BASE/"palette.npy"))
    coverage={}
    for t,lang in enumerate(("en","zh")):
        system=PixelPassageSearch(lang)
        def content(text):
            return set(system.strip[t].source_strips(face.encode(text))) & system.content[t]
        totals=Counter()
        for r in book:
            if r["t"]!=t:
                continue
            ref=content(r["ref"])
            totals["reference"]+=len(ref)
            for arm in ARMS:
                totals[arm]+=len(content(r["answers"][arm]["text"]) & ref)
        coverage[lang]=totals
    controlled=[r for r in panel if r["group"]=="object"]
    role={}
    for m in ARMS:
        exact=[r["answers"][m]["exact"] for r in controlled]
        role[m]=dict(exact=sum(exact),both_polarities=sum(all(exact[i:i+2]) for i in range(0,len(exact),2)),
                     both_roles_polarities=sum(all(exact[i:i+4]) for i in range(0,len(exact),4)))
    works=list(dict.fromkeys(r["doc"] for r in book))
    differences=[(counts[k]["ordered"]-counts[k]["unordered"])/counts[k]["n"] for k in works]
    interval=np.quantile([np.mean(x) for x in product(differences,repeat=len(works))],[.025,.975]).tolist()
    stats=dict(archive=base.sha(ARCHIVE),counts=counts,grades=grades,votes=votes,controls=ctrl,coverage=coverage,role=role,
           paired_work_interval=interval,stops={m:sum(r["answers"][m]["stop"] for r in book) for m in ARMS})
    base.dump(PUB/"assessment.json",stats)
    print(dict(counts["all"]),role,coverage,flush=True)


def main():
    configure_cuda_cli()
    with gpu_owner():
        generate()
        if not (PUB/"retrieval.json").exists():
            retrieval()
        assessment()


if __name__=="__main__":
    main()

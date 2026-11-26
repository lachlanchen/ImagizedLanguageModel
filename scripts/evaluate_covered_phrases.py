"""Compare free optical phrase transduction with the frozen neural and alignment writers."""
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from scripts.prepare_covered_phrases import PUB,ROOT,WORK,base,prior
from scripts.write_covered_phrases import CoveredWriter
from scripts.evaluate_ordered_alignment import ARCHIVE as PREVIOUS,OrderedWriter,LexicalWriter,material,png
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_visual_library import blind_votes
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/"publication/ilm-nature/evidence/covered_phrases.zip"
CONTROLS=("neural","unordered","ordered")
ARMS=(*CONTROLS,"phrase")


def generate():
    if (PUB/"generation.json").exists():
        assert base.read(PUB/"generation.json")["archive"]==base.sha(ARCHIVE)
        return
    assert base.sha(WORK/"panel.json")==base.read(PUB/"corpus.json")["panel"]
    pairs=base.read(WORK/"panel.json")
    assert base.sha(PREVIOUS)==base.read(prior.PUB/"generation.json")["archive"]
    panel=[]
    for i,pair in enumerate(pairs):
        other=pairs[(i+12)%len(pairs)]
        assert other["doc"]!=pair["doc"]
        for t in (0,1):
            panel.append(dict(doc=pair["doc"],group="book",id=i,t=t,source=pair["text"][1-t],
                        ref=pair["text"][t],foil=other["text"][t],answers={}))
    palette=np.load(base.BASE/"palette.npy")
    face=base.Ink(palette)
    names=base.read(base.BASE/"readback.json")
    reverse={p.tobytes():names[str(i)] for i,p in enumerate(palette) if i>=2 and str(i) in names}
    original=LexicalWriter("cuda")
    ordered=OrderedWriter("cuda")
    features,*_=material("cuda")
    paths=[]
    temp=ARCHIVE.with_suffix(".partial")
    with ZipFile(PREVIOUS) as prev, ZipFile(temp,"w",ZIP_DEFLATED) as z:
        controlled=json.loads(prev.read("records.json"))[96:]
        assert len(controlled)==472
        for r in controlled:
            panel.append(dict(r,answers=dict(r["answers"])))
        sources=[face.encode(r["source"]) for r in panel]
        refs=[face.encode(r["ref"]) for r in panel]
        assert not face.bad
        tiles=face.array()
        for i,r in enumerate(panel):
            raw=png(tiles[sources[i]])
            if r["group"]=="object":
                assert raw==prev.read(f"{i:04d}/source.png")
                for m in CONTROLS:
                    saved=prev.read(f"{i:04d}/{m}.png")
                    assert sha256(saved).hexdigest()==r["answers"][m]["png"]
                    z.writestr(f"{i:04d}/{m}.png",saved)
            z.writestr(f"{i:04d}/source.png",raw)
        for t in (0,1):
            writer=CoveredWriter(t)
            order=[i for i,r in enumerate(panel) if r["t"]==t]
            books=[i for i in order if panel[i]["group"]=="book"]
            for arm in CONTROLS:
                model=ordered if arm=="ordered" else original
                for j in range(0,len(books),12):
                    ids=books[j:j+12]
                    for i,(out,stop) in zip(ids,model.write_ids([sources[k] for k in ids],t,features,0. if arm=="neural" else .5),strict=True):
                        blob=png(palette[out])
                        z.writestr(f"{i:04d}/{arm}.png",blob)
                        panel[i]["answers"][arm]=dict(cells=len(out),text="".join(names[str(k)] for k in out),stop=stop,
                            exact=bool(stop and out==refs[i]),png=sha256(blob).hexdigest())
            for i in order:
                inp=tiles[sources[i]]
                strips=writer.source(inp)
                out,path,score=writer.translate(inp)
                covered=[a for c in path for a in range(c.start,c.end)]
                assert sorted(covered)==list(range(len(strips)))
                blob=png(out)
                z.writestr(f"{i:04d}/phrase.png",blob)
                panel[i]["answers"]["phrase"]=dict(text="".join(reverse[p.tobytes()] for p in out),stop=True,
                       exact=np.array_equal(out,tiles[refs[i]]),cells=len(out),png=sha256(blob).hexdigest())
                paths.append(dict(id=i,source_strips=len(strips),score=score,phrases=[
                     dict(span=[c.start,c.end],target=list(c.target),observations=c.evidence) for c in path]))
            print("wrote",t,len(order),flush=True)
        z.writestr("records.json",json.dumps(panel,ensure_ascii=False))
        z.writestr("paths.json",json.dumps(paths))
    temp.replace(ARCHIVE)
    base.dump(PUB/"generation.json",dict(archive=base.sha(ARCHIVE),requests=len(panel),cases_book=96,
           controlled_from=base.sha(PREVIOUS),weights=base.sha(writer_model_dir()/"fit.json"),
           calibration=base.sha(writer_model_dir()/"calibration.json"),probes=base.sha(PUB/"name-probes.json")))


def writer_model_dir():
    return ROOT/"models/covered_phrases"


def assess():
    if (PUB/"assessment.json").exists():
        assert base.read(PUB/"assessment.json")["archive"]==base.sha(ARCHIVE)
        return
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
    book=rows[:96]
    responses=[r["answers"].copy() for r in book]
    cache=WORK/"comparison.json"
    if cache.exists():
        meta=base.read(cache)
        assert meta["archive"]==base.sha(ARCHIVE)
        for resp,c in zip(responses,meta["outputs"],strict=True):
            resp.update(c)
    else:
        run=semantic(book,responses,ARMS)
        meta=dict(archive=base.sha(ARCHIVE),outputs=[{m:r[m] for m in ("qwen","reference","foil")} for r in responses],
                  qwen=run["qwen"],judge=run["judge"])
        base.dump(cache,meta)
    votes=blind_votes(book,responses)
    counts=defaultdict(Counter)
    grades=[]
    for i,(r,resp) in enumerate(zip(book,responses,strict=True)):
        g={m:bool(v["stop"] and votes[2*i][m] and votes[2*i+1][m]) for m,v in resp.items()}
        grades.append(g)
        for group in (r["doc"],("en","zh")[r["t"]],"all"):
            counts[group].update({m:int(v) for m,v in g.items()})
            counts[group]["n"]+=1
    roles={}
    for m in ARMS:
        marks=[r["answers"][m]["exact"] for r in rows[96:]]
        roles[m]=dict(exact=sum(marks),polarity=sum(all(marks[i:i+2]) for i in range(0,472,2)),
                      crossed=sum(all(marks[i:i+4]) for i in range(0,472,4)))
    delta=[(counts[w]["phrase"]-counts[w]["ordered"])/counts[w]["n"] for w in dict.fromkeys(r["doc"] for r in book)]
    ci=np.quantile([np.mean(x) for x in product(delta,repeat=4)],[.025,.975]).tolist()
    base.dump(PUB/"assessment.json",dict(archive=base.sha(ARCHIVE),counts=counts,roles=roles,votes=votes,grades=grades,
          comparator=meta,work_interval=ci,completed={m:sum(r["answers"][m]["stop"] for r in book) for m in ARMS}))
    print(dict(counts["all"]),roles,flush=True)


def main():
    configure_cuda_cli()
    with gpu_owner():
        generate()
        assess()


if __name__=="__main__":
    main()

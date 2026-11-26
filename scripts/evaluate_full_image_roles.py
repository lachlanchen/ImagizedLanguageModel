"""Score free image translation on reserved works and on crossed polarity and roles."""
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
import re
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from PIL import Image
from scripts.prepare_full_image_roles import BASE, PUB, ROOT, WORK, Ink, dump, read, sha
from scripts.write_full_image_roles import PicturedTranslator
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_visual_library import blind_votes
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/"publication/ilm-nature/evidence/full_image_roles.zip"
ARMS=("full","marked","displaced")


def rows():
    pairs=read(WORK/"panel.json")
    out=[dict(doc=r["doc"],group="book",id=i,t=t,source=r["text"][1-t],ref=r["text"][t],
              foil=pairs[(i+12)%len(pairs)]["text"][t]) for i,r in enumerate(pairs) for t in (0,1)]
    source=read(ROOT/"artifacts/translation_paraphrases_v1/voice-outputs.json")
    items=[r for r in source if r["voice"]=="active"]
    for i in range(0,len(items),2):
        positive,negative=items[i:i+2]
        a,verb,b=re.fullmatch(r"(.+?) did not (\w+) (.+)\.",negative["source"]).groups()
        assert not positive["negative"] and negative["negative"]
        if a.startswith("The "):
            a="the "+a[4:]
        for sign,r in enumerate((positive,negative)):
            vphrase=("did not " if sign else "did ")+verb
            for group,src in (("plain",r["source"]),("subject",f"It was {a} who {vphrase} {b}."),
                              ("object",f"It was {b} whom {a} {vphrase}.")):
                out.append(dict(doc="roles",group=group,id=i//2,t=1,negative=bool(sign),source=src,ref=r["reference"]))
    with ZipFile(ROOT/"publication/ilm-nature/evidence/visual_referents.zip") as z:
        ext=[r for r in json.loads(z.read("writing.json")) if r["form"]=="external"]
    for i,r in enumerate(ext):
        out.append(dict(doc="GENTLE",group="natural",id=i,t=1,source=r["source"],ref=r["reference"],verb=r["verb"]))
    return out


def png(packed):
    n=len(packed)
    width=min(32,max(1,n))
    height=max(1,(n+width-1)//width)
    tile=np.zeros((height*width,32,32),np.uint8)
    tile[:n]=np.unpackbits(packed,axis=1).reshape(n,32,32)
    page=tile.reshape(height,width,32,32).transpose(0,2,1,3).reshape(height*32,width*32)
    buf=BytesIO()
    Image.fromarray(255*(1-page)).save(buf,format="PNG")
    return buf.getvalue()


def generate():
    assert sha(WORK/"panel.json")==read(PUB/"corpus.json")["hashes"]["panel.json"]
    if (PUB/"generation.json").exists():
        assert read(PUB/"generation.json")["archive"]==sha(ARCHIVE)
        return
    panel=rows()
    palette=np.load(BASE/"palette.npy")
    face=Ink(palette)
    source_ids=[face.encode(r["source"]) for r in panel]
    assert not face.bad
    array=face.array()
    names=read(BASE/"readback.json")
    encoder={r.tobytes():names[str(i)] for i,r in enumerate(palette) if str(i) in names and i>=2}
    temp=ARCHIVE.with_suffix(".partial")
    with ZipFile(temp,"w",ZIP_DEFLATED) as z:
        for i,ids in enumerate(source_ids):
            z.writestr(f"{i:04d}/source.png",png(array[ids]))
            panel[i]["answers"]={}
        for mode in ARMS:
            writer=PicturedTranslator(mode,"cuda")
            for t in (0,1):
                order=[i for i,r in enumerate(panel) if r["t"]==t]
                for j in range(0,len(order),32):
                    ids=order[j:j+32]
                    preds=writer.translate([array[source_ids[i]] for i in ids],t)
                    for i,(img,stop) in zip(ids,preds,strict=True):
                        blob=png(img)
                        z.writestr(f"{i:04d}/{mode}.png",blob)
                        ref=panel[i]["ref"]
                        target=face.array()[face.encode(ref)] if all(c in face.chars for c in ref) else None
                        if target is None:
                            ref_ids=face.encode(ref)
                            target=face.array()[ref_ids]
                        text="".join(encoder[p.tobytes()] for p in img)
                        panel[i]["answers"][mode]=dict(text=text,stop=stop,exact=bool(stop and np.array_equal(img,target)),
                                                       cells=len(img),png=sha256(blob).hexdigest())
        z.writestr("records.json",json.dumps(panel,ensure_ascii=False))
    temp.replace(ARCHIVE)
    dump(PUB/"generation.json",dict(archive=sha(ARCHIVE),cases=len(panel),arms=list(ARMS),
                                     source=sha(WORK/"panel.json"),model_input="unaltered source image"))


def assess():
    checksum=sha(ARCHIVE)
    with ZipFile(ARCHIVE) as z:
        panel=json.loads(z.read("records.json"))
    book=[r for r in panel if r["group"]=="book"]
    answers=[dict(r["answers"]) for r in book]
    cache=WORK/"comparator.json"
    if cache.exists():
        control=read(cache)
        assert control["archive"]==checksum
        for a,c in zip(answers,control["outputs"],strict=True):
            a.update(c)
    else:
        comparison=semantic(book,answers,ARMS)
        control=dict(archive=checksum,outputs=[{k:a[k] for k in ("qwen","reference","foil")} for a in answers],
                     qwen=comparison["qwen"],judge=comparison["judge"],reference_assisted=comparison["votes"])
        dump(cache,control)
    votes=blind_votes(book,answers)
    docs=defaultdict(Counter)
    grades=[]
    for i,(r,a) in enumerate(zip(book,answers,strict=True)):
        result={k:bool(val["stop"] and votes[2*i][k] and votes[2*i+1][k]) for k,val in a.items()}
        grades.append(result)
        for group in (r["doc"],("English","Chinese")[r["t"]],"all"):
            docs[group].update({m:int(v) for m,v in result.items()})
            docs[group]["n"]+=1
    delta=[(v["marked"]-v["full"])/v["n"] for g,v in docs.items() if g not in ("all","English","Chinese")]
    boots=[np.mean(x) for x in product(delta,repeat=len(delta))]
    controlled=defaultdict(dict)
    for form in ("plain","subject","object"):
        examples=[r for r in panel if r["group"]==form]
        for mode in ARMS:
            hits=[r["answers"][mode]["exact"] for r in examples]
            controlled[form][mode]=dict(total=len(hits),exact=sum(hits),
                positive=sum(hits[::2]),negative=sum(hits[1::2]),
                both_polarities=sum(a and b for a,b in zip(hits[::2],hits[1::2],strict=True)),
                both_roles_and_polarities=sum(all(hits[i:i+4]) for i in range(0,len(hits),4)))
    dump(PUB/"assessment.json",dict(archive=checksum,totals=docs,grades=grades,votes=votes,controls=control,
         controlled=controlled,paired_work_interval=np.quantile(boots,[.025,.975]).tolist()))
    print(dict(docs["all"]),dict(controlled),flush=True)


def main():
    configure_cuda_cli()
    with gpu_owner():
        generate()
        if not (PUB/"assessment.json").exists():
            assess()


if __name__=="__main__":
    main()

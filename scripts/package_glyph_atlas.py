"""Package a portable selection of attested records and export actual reader examples."""
from collections import Counter
import json
import numpy as np
import torch
from PIL import Image

from scripts.read_glyph_atlas import answer,cell_grid,get_book,packed_text
from scripts.prepare_glyph_atlas import WORK,MODEL,PUB,read,dump,sha
from scripts.train_glyph_atlas import datafile,load_model
from scripts.glyph_atlas_wording import load

SAMPLE="言馬马中水語文一學学字車"
QUERIES=(
    ("yan","What is the evolution of 言?","言",(4,0,1,2,3)),
    ("ma","請列出马的金文、甲骨文。","马",(1,0)),
    ("contrast","I need the bronze form of 马, not 言.","马",(1,)),
    ("implicit","Where did the character 言 come from?","言",(4,0,1,2,3)),
    ("unseen","Show the meaning of Ω.","Ω",(4,)),
)


def subbook(book,rows):
    flat=[];spans=[]
    for row in rows:
        coords=[]
        for offset,n in book["spans"][row]:
            coords.append((len(flat),n))
            flat.extend(book["stream"][offset:offset+n])
        spans.append(coords)
    ids=np.unique(np.r_[flat,book["heads"][rows],0])
    return dict(tiles=book["tiles"][ids],heads=np.searchsorted(ids,book["heads"][rows]),
                spans=np.asarray(spans,np.int32),stream=np.searchsorted(ids,flat).astype(np.int32))


def main():
    torch.set_num_threads(2)
    book,meta=get_book(WORK)
    rows=[i for i,e in enumerate(meta) if e["char"] in SAMPLE]
    dest=MODEL/"example_atlas";dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/"atlas.npz",**subbook(book,rows))
    dump(dest/"sources.json",[meta[i] for i in rows])
    dump(dest/"atlas.json",dict(files={p:sha(dest/p) for p in ("atlas.npz","sources.json")},edition="example"))
    mini,mini_meta=get_book(dest)
    report={}
    examples=MODEL/"examples";examples.mkdir(exist_ok=True)
    mode=read(MODEL/"release.json")["reader"]
    net=load_model(mode,"cpu")
    labels=read(PUB/"corpus.json")["fields"]
    for name,text,head,expected in QUERIES:
        q=packed_text(text)
        image,raw,rec=answer(q,book,meta,net,"cpu")
        expected_names=[labels[f] for f in expected]
        correct=(rec["headword"]==head and rec["fields"]==expected_names) if name!="unseen" else rec["headword"] is None
        report[name]=dict(question=text,correct=correct,**rec)
        if name in ("ma","yan"):
            image.save(examples/(name+"-answer.png"))
            cell_grid(q,len(q)).save(examples/(name+"-question.png"))
            _,small,mini_rec=answer(q,mini,mini_meta,net,"cpu")
            if small!=raw or mini_rec["entry"]!=rec["entry"]:
                raise ValueError("Portable result differs")
            for f in (name+"-answer.png",name+"-question.png"):
                (PUB/f).write_bytes((examples/f).read_bytes())
    with np.load(PUB/"wording-outputs.npz") as result:
        flags=result[mode+"/fresh/retrieved"]
    fresh=load("fresh")
    miss=np.flatnonzero(~flags[:,3]&flags[:,5])[0]
    ob=load("obs")
    ids=fresh["query"][miss]
    ids=ids[:(ids!=0).sum()]
    q=ob["tiles"][0,ids]
    page,_,rec=answer(q,book,meta,net,"cpu")
    target=meta[fresh["entry"][miss]]
    from ilm.visual_lm.glyph_atlas import requested,mentions
    ref=requested(mentions(fresh["tag"][miss]),book["spans"][fresh["entry"][miss]])
    report["first_fresh_miss"]=dict(case=int(miss),expected_head=target["char"],expected_fields=[labels[i] for i in ref],**rec)
    cell_grid(q,min(32,len(q))).save(PUB/"miss-question.png")
    page.save(PUB/"miss-answer.png")
    dump(PUB/"examples.json",report)
    print({k:(v.get("headword"),v.get("correct")) for k,v in report.items()})


if __name__=="__main__":
    main()

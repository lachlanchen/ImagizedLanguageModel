"""Answer a pictured historical-form or dictionary-field request from a local atlas."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.glyph_atlas import address,mentions,requested,payload
from scripts.atlas_questions import FIELDS,LIMIT
from scripts.prepare_glyph_atlas import WORK,MODEL,PUB,read,sha,dump,Rasterizer,FACES
from scripts.train_glyph_atlas import load_model
from scripts.infer_continuous_image_language import image_cells

LABELS=("Oracle bone  甲骨文","Bronze  金文","Seal  篆書","Liushutong  六書通",
        "Original meaning  本義","Character structure  字形分解")


def get_book(folder):
    manifest=read(PUB/"corpus.json") if folder.resolve()==WORK.resolve() else read(folder/"atlas.json")
    for name in ("atlas.npz","sources.json"):
        if sha(folder/name)!=manifest["files"][name]:
            raise ValueError("Changed reference collection: "+name)
    with np.load(folder/"atlas.npz") as archive:
        book=dict(archive)
    sources=read(folder/"sources.json")
    if len(sources)!=len(book["heads"]) or book["spans"].shape!=(len(sources),6,2):
        raise ValueError("Reference record count mismatch")
    return book,sources


def packed_text(s):
    raster=Rasterizer()
    return np.asarray([np.frombuffer(raster.tile(c),np.uint8) for c in s])


def input_image(path):
    cells=image_cells(path)
    packed=np.packbits(cells.reshape(len(cells),1024),axis=1)
    valid=np.flatnonzero(packed.any(1))
    return packed[:valid[-1]+1] if len(valid) else packed[:0]


def cell_grid(bits,cols):
    n=len(bits)
    padded=np.zeros((math.ceil(n/cols)*cols,1024),np.uint8)
    padded[:n]=np.unpackbits(bits,axis=1)
    return Image.fromarray(255-255*padded.reshape(-1,cols,32,32).transpose(0,2,1,3).reshape(-1,32*cols))


def page(book,entry,fields):
    streams=payload(book,entry,fields)
    cols=max(13,min(24,max((len(line) for line in streams),default=0)))
    font=ImageFont.truetype(str(FACES[0]),18,index=2)
    small=ImageFont.truetype(str(FACES[0]),14,index=2)
    height=106+sum(52+32*max(1,math.ceil(len(line)/cols)) for line in streams)
    canvas=Image.new("RGB",(cols*32+64,height),"white")
    pen=ImageDraw.Draw(canvas)
    pen.text((32,14),"HANZIYUAN   /   字源",font=small,fill="#23575a")
    if entry>=0:
        canvas.paste(cell_grid(book["tiles"][book["heads"][entry:entry+1]],1),(32,44))
    if not fields:
        pen.text((83,45),"No matching entry or requested field",font=font,fill="#444444")
    y=95
    for f,line in zip(fields,streams):
        pen.line((32,y-5,cols*32+32,y-5),fill="#c0cfce",width=1)
        pen.text((32,y),LABELS[f],font=font,fill="#23575a")
        if len(line):
            canvas.paste(cell_grid(book["tiles"][line],min(cols,len(line))),(32,y+30))
        else:
            pen.text((32,y+32),"Unavailable in this edition",font=small,fill="#666666")
        y+=52+32*max(1,math.ceil(len(line)/cols))
    return canvas


@torch.inference_mode()
def answer(query,book,sources,net,device,approximate=False):
    if not 0<len(query)<=LIMIT or not query.any():
        raise ValueError(f"Supply 1 to {LIMIT} aligned, nonempty prompt cells")
    tensor=lambda cells:torch.as_tensor(observations({"tiles":cells},"ink"),device=device)
    q=tensor(query)[None]
    selected,pos,tag,_,_=address(net,q,torch.ones(q.shape[:2],dtype=torch.bool,device=device),
                               tensor(book["tiles"][book["heads"]]),approximate=approximate)
    entry=int(selected[0])
    # A pooled cell can alias another printed shape; use original bits for exact admission.
    if not approximate and entry>=0 and not np.array_equal(query[pos[0]],book["tiles"][book["heads"][entry]]):
        entry=-1
    actions=mentions(tag[0]); fields=requested(actions,book["spans"][entry]) if entry>=0 else ()
    streams=payload(book,entry,fields)
    available=entry>=0 and bool(fields)
    record=sources[entry] if entry>=0 else None
    receipt=dict(entry=record["entry"] if record else None,headword=record["char"] if record else None,
                 collection_entries=len(sources),head_cell=int(pos[0]),fields=[FIELDS[f] for f in fields],
                 cells=[len(v) for v in streams],attestations=[record["sources"][f] if f<4 else None for f in fields],
                 status="answered" if available else "unresolved",approximate=approximate)
    return page(book,entry,fields),b"".join(book["tiles"][s].tobytes() for s in streams),receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--question-image",type=Path)
    group.add_argument("--question-text")
    ap.add_argument("--atlas",type=Path,default=WORK if (WORK/"atlas.npz").exists() else MODEL/"example_atlas")
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--approximate",action="store_true",help="Permit nearest headword for an unfamiliar print face")
    ap.add_argument("--transcript",action="store_true",help="Export selected source text and references to historical images")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.device=="cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(2)
    book,sources=get_book(args.atlas)
    q=input_image(args.question_image) if args.question_image else packed_text(args.question_text or "")
    image,raw,result=answer(q,book,sources,load_model(read(MODEL/"release.json")["reader"],args.device),args.device,args.approximate)
    args.output.mkdir(parents=True,exist_ok=False)
    image.save(args.output/"answer.png")
    (args.output/"answer.bin").write_bytes(raw)
    dump(args.output/"answer.json",result)
    if args.transcript and result["entry"] is not None:
        entry=next(e for e in sources if e["entry"]==result["entry"] and e["char"]==result["headword"])
        sections=[entry["char"]]
        for name,count in zip(result["fields"],result["cells"]):
            f=FIELDS.index(name)
            value=entry["sources"][f] if count else None
            if value and f<4:
                value=f"[Hanziyuan {value}, see answer.png]"
            sections.append(LABELS[f]+" — "+(value or "Unavailable in this edition"))
        (args.output/"transcript.txt").write_text("\n".join(sections)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()

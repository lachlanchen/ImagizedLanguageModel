"""Answer a pictured origin request from a labelled page, using only its pixels and row geometry."""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scripts.train_origin_value_reader import MODEL, OriginValueReader, predict, read, sha

TILE=32


def png(bits):
    buf=BytesIO()
    Image.fromarray(255-np.asarray(bits,dtype=np.uint8)*255).save(buf,format="PNG")
    return buf.getvalue()


def strip(tiles):
    n=len(tiles)
    width=max(1,min(32,n))
    height=max(1,(n+width-1)//width)
    canvas=np.zeros((height*width,TILE,TILE),np.uint8)
    if n: canvas[:n]=np.unpackbits(np.frombuffer(b"".join(tiles),np.uint8)).reshape(n,TILE,TILE)
    return png(canvas.reshape(height,width,TILE,TILE).transpose(0,2,1,3).reshape(height*TILE,width*TILE))


def read_grid(blob):
    with Image.open(BytesIO(blob)) as im: pixels=np.asarray(im.convert("L"))<128
    h,w=pixels.shape
    if h%TILE or w%TILE: raise ValueError("Image must be aligned on 32-square cells")
    return np.packbits(pixels.reshape(h//TILE,TILE,w//TILE,TILE).transpose(0,2,1,3).reshape(h//TILE,w//TILE,1024),axis=2)


def decode(question,page,g):
    q=read_grid(question).reshape(-1,128)
    board=read_grid(page)
    n=g["question_cells"]; start=g["value_column"]
    if not 0<n<=len(q) or board.shape[0]!=len(g["rows"]) or not 0<start<board.shape[1]:
        raise ValueError("Geometry does not fit the supplied images")
    rows=[]
    for line,(label,value) in zip(board,g["rows"],strict=True):
        if not 0<label<=start or not 0<value<=len(line)-start: raise ValueError("Invalid field extent")
        rows.append(dict(label=list(map(bytes,line[:label])),value=list(map(bytes,line[start:start+value]))))
    return dict(question=list(map(bytes,q[:n])),page=rows)


def model(name,device):
    f=MODEL/name/"reader.safetensors"
    if sha(f)!=read(MODEL/name/"fit.json")["weights"]: raise ValueError("Origin reader weights changed")
    net=OriginValueReader(name=="matched").to(device)
    net.load_state_dict(load_file(str(f),device=device))
    return net.eval()


def answer(net,inputs,device):
    # Index identical input pixels within this batch. Index 0 is padding. Values
    # stay outside the reader until a chosen source row is actually emitted.
    table=[bytes(128)]
    index={}
    def encode(seq):
        handles=[]
        for v in seq:
            if v not in index:
                index[v]=len(table)
                table.append(v)
            handles.append(index[v])
        return handles
    items=[dict(question=encode(x["question"]),page=[dict(label=encode(r["label"]),value=r["value"]) for r in x["page"]]) for x in inputs]
    bank=torch.from_numpy(np.unpackbits(np.frombuffer(b"".join(table),np.uint8)).reshape(-1,32,32)).to(device)
    result=[]
    for values,stop in predict(net,bank,items,device):
        tiles=[]
        for i,value in enumerate(values):
            if i: tiles.extend([bytes(128)]*2)
            tiles.extend(value)
        result.append((strip(tiles),len(tiles),stop))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    query=p.add_mutually_exclusive_group(required=True)
    query.add_argument("--question",type=Path,help="Pictured question as a 32-cell PNG grid.")
    query.add_argument("--prompt",help="Typeset a question into the same optical grid before inference.")
    for key in ("page","geometry","output"): p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    torch.set_num_threads(2)
    geometry=read(args.geometry)
    if args.prompt is not None:
        from scripts.prepare_region_writer import Ink
        renderer=Ink()
        handles=renderer.encode(args.prompt)
        if not handles or renderer.bad:
            p.error("The prompt is empty or contains a glyph missing from the selected print face")
        question=strip([renderer.tiles[i] for i in handles])
        geometry["question_cells"]=len(handles)
    else:
        question=args.question.read_bytes()
    evidence=decode(question,args.page.read_bytes(),geometry)
    blob,n,stop=answer(model("matched",args.device),[evidence],args.device)[0]
    args.output.write_bytes(blob)
    print(f"{n} cells, stopped={stop}")


if __name__=="__main__": main()

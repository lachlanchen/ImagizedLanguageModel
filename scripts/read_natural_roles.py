"""Read the two marked participant words in a rasterized English clause."""
import argparse
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from ilm.visual_lm.natural_roles import NaturalRoleReader,tiles
MODELS=Path(__file__).resolve().parents[1]/"models/natural_pictured_roles"
WINDOW=224


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def strip_png(ink):
    n=len(ink)
    cols=min(32,n);rows=(n+cols-1)//cols
    data=np.zeros((rows*cols,32,32),np.uint8)
    data[:n]=ink
    page=data.reshape(rows,cols,32,32).transpose(0,2,1,3).reshape(rows*32,cols*32)
    stream=BytesIO()
    Image.fromarray(255-page).save(stream,format="PNG")
    return stream.getvalue()


def load_strip(blob,n):
    with Image.open(BytesIO(blob)) as im:
        ink=255-np.asarray(im.convert("L"))
    h,w=ink.shape
    if h%32 or w%32 or not 1<=n<=min(WINDOW,h*w//1024):
        raise ValueError("Invalid source grid geometry or occupied cell count")
    return ink.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,32,32)[:n].copy()


def load(arm):
    folder=MODELS/arm
    if sha(folder/"weights.safetensors")!=read(folder/"fit.json")["weights"]:
        raise ValueError("Participant reader differs from its fit")
    model=NaturalRoleReader().eval()
    model.load_state_dict(load_file(str(folder/"weights.safetensors")))
    return model


@torch.inference_mode()
def choose(model,ink,verb,candidates,arm="context"):
    if len(candidates)!=2 or len(ink)>WINDOW:raise ValueError("Supply two candidate spans")
    marks=torch.zeros((1,len(ink)),dtype=torch.long)
    for j,(start,end) in enumerate((verb,*candidates),1):
        if not 0<=start<end<=len(ink) or marks[0,start:end].any():
            raise ValueError("Marks must be nonempty, disjoint intervals on the source grid")
        marks[0,start:end]=j
    scores=model(tiles(torch.from_numpy(ink))[None],marks,torch.ones_like(marks,dtype=torch.bool),arm=="layout")
    return int(scores.argmax(1)),scores[0].tolist()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source",required=True,type=Path);cli.add_argument("--cells",required=True,type=int)
    for name in ("verb","first","second"):
        cli.add_argument("--"+name,nargs=2,required=True,type=int,metavar=("START","END"))
    cli.add_argument("--actor",required=True,type=Path);cli.add_argument("--undergoer",required=True,type=Path)
    cli.add_argument("--compact-paper",action="store_true",help="Merge repeated blank cells before reading")
    args=cli.parse_args();torch.set_num_threads(2)
    ink=load_strip(args.source.read_bytes(),args.cells)
    spans=[args.first,args.second]
    source,verb,marks=ink,args.verb,spans
    if args.compact_paper:
        from ilm.visual_lm.paper_spacing import compact_paper
        source,verb,marks=compact_paper(ink,args.verb,spans)
    which,_=choose(load("context"),source,verb,marks)
    for dest,span in ((args.actor,spans[which]),(args.undergoer,spans[1-which])):
        dest.write_bytes(strip_png(ink[slice(*span)]))
    print(f"Selected candidate {which+1} as the active-subject counterpart")


if __name__=="__main__":main()

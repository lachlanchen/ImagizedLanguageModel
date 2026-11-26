"""Find both participant head words from one pictured English sentence and its marked verb."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from safetensors.torch import load_file

from ilm.visual_lm.natural_roles import tiles
from ilm.visual_lm.role_localization import RoleLocator,best_pair
from scripts.read_natural_roles import load_strip,strip_png,read,sha,WINDOW,MODELS
OUT=MODELS.parent/"visual_role_locator"


def render_sentence(text):
    font=ImageFont.truetype("DejaVuSansMono.ttf",24)
    cells=[]
    for char in text:
        im=Image.new("L",(32,32))
        pen=ImageDraw.Draw(im)
        left,_,right,_=pen.textbbox((0,0),char,font=font)
        pen.text((max(0,(32-right+left)//2-left),0),char,font=font,fill=255)
        cells.append(np.asarray(im))
    return np.stack(cells)


class PicturedLocator:
    def __init__(self,arm="context"):
        self.arm=arm
        folder=OUT/arm
        if sha(folder/"weights.safetensors")!=read(folder/"fit.json")["weights"]:
            raise ValueError("The locator weights differ from their fit receipt")
        self.model=RoleLocator().eval()
        self.model.load_state_dict(load_file(str(folder/"weights.safetensors")))

    @torch.inference_mode()
    def locate(self,ink,verb):
        start,end=verb
        if not 0<=start<end<=len(ink):
            raise ValueError("Provide a nonempty verb span within the pictured sentence")
        v=torch.zeros((1,len(ink)),dtype=torch.bool);v[:,start:end]=True
        feat=self.model.eye(tiles(torch.from_numpy(ink)))[None]
        logits=self.model.read(feat,v,torch.ones_like(v),self.arm=="position")[0].numpy()
        legal=ink.reshape(len(ink),-1).any(1)
        legal[start:end]=False
        return best_pair(logits,legal)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source",type=Path);inputs.add_argument("--text")
    p.add_argument("--cells",type=int)
    predicate=p.add_mutually_exclusive_group(required=True)
    predicate.add_argument("--verb",nargs=2,type=int,metavar=("START","END"))
    predicate.add_argument("--predicate",help="Unique printed verb in a typed sentence")
    p.add_argument("--actor",required=True,type=Path);p.add_argument("--undergoer",required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    verb=args.verb
    if args.source:
        if args.cells is None or verb is None:p.error("A PNG requires --cells and --verb START END")
        ink=load_strip(args.source.read_bytes(),args.cells)
    else:
        if not 1<=len(args.text)<=WINDOW:p.error(f"Supply 1 to {WINDOW} glyph cells")
        if args.predicate is not None:
            if not args.predicate or args.text.count(args.predicate)!=1:p.error("The predicate must occur once")
            start=args.text.index(args.predicate);verb=(start,start+len(args.predicate))
        ink=render_sentence(args.text)
    a,b=PicturedLocator().locate(ink,verb)
    args.actor.write_bytes(strip_png(ink[slice(*a)]));args.undergoer.write_bytes(strip_png(ink[slice(*b)]))
    print(f"Selected actor cells {a} and undergoer cells {b}")


if __name__=="__main__":main()

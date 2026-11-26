"""Translate a complete English or Chinese glyph image with optional predicted role fields."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.full_image_roles import FullImageWriter,EventMarker,pictured_input
from ilm.visual_lm.portable_writer import read_image,save_grid
from scripts.prepare_full_image_roles import BASE,Ink,read
from scripts.train_full_image_roles import loaded
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import stopped


class PicturedTranslator:
    def __init__(self,arm="marked",device="cpu"):
        if arm not in ("full","marked","displaced"):
            raise ValueError("Unknown reader treatment")
        self.device=device
        self.arm=arm
        self.writer=loaded(FullImageWriter(),"full" if arm=="full" else "marked",device)
        self.marker=None if arm=="full" else loaded(EventMarker(),"marker",device)
        self.palette=np.load(BASE/"palette.npy")
        self.index={v.tobytes():i for i,v in enumerate(self.palette) if i>=2}
        self.inventory=[torch.tensor(inv,device=device) for inv in read(BASE/"inventories.json")]

    @torch.inference_mode()
    def translate(self,images,direction):
        if not images or direction not in (0,1):
            raise ValueError("Supply source images and an English or Chinese output direction")
        # Unfamiliar physical source glyphs also go through the same optical network.
        lookup=self.index.copy()
        extra=[]
        rows=[]
        for image in images:
            if image.dtype!=np.uint8 or image.ndim!=2 or image.shape[1]!=128 or not 1<=len(image)<=96:
                raise ValueError("Supply 1 to 96 packed 32×32 binary glyphs per image")
            ids=[]
            for cell in image:
                raw=cell.tobytes()
                if raw not in lookup:
                    lookup[raw]=len(self.palette)+len(extra)
                    extra.append(cell)
                ids.append(lookup[raw])
            rows.append(ids)
        pal=np.concatenate((self.palette,np.stack(extra))) if extra else self.palette
        feat=torch.as_tensor(observations({"tiles":pal},"ink"),device=self.device)
        ids=pack(rows,self.device)
        fields=torch.zeros((*ids.shape,3),dtype=torch.uint8,device=self.device)
        if self.marker is not None:
            fields=(255*self.marker(feat,ids).sigmoid()*ids.ne(0).unsqueeze(-1)).round().byte()
            if self.arm=="displaced":
                for i,r in enumerate(rows):
                    fields[i,:len(r)]=fields[i,:len(r)].roll(len(r)//2,0)
        seq=self.writer.generate(feat,pictured_input(ids,fields),self.inventory[direction],direction)
        return [(self.palette[a],s) for a,s in (stopped(row) for row in seq.cpu())]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    inp=cli.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image",type=Path)
    inp.add_argument("--text")
    cli.add_argument("--cells",type=int)
    cli.add_argument("--to",required=True,choices=("en","zh"))
    cli.add_argument("--arm",choices=("marked","full","displaced"),default="marked")
    cli.add_argument("--output",required=True,type=Path)
    args=cli.parse_args()
    torch.set_num_threads(2)
    if args.image:
        if args.cells is None:
            cli.error("Provide the occupied cell count for the PNG")
        packed=read_image(args.image,args.cells)
    else:
        ink=Ink()
        ids=ink.encode(args.text)
        if ink.bad:
            cli.error("The print face lacks a supplied character")
        packed=ink.array()[ids]
    writer=PicturedTranslator(args.arm)
    (cells,stop),=writer.translate([packed],int(args.to=="zh"))
    save_grid(cells.tobytes(),args.output)
    print(f"Emitted {len(cells)} pictured cells. Natural stop {stop}.")


if __name__=="__main__":
    main()

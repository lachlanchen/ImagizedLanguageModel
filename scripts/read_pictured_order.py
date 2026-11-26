"""Select one supplied translation image when both candidates contain the same visual strips."""
import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.select_visual_translation import PACKAGE, VisualSelector, candidate_bits
from scripts.write_strip_restoration import StripWriter, ROOT, read, sha
from scripts.write_neural_glyphs import grid_bits


class PicturedOrderReader:
    """Score alternative orders in one parallel pass over an empty strip canvas."""
    def __init__(self, device="cpu"):
        self.device=device
        self.writer=StripWriter("mask",device)
        assets=read(PACKAGE/"assets.json")
        path=ROOT/"models/independent_optical_actions/tables.json"
        if sha(path)!=assets["tables"] or sha(PACKAGE/"merges.json")!=assets["files"]["merges.json"]:
            raise ValueError("Strip encoding differs from the released training repertoire")
        tables=read(path)["strip"]
        pictured=np.packbits(self.writer.ink.cpu().numpy().reshape(-1,1024),axis=1)
        self.atom=[{bytes(pictured[line[0]]):i for i,line in enumerate(table) if len(line)==1} for table in tables]
        self.merges=[{(a,b):(rank,c) for rank,(a,b,c) in enumerate(lines)}
                     for lines in read(PACKAGE/"merges.json")]
        with torch.inference_mode():
            self.actions=[self.writer.model.actions(self.writer.tiles,table) for table in self.writer.tables]

    @torch.inference_mode()
    def choose(self, source, alternatives, direction):
        if len(alternatives)<2 or not 1<=len(source)<=96:
            raise ValueError("Provide one pictured source and at least two candidates")
        orders=[VisualSelector.actions(self,img,direction) for img in alternatives]
        if not orders[0] or any(Counter(p)!=Counter(orders[0]) for p in orders):
            raise ValueError("Candidates must contain the same nonempty multiset of learned visual strips")
        body=self.writer.model
        raw=torch.as_tensor(source,device=self.device)
        memory,valid=body.encode(pool_tiles(raw[None]),torch.tensor([len(source)],device=self.device))
        blank=torch.zeros((1,len(orders[0])),dtype=torch.long,device=self.device)
        scores=body.position_scores(blank,torch.ones_like(blank,dtype=torch.bool),memory,valid,
                                    self.actions[direction],direction)[0]
        columns=torch.tensor(orders,device=self.device)-1
        # Subtract each first-candidate position before summing to avoid large common offsets.
        values=(scores.gather(1,columns.T)-scores.gather(1,columns[:1].T)).double().sum(0).tolist()
        choice=min(range(len(orders)),key=lambda i:(-values[i],orders[i]))
        return choice,values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--cells",required=True,type=int)
    p.add_argument("--to",required=True,choices=("en","zh"))
    p.add_argument("--candidate",nargs=2,metavar=("PNG","CELLS"),action="append",required=True)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    blobs=[Path(name).read_bytes() for name,_ in args.candidate]
    images=[candidate_bits(BytesIO(blob),int(pair[1])) for blob,pair in zip(blobs,args.candidate,strict=True)]
    index,scores=PicturedOrderReader().choose(grid_bits(args.source,args.cells),images,int(args.to=="zh"))
    args.output.write_bytes(blobs[index])
    print(f"Selected candidate {index+1} of {len(images)}; relative order scores {scores}")


if __name__=="__main__":
    main()

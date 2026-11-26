"""Select a supplied candidate PNG for a single pictured English or Chinese source."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.write_independent_optical_actions import ImageWriter,ROOT,BASE,read,sha,grid_bits

PACKAGE=ROOT/"models/source_supported_writing/selection"


def candidate_bits(file,n):
    with Image.open(file) as image:
        ink=np.asarray(image.convert("L"))<128
    h,w=ink.shape
    if h%32 or w%32 or not 0<=n<=min(128,h*w//1024):
        raise ValueError("A candidate must contain 0–128 occupied 32 × 32 cells")
    return ink.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,32,32)[:n]


class VisualSelector:
    def __init__(self,device="cpu"):
        self.device=device
        self.writer=ImageWriter("strip",device)
        receipt=read(PACKAGE/"assets.json")
        if any(sha(PACKAGE/k)!=v for k,v in receipt["files"].items()) or sha(BASE/"palette.npy")!=receipt["palette"]:
            raise ValueError("Selection assets changed")
        tables_path=ROOT/"models/independent_optical_actions/tables.json"
        if sha(tables_path)!=receipt["tables"]:
            raise ValueError("Visual action repertoire changed")
        tables=read(tables_path)["strip"]
        palette=np.load(BASE/"palette.npy")
        self.atom=[{bytes(palette[seq[0]]):i for i,seq in enumerate(table) if len(seq)==1} for table in tables]
        self.merges=[{(a,b):(rank,c) for rank,(a,b,c) in enumerate(pairs)} for pairs in read(PACKAGE/"merges.json")]
        self.bank=np.load(PACKAGE/"background.npz")["cells"]
        self.cache={}
        self.bodies={}
        with torch.inference_mode():
            for t in (0,1):
                self.bodies[t]=self.writer.model.actions(self.writer.tiles,self.writer.tables[t])

    def actions(self,ink,t):
        try:
            path=[self.atom[t][bytes(row)] for row in np.packbits(ink.reshape(len(ink),1024),axis=1)]
        except KeyError as exc:
            raise ValueError("Candidate contains an image outside this writer's target repertoire") from exc
        while len(path)>1:
            best=None
            for j,pair in enumerate(zip(path,path[1:])):
                rule=self.merges[t].get(pair)
                if rule is not None and (best is None or rule[0]<best[0]):
                    best=rule[0],j,rule[1]
            if best is None:
                break
            _,j,merged=best
            path[j:j+2]=[merged]
        return path

    def background(self,n,t):
        if (n,t) not in self.cache:
            grid=torch.tensor(self.bank[t,n-1].astype(int),device=self.device)
            lens=grid.ne(0).sum(1)
            self.cache[n,t]=self.writer.model.encode(self.writer.tiles[grid[:,:int(lens.max())]],lens)
        return self.cache[n,t]

    @torch.inference_mode()
    def score(self,source,candidates,t):
        if not 1<=len(source)<=96 or not candidates:
            raise ValueError("Supply a 1–96-cell source and at least one candidate")
        body=self.writer.model
        source=torch.as_tensor(source,device=self.device)
        memory,mask=body.encode(pool_tiles(source[None]),torch.tensor([len(source)],device=self.device))
        background,back_mask=self.background(len(source),t)
        history,keys=self.bodies[t]
        results=[]
        for ink in candidates:
            path=self.actions(ink,t)
            ids=torch.tensor(path,device=self.device,dtype=torch.long)
            previous=torch.cat((body.begin[t,None],history[ids]),0)[None]
            labels=torch.tensor(path+[0],device=self.device)
            def mass(mem,valid):
                x,_=body.decode(previous.expand(len(valid),-1,-1),mem,valid)
                logits=(x@keys.T)*keys.size(-1)**-.5
                lp=logits.log_softmax(-1).gather(-1,labels[None,:,None].expand(len(valid),-1,-1))
                return lp.double().sum((1,2))
            actual=mass(memory,mask).item()
            mean=mass(background,back_mask).mean().item()
            results.append(dict(raw=actual,average=actual/len(labels),background=mean,corrected=actual-mean,actions=len(path)))
        return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--cells",type=int,required=True)
    p.add_argument("--to",choices=("en","zh"),required=True)
    p.add_argument("--candidate",nargs=2,action="append",metavar=("PNG","CELLS"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    torch.set_num_threads(2)
    src=grid_bits(args.source,args.cells)
    imgs=[candidate_bits(path,int(n)) for path,n in args.candidate]
    scores=VisualSelector(args.device).score(src,imgs,int(args.to=="zh"))
    best=max(range(len(scores)),key=lambda i:scores[i]["corrected"])
    shutil.copyfile(args.candidate[best][0],args.output)
    print(json.dumps(dict(selected=best,scores=scores,output=str(args.output))))


if __name__=="__main__":
    main()

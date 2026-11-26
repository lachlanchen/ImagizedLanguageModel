"""Generate a complete answer PNG from source pixels with a bilingual optical-strip expert."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.full_image_roles import pictured_input
from ilm.visual_lm.optical_lexicon import StripInventory,StripPrefix,expand_table
from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.fit_optical_lexicon import BASE,MODEL,PUB,TABLE,Ink,read,sha
from scripts.train_ordered_translation import pack
from scripts.write_full_image_roles import PicturedTranslator


class LexicalWriter:
    def __init__(self,device="cpu"):
        self.neural=PicturedTranslator("marked",device)
        self.device=device
        tables=read(TABLE)["strip"]
        invs=read(BASE/"inventories.json")
        self.invs=invs
        self.strips=[StripInventory(z,inv) for z,inv in zip(tables,invs,strict=True)]
        self.models=[]
        for lang in ("en","zh"):
            folder=MODEL/lang
            assert sha(folder/"probabilities.npz")==read(folder/"fit.json")["weights"]
            with np.load(folder/"probabilities.npz") as z:
                prior=z["prior"].astype(np.float64)
                prior/=prior.sum()
                self.models.append((expand_table(z["ids"],z["mass"],z["tail"],prior.astype(np.float32)),prior))

    def laws(self,sources,t):
        table,prior=self.models[t]
        return [(StripPrefix(self.strips[t],table[self.strips[1-t].source_strips(ids)].mean(0)),
                 StripPrefix(self.strips[t],prior)) for ids in sources]

    @staticmethod
    def correction(states):
        # EOS has no lexical correction. Each lexical model conditions on a decision to emit ink.
        return np.stack([np.clip(np.log(np.maximum(s.probabilities()[1:],1e-30))-
                                 np.log(np.maximum(p.probabilities()[1:],1e-30)),-3,3) for s,p in states])

    @staticmethod
    def calibrated(logits,ratio,strength):
        ink=logits[:,1:]
        tilted=ink+ratio*strength
        offset=ink.logsumexp(-1,keepdim=True)-tilted.logsumexp(-1,keepdim=True)
        return torch.cat((logits[:,:1],tilted+offset),1)

    def inputs(self,feat,src):
        marker=self.neural.marker
        fields=(255*marker(feat,src).sigmoid()*src.ne(0).unsqueeze(-1)).round().byte()
        return pictured_input(src,fields)

    @torch.inference_mode()
    def write_ids(self,sources,t,feat,strength,donors=None):
        inv=self.neural.inventory[t]
        net=self.neural.writer
        src=pack(sources,self.device)
        bank=net.optical(feat)
        context,mask=net.encode(bank,self.inputs(feat,src))
        prototypes=net.prototypes(bank,inv)
        states=self.laws(sources if donors is None else donors,t)
        x=net.begin[t].expand(len(sources),1,-1)
        past=None
        result=[[] for _ in sources]
        done=[False]*len(sources)
        for j in range(129):
            h,past=net.decode(x,context,mask,past,j)
            scores=(h[:,0]@prototypes.T)/math.sqrt(h.size(-1))
            if strength:
                ratio=torch.as_tensor(self.correction(states),device=self.device,dtype=scores.dtype)
                scores=self.calibrated(scores,ratio,strength)
            choices=scores.argmax(-1)
            ids=inv[choices]
            for i,(selected,g) in enumerate(zip(choices.tolist(),ids.tolist(),strict=True)):
                if done[i]:
                    continue
                if g==1:
                    done[i]=True
                elif j<128:
                    result[i].append(g)
                    if strength:
                        for state in states[i]:
                            state.advance(selected,g)
            if all(done):
                break
            x=bank[ids][:,None]
        return list(zip(result,done,strict=True))

    @torch.inference_mode()
    def translate(self,images,t,strength):
        lookup=self.neural.index.copy()
        extra=[]
        rows=[]
        for image in images:
            if image.dtype!=np.uint8 or image.ndim!=2 or image.shape[1]!=128 or not 1<=len(image)<=96:
                raise ValueError("Expected 1 to 96 physical 32-pixel glyphs")
            ids=[]
            for cell in image:
                k=cell.tobytes()
                if k not in lookup:
                    lookup[k]=len(self.neural.palette)+len(extra)
                    extra.append(cell)
                ids.append(lookup[k])
            rows.append(ids)
        pal=np.concatenate((self.neural.palette,np.stack(extra))) if extra else self.neural.palette
        f=torch.tensor(observations({"tiles":pal},"ink"),device=self.device)
        return [(self.neural.palette[a],s) for a,s in self.write_ids(rows,t,f,strength)]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",type=Path)
    group.add_argument("--text")
    cli.add_argument("--cells",type=int)
    cli.add_argument("--to",required=True,choices=("en","zh"))
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(2)
    if args.image:
        if args.cells is None:
            cli.error("Supply the occupied source cell count")
        image=read_image(args.image,args.cells)
    else:
        face=Ink()
        ids=face.encode(args.text)
        if face.bad:
            cli.error("Source font has missing glyphs")
        image=face.array()[ids]
    (out,stop),=LexicalWriter().translate([image],int(args.to=="zh"),read(PUB/"calibration.json")["strength"])
    save_grid(out.tobytes(),args.output)
    print(f"Emitted {len(out)} cells. Natural stop {stop}.")


if __name__=="__main__":
    main()

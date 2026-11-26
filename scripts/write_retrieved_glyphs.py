"""Autoregressively write physical glyphs using bilingual training passages selected by the source image."""
import argparse
from collections import Counter,defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.covered_phrases import character_fscore
from ilm.visual_lm.optical_lexicon import StripInventory
from ilm.visual_lm.portable_writer import read_image,save_grid
from scripts.prepare_retrieved_glyph_writer import PUB,WORK,RULE,base,prior
from scripts.probe_visual_passage_memory import index,neighbors
from scripts.write_glyph_draft_editor import PictureEditor,pictured_input,pack,stopped
from scripts.write_covered_phrases import CoveredWriter
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


@dataclass
class Support:
    ids: list
    targets: list
    votes: dict


class PassageWriter:
    def __init__(self,direction,device='cpu',neural=None):
        self.t=direction
        self.device=device
        s=1-direction
        start=monotonic()
        self.neural=neural or PictureEditor('editor',device)
        self.phrase=CoveredWriter(direction)
        tables=base.read(base.TABLE)['strip']
        inv=base.read(base.BASE/'inventories.json')
        self.lookup=np.full(len(self.neural.palette),-1,np.int32)
        self.lookup[inv[direction]]=np.arange(len(inv[direction]))
        self.strip=StripInventory(tables[s],inv[s])
        self.table=tables[direction]
        paths=[prior.WORK/f'strips-{j}.npy' for j in (0,1)]
        assert [base.sha(p) for p in paths]==base.read(prior.PUB/'corpus.json')['strips']
        self.target=np.load(paths[direction],mmap_mode='r')
        sources=np.load(paths[s],mmap_mode='r')
        self.matrix,self.idf,self.length=index(sources,len(tables[s]))
        self.bins={k:np.flatnonzero(self.length//8==k) for k in np.unique(self.length//8)}
        self.index_seconds=monotonic()-start
        self.index_bytes=sum(x.nbytes for x in (self.matrix.data,self.matrix.indices,self.matrix.indptr,self.idf,self.length,*self.bins.values()))

    def support(self,source,mode):
        query=self.strip.source_strips(source)[1:]
        ids=neighbors(query,self.matrix,self.idf)
        if mode=='random':
            seed=int.from_bytes(sha256(np.asarray(source,dtype='<u2').tobytes()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            draws=[]
            for j in ids:
                choices=self.bins[self.length[j]//8]
                while True:
                    k=int(rng.choice(choices))
                    if k not in ids and k not in draws:break
                draws.append(k)
            ids=draws
        elif mode!='related':
            raise ValueError('Expected related or random support')
        votes=defaultdict(Counter)
        targets=[]
        for j in ids:
            seq=[g for k in self.target[j] if k for g in self.table[k]]
            targets.append(seq)
            hist=(0,0,0)
            for g in seq+[1]:
                votes[hist][g]+=1
                hist=(*hist[1:],g)
        data={}
        for h,c in votes.items():
            local=self.lookup[np.array(list(c))]
            assert (local>=0).all()
            data[h]=(local,np.array(list(c.values()),np.float32),sum(c.values()))
        return Support(list(map(int,ids)),targets,data)

    @torch.inference_mode()
    def write(self,sources,drafts,supports,strength,limit=128):
        net=self.neural
        model=net.model
        ids=pack(sources,self.device)
        field=(255*net.marker(net.features,ids).sigmoid()*ids.ne(0).unsqueeze(-1)).round().byte()
        bank=model.optical(net.features)
        context,valid=model.encode(bank,pictured_input(ids,field))
        inventory=net.inv[self.t]
        draft=pack([r or [0] for r in drafts],self.device)
        memory=model.draft_memory(bank,draft,net.local[self.t])
        prototypes=model.prototypes(bank,inventory)
        x=model.begin[self.t].expand(len(ids),1,-1)
        local_hist=[(0,0,0)]*len(ids)
        past=None
        done=[False]*len(ids)
        outputs=[[] for _ in done]
        for offset in range(limit+1):
            h,past=model.decode(x,context,valid,past,offset)
            logp=model.distribution(h,prototypes,memory)[:,0]
            if strength:
                # Rescale by 1/(1-gate) to keep the neural masses intact.
                # Only candidate memory cells receive an additional term.
                prob=logp.exp()
                rows,columns,mass=[],[],[]
                for j,(history,sup) in enumerate(zip(local_hist,supports,strict=True)):
                    found=sup.votes.get(history)
                    if done[j] or found is None:continue
                    target,counts,n=found
                    rows.extend([j]*len(counts))
                    columns.extend(target)
                    mass.extend(counts*(strength/(2+n*(1-strength))))
                if mass:
                    r=torch.as_tensor(rows,device=self.device)
                    c=torch.as_tensor(np.asarray(columns),dtype=torch.long,device=self.device)
                    prob[r,c]+=torch.as_tensor(np.asarray(mass),device=self.device)
                pick=prob.argmax(-1)
            else:
                pick=logp.argmax(-1)
            selected=inventory[pick]
            for j,g in enumerate(selected.tolist()):
                if not done[j]:
                    outputs[j].append(g)
                    done[j]=g==1
                    local_hist[j]=(*local_hist[j][1:],g)
            if all(done):break
            x=bank[selected][:,None]
        return [stopped(torch.tensor(seq),limit) for seq in outputs]

    def translate(self,imgs,mode='related',strength=None):
        if not imgs or any(not 1<=len(img)<=96 for img in imgs):
            raise ValueError('Expected 1–96 occupied source glyphs per image')
        if strength is None:strength=base.read(PUB/'calibration.json')['chosen'][self.t]
        src=[self.neural.addresses(img) for img in imgs]
        drafts=[self.neural.addresses(self.phrase.translate(img)[0]) for img in imgs]
        supports=[self.support(r,mode) for r in src]
        return [(self.neural.palette[x],stop) for x,stop in self.write(src,drafts,supports,strength)]


def calibrate():
    assert base.read(PUB/'selection.json')['protocol']==base.sha(RULE)
    dev=base.read(base.WORK/'dev.json')
    strengths=[0.,.35,.70,1.]
    scores=[]
    chosen=[]
    inds=[]
    net=PictureEditor('editor','cuda')
    for t in (0,1):
        w=PassageWriter(t,'cuda',net)
        inds.append(dict(seconds=w.index_seconds,bytes=w.index_bytes))
        total=np.zeros(len(strengths))
        for j in range(0,len(dev),24):
            records=dev[j:j+24]
            src=[r['cells'][1-t] for r in records]
            ds=[net.addresses(w.phrase.translate(net.palette[s])[0]) for s in src]
            memory=[w.support(s,'related') for s in src]
            for k,alpha in enumerate(strengths):
                for r,(out,stop) in zip(records,w.write(src,ds,memory,alpha),strict=True):
                    total[k]+=character_fscore(r['cells'][t],out) if stop else 0.
        scores.append((total/len(dev)).tolist())
        chosen.append(strengths[int(total.argmax())])
    base.dump(PUB/'calibration.json',dict(strengths=strengths,chosen=chosen,scores=scores,samples=len(dev),indices=inds,
              dataset=base.sha(base.WORK/'dev.json'),selection=base.sha(PUB/'selection.json'),implementation=base.sha(Path(__file__))))
    print('strength',chosen,'scores',scores,flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    inp=cli.add_mutually_exclusive_group(required=True)
    inp.add_argument('--image',type=Path)
    inp.add_argument('--text')
    inp.add_argument('--calibrate',action='store_true')
    cli.add_argument('--cells',type=int)
    cli.add_argument('--to',choices=('en','zh'))
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    if args.calibrate:
        configure_cuda_cli()
        with gpu_owner():calibrate()
        return
    if args.to is None or args.output is None:cli.error('Supply --to and --output')
    if args.image:
        if args.cells is None:cli.error('Supply --cells for the source PNG')
        img=read_image(args.image,args.cells)
    else:
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad:cli.error('Source glyph outside the trained print face')
        img=face.array()[ids]
    torch.set_num_threads(3)
    p,stop=PassageWriter(int(args.to=='zh')).translate([img])[0]
    save_grid(p.tobytes(),args.output)
    print(f'Wrote {len(p)} glyphs; learned stop={stop}.')


if __name__=='__main__':main()

"""Return a complete word image using measured page spans and a frozen optical prior."""
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ilm.visual_lm.book_ink_inference import BookInkWriter
from ilm.visual_lm.portable_writer import read_text, read_image, save_grid, DEFAULT_FONT


class PageWordMemory:
    """Addresses below refer to physical motor bitmaps. Missing input shapes split lexical spans."""
    def __init__(self,device='cpu'):
        self.writer=BookInkWriter(device=device)
        self.device=device
        self.motor=self.writer.motor
        self.ids={row.tobytes():i for i,row in enumerate(self.motor) if i>0}
        self.alpha=set(self.encode(read_text('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',DEFAULT_FONT)))
        self.space=self.encode(read_text(' ',DEFAULT_FONT))[0]
        self.neutral={}
        self.symbols={}
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'-":
            idx=self.encode(read_text(c,DEFAULT_FONT))[0]
            if idx>0:self.symbols[idx]=c
        assert -1 not in self.alpha and len(self.alpha)==52

    def encode(self,image):
        return tuple(self.ids.get(c.tobytes(),-1) for c in image)

    def spans(self,seq):
        words={}
        start=0
        for i,v in enumerate((*seq,-1)):
            if v in self.alpha:continue
            if 0<i-start<=23:
                word=seq[start:i]
                words.pop(word,None)
                words[word]=None
            start=i+1
        return list(reversed(words))

    @torch.inference_mode()
    def scores(self,prefixes,words):
        if not words:return []
        width=len(prefixes[0])
        assert all(len(p)==width for p in prefixes)
        rows=[p+w+(self.space,) for p,w in zip(prefixes,words,strict=True)]
        y=pad_sequence([torch.tensor(row) for row in rows],batch_first=True).to(self.device)
        net=self.writer.model
        x=torch.cat((net.begin[0].expand(len(y),1,-1),self.writer.bank[y[:,:-1]]),1)
        z=x.new_zeros(len(y),4,1,x.size(-1)//4)
        h,_=net.decode(x,[(z,z)]*len(net.dec),None)
        law=net.distribution(h[:,width:],self.writer.prototypes,None)
        ids=y[:,width:]
        values=law.gather(-1,ids[...,None]).squeeze(-1)
        return ((values*ids.ne(0)).double().sum(1)/ids.ne(0).sum(1)).tolist()

    def rank_many(self,queries,banks,free):
        """Rank candidates once per original query shared by its alternative page conditions."""
        work=[]
        choices=[]
        for i,(conditions,novel) in enumerate(zip(banks,free,strict=True)):
            options={k:list(dict.fromkeys([*words,*([novel] if novel else [])])) for k,words in conditions.items()}
            choices.append(options)
            work.extend((i,w) for w in dict.fromkeys(w for candidates in options.values() for w in candidates))
        missing=list(dict.fromkeys(w for _,w in work if w not in self.neutral))
        for j in range(0,len(missing),256):
            batch=missing[j:j+256]
            self.neutral.update(zip(batch,self.scores([(self.space,)]*len(batch),batch),strict=True))
        scores=[{} for _ in queries]
        for j in range(0,len(work),256):
            batch=work[j:j+256]
            ll=self.scores([queries[i] for i,_ in batch],[w for _,w in batch])
            for (i,w),x in zip(batch,ll,strict=True):
                scores[i][w]=x-self.neutral[w]
        return [{k:max(v,key=scores[i].get) if v else () for k,v in conditions.items()}
                for i,conditions in enumerate(choices)]

    def first_word(self,seq,stop):
        text=''.join(self.symbols.get(i,' ') for i in seq)
        m=re.match("[A-Za-z]+(?:['-][A-Za-z]+)*",text)
        if m and (m.end()<len(text) or stop) and m.end()<=23:return tuple(seq[:m.end()])
        return ()

    def write(self,image):
        page=self.encode(image)
        if len(page)<32 or any(i<0 for i in page[-32:]):
            raise ValueError('The final query must contain 32 supported pictured cells')
        out,stop=self.writer.write(image[-32:],24)
        free=self.first_word(self.encode(out),stop)
        predicted=self.rank_many([page[-32:]],[{'page':self.spans(page)}],[free])[0]['page']
        return self.motor[list(predicted)]


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    src=cli.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',type=Path)
    src.add_argument('--text')
    cli.add_argument('--cells',type=int)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(2)
    image=read_image(args.image,args.cells) if args.image else read_text(args.text,DEFAULT_FONT)
    out=PageWordMemory().write(image)
    save_grid(out.tobytes(),args.output)
    print(f'{len(out)} output cells')

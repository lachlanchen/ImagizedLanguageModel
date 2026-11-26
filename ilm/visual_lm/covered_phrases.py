"""Phrase emission with explicit coverage of the physical source strips."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np


def unpack(code):
    result=[]
    while code:
        result.append(int(code)&8191)
        code=int(code)>>13
    return tuple(reversed(result))


@dataclass(frozen=True)
class Choice:
    start: int
    end: int
    target: tuple
    evidence: int
    log_translation: float
    internal_lm: float


class PhraseInventory:
    def __init__(self,path,vocab_size,backoff):
        self.backoff=backoff
        self.boundary=vocab_size
        with np.load(path) as z:
            self.source=z["source"]
            self.target=z["target"]
            self.count=z["count"]
            self.weight=np.log(z["prob"])+.35*np.log(z["reverse"])
            counts=np.zeros((vocab_size+1,vocab_size+1),np.float32)
            counts[z["left"],z["right"]]=z["bigram"]
        total=counts.sum(0,dtype=np.float64)
        total[1:]+=.01
        prior=total/total.sum()
        types=np.maximum(1,(counts>0).sum(1))[:,None]
        smooth=(counts+types*prior)/(counts.sum(1,dtype=np.float64)[:,None]+types)
        self.lm=np.log(np.maximum(smooth,1e-30)).astype(np.float32)

    @lru_cache(maxsize=32000)
    def phrases(self,key):
        a,b=np.searchsorted(self.source,key,side="left"),np.searchsorted(self.source,key,side="right")
        options=[(unpack(code),int(count),float(weight)) for code,count,weight in
                 zip(self.target[a:b],self.count[a:b],self.weight[a:b],strict=True)]
        if key<8192 and not options:
            # Every seen single strip can back off to the earlier optical emission table.
            distribution=self.backoff[key]
            ids=np.argsort(-distribution[1:],kind="stable")[:4]+1
            options.extend(((int(i),),0,math.log(float(distribution[i]))-3.) for i in ids)
        return options

    def lattice(self,source):
        options=[]
        for a in range(len(source)):
            start=[]
            key=0
            for b in range(a,min(len(source),a+4)):
                key=(key<<13)+int(source[b])
                for phrase,count,logp in self.phrases(key):
                    lm=sum(float(self.lm[p,q]) for p,q in zip(phrase,phrase[1:]))
                    start.append(Choice(a,b+1,phrase,count,logp,lm))
            options.append(start)
        return options

    def decode(self,source,fluency,reward,beam=12,lattice=None):
        if not source:
            raise ValueError("Supply at least one pictured source strip")
        lattice=self.lattice(source) if lattice is None else lattice
        n=len(source)
        complete=(1<<n)-1
        # All states in one stack have consumed the same number of source positions.
        stacks=[{} for _ in range(n+1)]
        stacks[0][(0,self.boundary,0)]=(0.,())
        for consumed in range(n):
            best=sorted(stacks[consumed].items(),key=lambda kv:-kv[1][0])[:beam]
            for (mask,last,previous_end),(score,path) in best:
                remaining=complete^mask
                front=(remaining & -remaining).bit_length()-1
                for a in range(front,min(front+3,n)):
                    for c in lattice[a]:
                        length=c.end-a
                        covered=((1<<length)-1)<<a
                        if mask & covered:
                            continue
                        # A leap may pass at most two unconsumed source positions.
                        newmask=mask|covered
                        next_score=(score+c.log_translation-.6-.75*abs(a-previous_end)
                              +fluency*(float(self.lm[last,c.target[0]])+c.internal_lm)
                              +reward*len(c.target))
                        if newmask==complete:
                            next_score+=fluency*float(self.lm[c.target[-1],self.boundary])
                        state=(newmask,c.target[-1],c.end)
                        dest=stacks[consumed+length]
                        if state not in dest or next_score>dest[state][0]:
                            dest[state]=(next_score,path+(c,))
            stacks[consumed].clear()
        if not stacks[n]:
            raise RuntimeError("No complete source-covering translation")
        score,path=max(stacks[n].values(),key=lambda x:x[0])
        return tuple(q for c in path for q in c.target),path,score


def character_fscore(reference,prediction):
    scores=[]
    for length in range(1,5):
        grams=lambda seq: Counter(tuple(seq[i:i+length]) for i in range(len(seq)-length+1))
        target,actual=grams(reference),grams(prediction)
        if target or actual:
            same=sum((target&actual).values())
            scores.append(5*same/max(1,4*sum(target.values())+sum(actual.values())))
    return sum(scores)/max(1,len(scores))

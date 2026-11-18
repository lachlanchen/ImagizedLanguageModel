"""Choose a complete glyph-stream restoration using the two visible flanks and evidence."""
from pathlib import Path
import numpy as np


def tiles(packed):
    if len(packed)%128:
        raise ValueError("Expected whole 32-pixel binary glyphs")
    return [packed[j:j+128] for j in range(0,len(packed),128)]


class SpanBackoff:
    def __init__(self,folder):
        self.glyphs=np.load(Path(folder)/"glyphs.npy")
        self.index={bytes(v):i for i,v in enumerate(self.glyphs) if i}
        with np.load(Path(folder)/"neighbors.npz") as data:
            self.keys=data["keys"];self.tails=data["spans"]
        self.size=len(self.glyphs)
        if not np.all(np.diff(self.keys)>0):
            raise ValueError("Unsorted neighbor table")

    def lookup(self,left,right):
        l=self.index.get(left,-1) if left is not None else 0
        r=self.index.get(right,-1) if right is not None else 0
        n=self.size
        candidates=[(l*n+r if min(l,r)>=0 else -1,0),(n*n+l if l>=0 else -1,1),
                    (n*n+n+r if r>=0 else -1,2),(n*n+2*n,3)]
        for k,level in candidates:
            i=np.searchsorted(self.keys,k)
            if i<len(self.keys) and self.keys[i]==k:
                row=self.tails[i]
                return self.glyphs[row[row!=0]].tobytes(),level
        raise ValueError("Global fallback absent")

    @staticmethod
    def support(candidate,stream,left,right):
        n=len(candidate)
        if not n:
            return 0,0
        copies=anchors=0
        for k in range(len(stream)-n+1):
            if stream[k:k+n]!=candidate:
                continue
            copies+=1
            count=int(k>0 and stream[k-1]==left)+int(k+n<len(stream) and stream[k+n]==right)
            anchors=max(count,anchors)
        return copies,anchors

    def features(self,passage,claim,mask,neural,stopped):
        claim=tiles(claim);source=tiles(passage)
        if claim.count(mask)!=1:
            raise ValueError("One visual gap required")
        k=claim.index(mask)
        left=claim[k-1] if k else None
        right=claim[k+1] if k+1<len(claim) else None
        backoff,order=self.lookup(left,right)
        features=[1.,float(stopped),float(order==0),float(order==1),float(order==2)]
        for raw in (neural,backoff):
            seq=tiles(raw)
            count,anchors=self.support(seq,source,left,right)
            repeat,_=self.support(seq,claim,left,right)
            features.extend([min(len(seq),5),float(len(seq)==1),np.log1p(count),anchors,float(repeat>0)])
        return backoff,np.asarray(features,np.float64)

    def choose(self,passage,claim,mask,neural,stopped,coeff):
        alternate,features=self.features(passage,claim,mask,neural,stopped)
        if features@coeff>0:
            return alternate,True
        return neural,stopped

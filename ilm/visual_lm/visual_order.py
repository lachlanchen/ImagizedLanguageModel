"""Order two selected source images from the pictured words around their label mentions."""
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import blake2b
from pathlib import Path

import numpy as np

WIDTH = 32768
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/"models/origin_instructions/visual_order"
# Image identifiers hash physical binary tiles. Neither sorting nor feature extraction uses character codes.
@lru_cache(maxsize=18000)
def mark(tile):
    return blake2b(tile,digest_size=8).digest()


def blocks(query,label,other):
    unique=set(label[1:])-set(other[1:])
    blank=bytes(128)
    matches=[]
    for a,_,n in SequenceMatcher(None,query,label[1:],autojunk=False).get_matching_blocks():
        while n and query[a]==blank:
            a+=1
            n-=1
        while n and query[a+n-1]==blank:
            n-=1
        informative=sum(c in unique for c in query[a:a+n])
        if informative:
            matches.append((a,a+n,informative,n))
    return matches


def skeleton(query,first,second):
    """Locate nonoverlapping row-specific label evidence, retaining physical mention order."""
    a=blocks(query,first,second)
    b=blocks(query,second,first)
    best=None
    for left in a:
        for right in b:
            if left[1]>right[0] and right[1]>left[0]:
                continue
            weight=(min(left[2],right[2]),left[2]+right[2],left[3]+right[3],-min(left[0],right[0]))
            if best is None or weight>best[0]:
                best=(weight,left,right)
    if best is None:
        return None
    _,l,r=best
    first_earlier=l[0]<r[0]
    if not first_earlier:
        l,r=r,l
    head={first[0],second[0]}
    zones=[query[:l[0]],query[l[1]:r[0]],query[r[1]:]]
    return tuple(tuple(b'HEADMARK' if c in head else mark(c) for c in zone) for zone in zones),first_earlier


@lru_cache(maxsize=20000)
def features(zones):
    active=set()
    for location,zone in enumerate(zones):
        for i in range(len(zone)):
            for n in range(1,min(5,len(zone)-i)+1):
                key=b''.join(zone[i:i+n])
                for region in (location,3):
                    digest=blake2b(bytes([region])+key,digest_size=4).digest()
                    active.add(int.from_bytes(digest,'little')%WIDTH)
    return tuple(sorted(active))


class VisualOrder:
    def __init__(self, folder=MODEL):
        import json
        from hashlib import sha256
        path=folder/"order.npz"
        if sha256(path.read_bytes()).hexdigest()!=json.loads((folder/"fit.json").read_text())["weights"]:
            raise ValueError("Visual order weights changed")
        with np.load(path) as weights:
            self.coef=weights["coef"]
            self.bias=float(weights["bias"].item())

    def reorder(self,query,page,rows):
        if len(rows)!=2 or rows[0]==rows[1]:
            return rows
        found=skeleton(query,page[rows[0]]["label"],page[rows[1]]["label"])
        if found is None:
            return rows
        zones,first_earlier=found
        direct=self.bias+self.coef[list(features(zones))].sum()>=0
        return rows if bool(direct)==first_earlier else rows[::-1]

"""Generate complete pictured sentences shared by independently authored reading questions."""
from collections import defaultdict
from itertools import combinations

import numpy as np

from scripts.train_inverse_role_writer import WHO, Tiles


class SentenceCorpus:
    def __init__(self, ds, bank, split):
        self.data = ds
        ends = set(Tiles(bank)("。！？!?"))
        self.bounds = []
        for p in ds.passages:
            cuts = sorted({j+1 for j,x in enumerate(p) if x in ends} | {len(p)})
            self.bounds.append(list(zip([0]+cuts[:-1],cuts,strict=True)))
        with np.load(WHO/"cohort.npz") as z:
            self.who = z[split]
            self.who_pairs = self.who[z[split+"_pairs"]]
        people = {tuple(pair) for pair in self.who_pairs.tolist()}
        within = defaultdict(list)
        targets = []
        for i,g in enumerate(ds.groups):
            spans = ds.spans[i]
            p,q = ds.passages[g], ds.questions[i]
            targets.append({p[a:b].tobytes() for a,b in spans})
            for j,(l,r) in enumerate(self.bounds[g]):
                if any(l<=a and b<=r for a,b in spans):
                    within[g,j].append(i)
        candidates = defaultdict(list)
        for (g,j), qs in within.items():
            for a,b in combinations(qs,2):
                if targets[a].isdisjoint(targets[b]) and not np.array_equal(ds.questions[a],ds.questions[b]):
                    candidates[a,b].append(j)
        self.pairs, self.people = [], []
        for (a,b),js in candidates.items():
            g = ds.groups[a]
            l,r = min((self.bounds[g][j] for j in js),key=lambda s:(s[1]-s[0],s[0]))
            item=(a,b,l,r)
            self.pairs.append(item)
            if (a,b) in people:
                self.people.append(item)
        assert len(self.people)==(190 if split=="train" else 27)

    def sample(self,i,l,r,question=None):
        ds = self.data
        p = ds.passages[ds.groups[i]]
        spans=[(a-l,b-l) for a,b in ds.spans[i] if l<=a and b<=r and b-a<=64]
        return p[l:r],ds.questions[i if question is None else question],spans

    def common(self,i,epoch):
        g = self.data.groups[i]
        regions=[(l,r) for l,r in self.bounds[g] if any(l<=a and b<=r for a,b in self.data.spans[i])]
        if regions and epoch%2==0:
            l,r=min(regions,key=lambda s:(s[1]-s[0],s[0]))
        else:
            l,r=0,len(self.data.passages[g])
        return self.sample(i,l,r)

    def pair(self,record,exchange=False):
        a,b,l,r=record
        return (self.sample(a,l,r,b if exchange else a), self.sample(b,l,r,a if exchange else b))

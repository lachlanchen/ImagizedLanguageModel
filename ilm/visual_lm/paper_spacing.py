"""Canonicalize unmarked paper while retaining exact source-coordinate ownership."""
import numpy as np


def compact_paper(ink, verb, candidates):
    spans=[verb,*candidates]
    marked=np.zeros(len(ink),dtype=bool)
    for a,b in spans:
        if not 0<=a<b<=len(ink) or marked[a:b].any():
            raise ValueError("Expected nonempty disjoint marked spans")
        marked[a:b]=True
    paper=~ink.reshape(len(ink),-1).any(axis=1) & ~marked
    keep=np.ones(len(ink),dtype=bool)
    keep[1:]=~(paper[1:] & paper[:-1])
    boundaries=np.r_[0,np.cumsum(keep)]
    mapped=[(int(boundaries[a]),int(boundaries[b])) for a,b in spans]
    for old,new in zip(spans,mapped,strict=True):
        assert np.array_equal(ink[slice(*old)],ink[keep][slice(*new)])
    return ink[keep],mapped[0],mapped[1:]

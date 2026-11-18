"""Check sorted physical histories against a direct recount within original answer boundaries."""
from collections import Counter
import numpy as np
from scripts.prepare_visual_answer_language import DEST,SEED
from scripts.prepare_pictured_entry_answers import load,write_json
from scripts.train_composed_answer_writer import ROOT


def audit():
    original=np.load(DEST/"sequence.npy")
    lengths=np.load(DEST/"lengths.npy")
    limits=np.r_[0,lengths.cumsum()]
    base=len(np.load(DEST/"tiles.npy",mmap_mode="r"))
    locations=[(k,int(i),min(k,int(lengths[i]-1))) for k in range(1,5)
               for i in np.random.default_rng(19480+k).choice(len(lengths),4,replace=False)]
    reports={}
    for arm in ("ordered","shuffled"):
        seq=original.copy()
        if arm=="shuffled":
            rng=np.random.default_rng(SEED)
            for l,r in zip(limits[:-1],limits[1:]):rng.shuffle(seq[l:r])
        keys=np.load(DEST/arm/"keys.npy",mmap_mode="r")
        successors=np.load(DEST/arm/"next.npy",mmap_mode="r")
        assert np.all(keys[1:]>=keys[:-1]) and len(keys)==len(seq)==len(successors)
        assert np.array_equal(np.bincount(original),np.bincount(successors))
        assert int((keys==0).sum())==len(lengths)
        seen=[]
        for k,i,offset in locations:
            at=limits[i]+offset
            prefix=tuple(map(int,seq[at-k:at]))
            assert len(prefix)==k
            key=0
            for b in reversed(prefix):key=key*base+b
            w=base**(4-k)
            l,r=keys.searchsorted([key*w,(key+1)*w])
            indexed=Counter(map(int,successors[l:r]))
            ends=np.flatnonzero(seq==prefix[-1])+1
            ends=ends[(ends>=k)&(ends<len(seq))]
            for j,b in enumerate(prefix[:-1]):ends=ends[seq[ends-k+j]==b]
            ids=np.searchsorted(limits[1:],ends,side="right")
            ends=ends[ends-k>=limits[ids]]
            direct=Counter(map(int,seq[ends]))
            assert indexed==direct
            seen.append(len(ends))
        reports[arm]=dict(prefix_lengths_checked=[1,2,3,4],counts=seen,
                          positions=len(seq),unigram_counts_exact=True,
                          start_of_answer_keys=len(lengths))
    return reports


if __name__=="__main__":
    target=ROOT/"references/evidence/bitmap_history_20261119/index_audit.json"
    report=audit();write_json(target,report)

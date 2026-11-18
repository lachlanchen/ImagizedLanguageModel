"""Check exact physical questions and long passage runs against the inherited training corpus."""
from collections import defaultdict
from time import monotonic
import numpy as np
from numba import njit

from scripts.audit_same_passage_writer import EVIDENCE
from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_grounded_glyph_writer import read_large, small
from scripts.train_same_passage_writer import PairCorpus


@njit
def window_hashes(seq, width):
    base = np.uint64(1000003)
    h = np.uint64(0)
    scale = np.uint64(1)
    for j in range(width):
        h = h*base + np.uint64(seq[j])
        scale *= base
    out = np.empty(len(seq)-width+1,np.uint64)
    out[0] = h
    for j in range(1,len(out)):
        h = h*base + np.uint64(seq[j+width-1]) - scale*np.uint64(seq[j-1])
        out[j] = h
    return out


def scan(sequences, lookup, hashes):
    matched = set()
    for start in range(0,len(sequences),4096):
        # Positive bitmap addresses exclude the zero separator from every reference window.
        batch = sequences[start:start+4096]
        data = np.zeros(sum(map(len,batch))+len(batch),np.uint16)
        j=0
        for seq in batch:
            data[j:j+len(seq)] = seq
            j += len(seq)+1
        if len(data)<32:
            continue
        hs = window_hashes(data,32)
        pos = np.minimum(np.searchsorted(hashes,hs),len(hashes)-1)
        for k in np.flatnonzero(hashes[pos]==hs):
            key = data[k:k+32].tobytes()
            matched.update(lookup.get(key,()))
    return matched


def main():
    start=monotonic()
    bank, large=read_large()
    adapt,_=small(bank,"train")
    dev=PairCorpus(bank,"development")
    assert len(bank.tiles)<65536
    questions={np.asarray(dev.questions[i],np.uint16).tobytes():int(i) for i in np.unique(dev.rows[:,1:3])}
    ids=np.unique(dev.rows[:,0])
    lookup=defaultdict(set)
    hashes=set()
    for i in ids:
        p=dev.passages[i].astype(np.uint16)
        hs=window_hashes(p,32)
        for k in range(0,len(hs),32):
            lookup[p[k:k+32].tobytes()].add(int(i))
            hashes.add(int(hs[k]))
    keys=np.asarray(sorted(hashes),np.uint64)
    question_hits=set()
    groups={}
    for label, ds in (("large",large),("adapt",adapt)):
        for q in ds.q:
            found=questions.get(np.asarray(q,np.uint16).tobytes())
            if found is not None: question_hits.add(found)
        groups[label]={k:sorted(scan(seqs,lookup,keys)) for k,seqs in
                       (("passages",ds.p),("questions",ds.q),("answers",ds.y))}
    EVIDENCE.mkdir(exist_ok=True)
    matched=set(g for ds in groups.values() for gs in ds.values() for g in gs)
    report=dict(prior_training_sizes=[len(large.q),len(adapt.q)],development_questions=len(np.unique(dev.rows[:,1:3])),
                development_passages=len(ids),exact_prior_question_shapes=len(question_hits),
                passages_with_a_matched_32_cell_anchor=len(matched),
                counts_by_source={k:{s:len(v) for s,v in ds.items()} for k,ds in groups.items()},
                distinct_anchors=len(lookup),seconds=monotonic()-start,script_sha256=sha(__file__))
    write_json(EVIDENCE/"prior_overlap.json",report)
    print(report)


if __name__=="__main__":
    main()

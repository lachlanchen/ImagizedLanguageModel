"""Audit distribution support, source overlap and first-passage PNG reproduction."""
import json
import numpy as np
from scripts.evaluate_book_prefix import held
from scripts.prepare_book_prefix import corpus,MODEL,PUB,WORK
from scripts.prepare_dialogue_ink import dump
from scripts.write_book_prefix import write
from ilm.visual_lm.portable_writer import save_grid,read_image
from ilm.visual_lm.prefix_counts import PrefixCounts,history_windows
import torch


def main():
    meta,d = corpus()
    _,works,_,prefix,_,truth = held(meta,d["stream"])
    with np.load(PUB/'ink-hybrid-predictions.npz') as archive:
        pred = dict(archive)
    if not np.array_equal(pred['truth'],truth):
        raise ValueError("Incorrect held split")
    tiles = d['tiles']
    bank = meta['train_shapes']
    novel = truth>bank
    reachable = (truth[:,:,None]==prefix[:,None,:]).any(-1)|(truth<=bank)
    generated = pred['free']
    assert ((generated<=bank)|(generated[:,:,None]==prefix[:,None,:]).any(-1)).all()
    right = generated == truth
    teacher = pred['teacher'] == truth
    overlap = {}
    for n in (8,16):
        held_strings = {row[:n].tobytes() for row in truth}
        hits = set()
        for split,_,_,_,a,b in meta['records']:
            if split=='train' and b-a >= n:
                for row in np.lib.stride_tricks.sliding_window_view(d['stream'][a:b],n):
                    key = row.tobytes()
                    if key in held_strings:
                        hits.add(key)
        overlap[n] = sum(r[:n].tobytes() in hits for r in truth)
    # Scalar suffix counts independent of trie and CSR lookup for real training observations.
    train = [r for r in meta['records'] if r[0]=='train']
    hist = history_windows(d['stream'],train)
    pos = np.concatenate([np.arange(a,b) for *_,a,b in train])
    keys,truth_train = hist[pos],d['stream'][pos]
    with np.load(MODEL/'counts.npz') as archive:
        model = PrefixCounts(dict(archive))
    queries = np.row_stack([keys[::max(1,len(keys)//25)], np.zeros(6,np.uint16), np.full(6,bank+1,np.uint16)])
    for query in queries:
        match = np.ones(len(keys),bool)
        expected = model.prior
        for i,x in enumerate(query):
            match &= keys[:,i]==x
            if x==0 or match.sum()<2:
                break
            counts = np.bincount(truth_train[match],minlength=bank+1)
            expected = (counts+2*model.prior)/(counts.sum()+2)
        node,_ = model.lookup(query[None,:])
        dense = model.dense(node)[0]
        assert np.allclose(dense,expected,atol=1e-7) and np.isclose(dense.sum(),1)
        idx = dense.argmax()
        assert np.isclose(model.target_prob(node,np.array([idx]))[0],dense[idx])
    torch.set_num_threads(2)
    reps = json.loads((WORK/'representatives.json').read_text())
    firsts = {}
    for w in sorted(set(works)):
        i = int(np.flatnonzero(works==w)[0])
        example = {'prefix':prefix[i],'reference':truth[i],'generated':generated[i]}
        for name,seq in example.items():
            save_grid(tiles[seq].tobytes(),PUB/f'{w}-{name}.png')
        out = WORK/f'{w}-cpu.png'
        actual = write(MODEL,read_image(PUB/f'{w}-prefix.png',32),16,out,'cpu')
        assert np.array_equal(actual,tiles[generated[i]]) and np.array_equal(read_image(out,16),actual)
        firsts[w] = {k:''.join(reps[str(int(x))] for x in seq) for k,seq in example.items()}
    dump(PUB/'audit.json',dict(first=firsts,full_reference_training_matches=overlap,
         correct_by_step=right.sum(0).tolist(),teacher_by_step=teacher.sum(0).tolist(),
         correct_novel_free=int((right&novel).sum()),correct_novel_teacher=int((teacher&novel).sum()),
         allowed_error_count=int((~right&reachable).sum()),count_checks=len(queries),
         cpu_png_roundtrips=len(firsts)))
    print(firsts,overlap,flush=True)


if __name__=='__main__':
    main()

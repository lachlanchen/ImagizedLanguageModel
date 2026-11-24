"""Cross-check the release from PNGs and compare mistakes with the physical output support."""
import json
from collections import Counter
import numpy as np
import torch

from ilm.visual_lm.portable_writer import save_grid,read_image
from scripts.prepare_verse_canvas import corpus,PUB,MODEL,WORK
from scripts.evaluate_verse_canvas import cases
from scripts.write_verse_canvas import write
from scripts.prepare_dialogue_ink import dump,sha


def main():
    meta,data=corpus()
    score=json.loads((PUB/'results.json').read_text())
    assert sha(PUB/'outputs.npz')==score['archive']
    with np.load(PUB/'outputs.npz') as f:
        archive=dict(f)
    idx,rows,donors,gaps=cases(data)
    assert np.array_equal(idx,archive['idx']) and np.array_equal(donors,archive['donors'])
    assert np.array_equal(gaps,archive['gaps'])
    good=archive['ink_gap_original']
    exchanged=archive['ink_gap_exchanged']
    blank=archive['ink_blank_original']
    left=archive['ink_blank_left']
    legal=np.arange(7)<rows[:,1,None]
    truth=rows[:,9:16]
    correct=(good==truth).all(1)
    source_sensitive=correct&~(exchanged==truth).all(1)
    full=(blank==truth).all(1)
    reps=json.loads((WORK/'readback.json').read_text())
    examples={}
    picks=[('first',0),('miss',int(np.flatnonzero(~correct)[0]))]
    if source_sensitive.any():
        picks.append(('restore',int(np.flatnonzero(source_sensitive)[0])))
    # Freeze the first two non-overlapping illustrative failures/successes by sample order.
    if full.any():
        picks.append(('full',int(np.flatnonzero(full)[0])))
    torch.set_num_threads(2)
    for label,i in picks:
        n=rows[i,1]
        seqs=dict(source=rows[i,2:2+n],reference=truth[i,:n],blank=blank[i,:n],left=left[i,:n],
                  template=np.where(gaps[i,:n],0,truth[i,:n]),restored=good[i,:n],donor_result=exchanged[i,:n])
        images=WORK/'samples'/label
        images.mkdir(parents=True,exist_ok=True)
        for k,seq in seqs.items():
            save_grid(data['tiles'][0,seq].tobytes(),images/(k+'.png'))
        src=read_image(images/'source.png',None)
        for name,mask,order in [('blank',np.zeros_like(src),'confidence'),('left',np.zeros_like(src),'left'),
                                 ('restored',read_image(images/'template.png',None),'confidence')]:
            packed=write(MODEL,src,mask,order,'cpu')
            assert np.array_equal(packed,read_image(images/(name+'.png'),None))
        examples[label]=dict(index=int(i),poet=reps['authors'][rows[i,0]],
                              **{k:''.join(reps['shapes'][j] if j else '□' for j in v) for k,v in seqs.items()})
    # Training reference lines are excluded even when an author alias is used.
    train_lines={tuple(r[c:c+r[1]]) for r in data['train'] for c in (2,9)}
    train_authors=set(data['train'][:,0])
    assert not train_authors.intersection(data['test'][:,0])
    assert all(tuple(r[c:c+r[1]]) not in train_lines for r in data['test'] for c in (2,9))
    per_task={}
    for task,mask in [('blank',legal),('gap',gaps)]:
        names=['ink_'+task+'_'+order for order in ('original','left')]
        hits=[(archive[k]==truth)&mask for k in names]
        per_task[task]=dict(both_correct=int((hits[0]&hits[1]).sum()),
                            confidence_only=int((hits[0]&~hits[1]).sum()),
                            left_only=int((hits[1]&~hits[0]).sum()),
                            out_of_support=int(((truth>meta['train_shapes'])&mask).sum()))
    dump(PUB/'audit.json',dict(examples=examples,counts=per_task,
                               cpu_roundtrips=3*len(picks),source_sensitive_gap_lines=int(source_sensitive.sum())))
    print(examples,per_task,flush=True)


if __name__=='__main__':
    main()

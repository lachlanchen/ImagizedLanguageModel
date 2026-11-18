"""Count usable sentence assignment contrasts after controlling for visible glyph overlap."""
import json
from collections import Counter

import numpy as np

from scripts.prepare_translation_reading import OUT, WORK, cosine, load, require, sha, tiles, write_new


def run():
    with np.load(WORK/'sites.npz') as data:
        pairs=data['locations'][data['pairs'][data['calibration']]][:,:,0]
    require(len(pairs)==6720 and sha(WORK/'rows.jsonl')==load(WORK/'preparation.json')['rows_sha256'],
            'Calibration roster changed')
    indices=set(pairs.ravel())
    # Count identical bitmaps together even when two glyph records share the same drawing.
    bitmap=np.unique(tiles(),axis=0,return_inverse=True)[1]
    def features(ids):
        items=bitmap[ids].tolist()
        return Counter(items),Counter(zip(items,items[1:])),len(items)
    with (WORK/'rows.jsonl').open() as file:
        rows={}
        for i,line in enumerate(file):
            if i in indices:
                row=json.loads(line)
                rows[i]=features(row['x']),features(row['y'])
    margins=[]
    for i,j in pairs:
        x,y=rows[i]
        u,v=rows[j]
        score=[cosine(x[k],y[k])+cosine(u[k],v[k])-cosine(x[k],v[k])-cosine(u[k],y[k]) for k in (0,1)]
        margins.append([*score,abs(x[2]-v[2])+abs(u[2]-y[2])-abs(x[2]-y[2])-abs(u[2]-v[2])])
    values=np.array(margins)
    metrics={name:dict(correct=int((v>1e-8).sum()),wrong=int((v < -1e-8).sum()),ties=int((abs(v)<=1e-8).sum()))
             for name,v in zip(('unigram','bigram','length'),values.T)}
    hard=np.flatnonzero((values[:,:2]<=1e-8).all(1))
    write_new(OUT/'calibration_overlap.json',dict(pairs=len(pairs),metrics=metrics,hard_pairs=len(hard),
              distinct_hard_sentence_pairs=len({tuple(sorted(pairs[n].tolist())) for n in hard}),
              distinct_sentence_pairs=len({tuple(sorted(p.tolist())) for p in pairs}),hard_ids=hard.tolist(),
              initial_sample=load(OUT/'preparation.json')['metrics'],matched_roster_sha256=sha(WORK/'sites.npz')))
    print(metrics,'both overlap controls fail',len(hard))


if __name__=='__main__':
    run()

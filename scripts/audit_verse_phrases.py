"""Compare commonness and reduplication of the actual restored two-glyph phrases."""
from collections import Counter
import json
import numpy as np
from scripts.prepare_verse_canvas import corpus,PUB
from scripts.prepare_dialogue_ink import sha,dump


def main():
    meta,data=corpus()
    record=json.loads((PUB/'confirmation.json').read_text())
    path=PUB/'confirmation.npz'
    if sha(path)!=record['archive']:
        raise ValueError('Changed confirmation output')
    with np.load(path) as archive:
        rows=data['dev'][archive['ix']]
        truth=rows[:,9:]
        gaps=archive['gaps']
        positions=gaps.argmax(1)
        r=np.arange(len(rows))
        if not np.all(gaps[r,positions]&gaps[r,positions+1]) or not np.all(gaps.sum(1)==2):
            raise ValueError('Expected contiguous two-cell gaps')
        freq=Counter((int(a),int(b)) for row in data['train']
                     for a,b in zip(row[9:9+row[1]-1],row[10:9+row[1]]))
        summary={}
        for name in ('reference','ink','ink-bound','byte-bound','aligned'):
            out=truth if name=='reference' else archive[name+'_gap_original']
            pair=list(zip(out[r,positions],out[r,positions+1]))
            summary[name]=dict(in_fitting=sum(p in freq for p in pair),
                               reduplicated=int(sum(a==b for a,b in pair)),
                               unique=len(set(pair)))
        dump(PUB/'phrases.json',dict(scores=summary,rows=len(rows),outputs=record['archive'],
              out_of_palette=int(((truth>meta['train_shapes'])&gaps).sum()),
              repeated_reference_lines=int(sum(len(set(row[9:9+row[1]]))<row[1] for row in rows))))
    print(summary,flush=True)


if __name__=='__main__':
    main()

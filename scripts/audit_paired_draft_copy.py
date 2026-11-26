"""Determine which free outputs were already present in their complete source-derived draft."""
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile
import json

import numpy as np
from scripts.measure_paired_visual_writer import ARCHIVE, PUB, base, unpack, CoveredWriter, png


def main():
    grade=base.read(PUB/'free-meaning.json')
    natural=set(base.read(PUB/'provenance.json')['natural']['audit'])
    writer=base.read(PUB/'writer.json')
    assert base.sha(ARCHIVE)==writer['archive']
    own=np.asarray(grade['votes']).all(axis=3).diagonal(axis1=2,axis2=3)
    indices={i:n for n,i in enumerate(grade['ids'])}
    engines=[CoveredWriter(t) for t in (0,1)]
    shape=base.Ink(np.load(base.BASE/'palette.npy'))
    totals={'corpus':Counter(),'constructed':Counter()}
    items=[]
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read('records.json'))
        for r in records:
            i,t,s=r['i'],r['t'],r['side']
            prefix=f'{i}/{t}/{s}/'
            src=unpack(z.read(prefix+'source.png'),len(shape.encode(r['source'])))
            physical=engines[t].translate(src)[0]
            output=unpack(z.read(prefix+'answer.png'),r['cells'])
            ref=unpack(z.read(prefix+'reference.png'),len(shape.encode(r['ref'])))
            copied=np.array_equal(physical,output)
            ref_before=np.array_equal(physical,ref)
            ref_after=np.array_equal(output,ref)
            assert ref_after==r['exact'] and r['stop']
            correct=bool(own[indices[i],t,s])
            group='corpus' if i in natural else 'constructed'
            totals[group].update(dict(n=1,copied=int(copied),credited_copy=int(correct and copied),credited_change=int(correct and not copied),
                reference_before=int(ref_before),reference_after=int(ref_after),exact_gains=int(ref_after and not ref_before),
                exact_losses=int(ref_before and not ref_after)))
            items.append(dict(i=i,t=t,s=s,copied=copied,ref_before=ref_before,ref_after=ref_after,
                              draft_bitmap=sha256(png(physical)).hexdigest()))
    assert not shape.bad
    base.dump(PUB/'draft-copy.json',dict(archive=base.sha(ARCHIVE),free=base.sha(PUB/'free-meaning.json'),rows=items,totals=totals))
    print(totals,flush=True)


if __name__=='__main__':
    main()

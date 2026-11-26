"""Measure the exact shared-prefix coefficient in a length-normalized sequence comparison."""
from itertools import takewhile
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
from scripts.prepare_visual_sequence_preference import ROOT,WORK,PUB,base,previous
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_visual_sequence_preference import ARCHIVE


def coefficient(positive,negative):
    a,b=len(positive),len(negative)
    assert a and b
    k=sum(1 for _ in takewhile(lambda pair:pair[0]==pair[1],zip(positive,negative)))
    w=(b-a)/(a*b)
    common=k*abs(w)
    remaining=(a-k)/a+(b-k)/b
    return dict(gold=a,free=b,shared=k,coefficient=w,
                shared_absolute_fraction=common/(common+remaining) if common+remaining else 0.)


def summary(rows):
    return dict(pairs=len(rows),shared=sum(r['shared']>0 for r in rows),
                shared_prefix_penalized=sum(r['shared']>0 and r['coefficient']<0 for r in rows),
                shared_prefix_rewarded=sum(r['shared']>0 and r['coefficient']>0 for r in rows),
                shared_prefix_cancelled=sum(r['shared']>0 and r['coefficient']==0 for r in rows),
                mean_absolute_fraction=float(np.mean([r['shared_absolute_fraction'] for r in rows])))


def main():
    selection=base.read(PUB/'confirmed.json')
    mined=base.read(WORK/'emissions.json')
    pairs=np.load(previous.WORK/'pairs.npy',mmap_mode='r')
    training=[]
    for i in selection['admitted']:
        r=mined[i]
        x=pairs[r['i'],r['t']]
        training.append(dict(i=i,**coefficient(x[x!=0].tolist()+[1],r['cells']+[1])))
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    lut={p.tobytes():i for i,p in enumerate(face.array()) if i>=2}
    held={}
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        for m in ('parent','likelihood','preference'):
            data=[]
            for i,r in enumerate(rows):
                ref=face.encode(r['ref'])+[1]
                answer=r['answers'][m]
                cells=unpack(z.read(f'{i:04d}/{m}.png'),answer['cells'])
                ids=[lut[p.tobytes()] for p in cells]+([1] if answer['stop'] else [])
                data.append(coefficient(ref,ids))
            held[m]=dict(summary=summary(data),rows=data)
    assert not face.bad
    base.dump(PUB/'prefix-weights.json',dict(training=dict(summary=summary(training),rows=training),held=held,
            proof=base.sha(ROOT/'formal/span_list/SequenceMarginGreedy.lean'),source=base.sha(Path(__file__))))
    print('Shared-prefix ranking weights',summary(training),flush=True)


if __name__=='__main__':
    main()

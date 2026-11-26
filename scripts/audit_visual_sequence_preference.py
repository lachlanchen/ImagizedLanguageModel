"""Audit training labels, the raw panel, prefix scores, exchanged sources and CPU reproductions."""
from collections import Counter
from hashlib import sha256
import json
from zipfile import ZipFile
import numpy as np
import torch
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_visual_sequence_preference import ARCHIVE,ARMS,PUB,WORK,base,png,panel_rows
from scripts.train_visual_sequence_preference import SEED,MODEL
from scripts.write_visual_sequence_preference import SequenceWriter


def label_report():
    rows=base.read(WORK/'emissions.json')
    verdict=base.read(PUB/'verification.json')
    voters=verdict['votes']
    assert verdict['emissions']==base.sha(WORK/'emissions.json')
    result={}
    identical=[i for i,r in enumerate(rows) if r['ref']==r['free']]
    for j,v in enumerate(voters):
        counts=Counter(tuple(v[str(i)]) for i in range(len(rows)))
        equal_disagree=sum(v[str(i)][0]!=v[str(i)][1] for i in identical)
        result[str(j)]=dict(counts={'ref_'+str(int(a))+'_free_'+str(int(b)):n for (a,b),n in counts.items()},
                            identical=len(identical),inconsistent_identical=equal_disagree)
    admitted=verdict['admitted']
    assert all(all(v[str(i)]==[True,False] for v in voters) and rows[i]['stop'] and i not in identical for i in admitted)
    final=base.read(PUB/'confirmed.json')
    confirmed=[i for i in admitted if final['votes'][str(i)]==[True,False]]
    assert confirmed==final['admitted']
    counts=[sum(rows[i]['t']==t for i in confirmed) for t in (0,1)]
    return dict(raters=result,grouped=len(admitted),confirmed=counts,total=len(rows),identical=len(identical))


def main():
    torch.set_num_threads(3)
    report=base.read(PUB/'assessment.json')
    assert report['archive']==base.sha(ARCHIVE)
    official,_=panel_rows()
    paths=base.read(PUB/'paths.json')
    replay=[]
    roles={}
    prefix={}
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        assert len(rows)==len(official)==568
        for i,(r,truth) in enumerate(zip(rows,official,strict=True)):
            assert all(r[k]==truth[k] for k in ('source','ref','t','group','donor'))
            for m in ARMS:
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==r['answers'][m]['png']
        face=base.Ink(np.load(base.BASE/'palette.npy'))
        for m in ('likelihood','preference'):
            writer=SequenceWriter(m,'cpu')
            for i in (0,1,24,25,48,49,72,73,96,97,332,333):
                r=rows[i]
                x=unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source'])))
                pixels,stop=writer.translate([x],r['t'])[0]
                actual=sha256(png(pixels)).hexdigest()
                replay.append(dict(i=i,arm=m,actual=actual,same=actual==r['answers'][m]['png'] and stop==r['answers'][m]['stop']))
        for m in ('parent','likelihood','preference'):
            exact=[]
            agreement=[]
            for i,r in enumerate(rows):
                ids=face.encode(r['ref'])
                ref=writer.palette[ids]
                a=r['answers'][m]
                got=unpack(z.read(f'{i:04d}/{m}.png'),a['cells'])
                eq=a['stop'] and np.array_equal(got,ref)
                assert eq==a['exact']
                if i>=96:
                    exact.append(eq)
                else:
                    physical=next((k for k,(p,q) in enumerate(zip(got,ref)) if not np.array_equal(p,q)),min(len(ref),len(got)))
                    if eq:physical+=1
                    predicted=paths[i][m]['ref']['first_error']
                    agreement.append(dict(i=i,free_prefix=physical,supplied_prefix=predicted))
            roles[m]=dict(exact=sum(exact),polarity=sum(all(exact[k:k+2]) for k in range(0,472,2)),
                          crossed=sum(all(exact[k:k+4]) for k in range(0,472,4)))
            assert roles[m]==report['roles'][m]
            prefix[m]=dict(agreements=sum(v['free_prefix']==v['supplied_prefix'] for v in agreement),
                           differences=[v for v in agreement if v['free_prefix']!=v['supplied_prefix']])
        for i,r in enumerate(rows):
            assert rows[r['donor']]['source']!=r['source']
        counterfactual=sum(r['answers']['exchanged']['png']==rows[r['donor']]['answers']['preference']['png'] for r in rows)
        teacher=report['teacher']
        assert len(report['votes'])==192 and len(teacher)==96
        counts=Counter()
        for i,r in enumerate(rows[:96]):
            table=dict(r['answers'],qwen=teacher[i],reference={'stop':True},foil={'stop':True})
            votes=report['votes'][2*i:2*i+2]
            grade={m:bool(a['stop'] and all(v[m] for v in votes)) for m,a in table.items()}
            assert grade==report['grades'][i]
            counts.update(grade)
        assert all(counts[m]==report['counts']['all'][m] for m in counts)
    assert not face.bad
    base.dump(PUB/'audit.json',dict(archive=report['archive'],labels=label_report(),roles=roles,prefix=prefix,reads=replay,
              same_cpu=sum(r['same'] for r in replay),same_donor=counterfactual,work_split=base.sha(WORK/'panel.json'),
              proof=base.sha(WORK.parents[1]/'formal/span_list/SequenceMarginGreedy.lean')))
    print('CPU images',sum(r['same'] for r in replay),len(replay),'swapped-source agreement',counterfactual,len(rows),flush=True)


if __name__=='__main__':
    main()

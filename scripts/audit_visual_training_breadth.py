"""Audit the physical outputs and measure separate source and phrase-draft effects on answer scores."""
import json
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile
import numpy as np
import torch
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_visual_training_breadth import ARCHIVE, ARMS, PUB, ROOT, WORK, MODEL, cases, base, png
from scripts.train_visual_training_breadth import restore, COHORT, PASSES
from scripts.write_glyph_draft_editor import pack,pictured_input
from scripts.write_covered_phrases import CoveredWriter


@torch.inference_mode()
def score(net, source, draft, answer, direction, stop):
    ids=pack([source],'cpu')
    target=pack([answer+([1] if stop else [])],'cpu')
    ds=pack([draft or [0]],'cpu')
    field=(255*net.marker(net.features,ids).sigmoid()*ids.ne(0).unsqueeze(-1)).round().byte()
    law=net.model(net.features,pictured_input(ids,field),target,net.inv[direction],direction,ds,net.local[direction])[0]
    wanted=net.local[direction][target[0]]
    return law.gather(1,wanted[:,None]).flatten().double().numpy(),law.argmax(-1).eq(wanted).numpy()


def main():
    torch.set_num_threads(3)
    assert base.read(PUB/'generation.json')['archive']==base.sha(ARCHIVE)
    official,_=cases()
    selection=base.read(PUB/'selection.json')
    active=np.load(WORK/'indices.npy')
    pairs=np.load(WORK/'pairs.npy',mmap_mode='r')
    lengths=np.count_nonzero(pairs[:PASSES*COHORT],axis=-1).reshape(PASSES,COHORT,2)
    assert len(np.unique(active))==COHORT*PASSES and np.all(lengths==lengths[0])
    epoch_positions=int(lengths[0].sum()+np.count_nonzero(pairs[PASSES*COHORT:])+2*(COHORT+selection['roles']))
    for m in ('narrow','broad'):
        assert all(ep['positions']==epoch_positions for ep in base.read(MODEL/m/'fit.json')['epochs'])
    nets={m:restore(m,'cpu') for m in ARMS[:3]}
    face=base.Ink(nets['parent'].palette)
    replay=[]
    results={m:[] for m in nets}
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        images=[]
        for i,(r,truth) in enumerate(zip(rows,official,strict=True)):
            assert all(r[k]==truth[k] for k in ('source','ref','t','group','donor'))
            images.append(unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source']))))
            for m in ARMS:
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==r['answers'][m]['png']
        for i in (0,1,24,25,48,49,72,73,96,97,332,333):
            for m,net in nets.items():
                pixels,stop=net.translate([images[i]],rows[i]['t'])[0]
                expected=rows[i]['answers'][m]
                replay.append(dict(index=i,model=m,same=(expected['png']==sha256(png(pixels)).hexdigest() and stop==expected['stop'])))
        donor=sum(r['answers']['exchanged']['png']==rows[r['donor']]['answers']['broad']['png'] for r in rows)
        engines=[CoveredWriter(t) for t in (0,1)]
        src=[nets['parent'].addresses(x) for x in images[:96]]
        drafts=[nets['parent'].addresses(engines[r['t']].translate(images[i])[0]) for i,r in enumerate(rows[:96])]
        for m,net in nets.items():
            for i,r in enumerate(rows[:96]):
                j=r['donor']
                record=r['answers'][m]
                target=face.encode(r['ref'])
                free=net.addresses(unpack(z.read(f'{i:04d}/{m}.png'),record['cells']))
                margins={}
                observed={}
                for name,si,di in (('empty',j,j),('source',i,j),('draft',j,i),('intact',i,i)):
                    ref,top=score(net,src[si],drafts[di],target,r['t'],True)
                    own,_=score(net,src[si],drafts[di],free,r['t'],record['stop'])
                    observed[name]=(ref,top)
                    margins[name]=float(ref.mean()-own.mean())
                a,b,c,d=[margins[k] for k in ('empty','source','draft','intact')]
                dx=((b-a)+(d-c))/2
                dy=((c-a)+(d-b))/2
                assert abs(a+dx+dy-d)<1e-10
                ref,correct=observed['intact']
                row=dict(i=i,margins=margins,source_attribution=dx,draft_attribution=dy,cells=len(ref),correct=int(correct.sum()))
                for name,donorkey in (('source','draft'),('draft','source')):
                    other,hits=observed[donorkey]
                    row[name]=dict(log_gain=float((ref-other).mean()),repairs=int((correct & ~hits).sum()),losses=int((~correct & hits).sum()))
                results[m].append(row)
    assert not face.bad
    totals={}
    for m,values in results.items():
        totals[m]=dict(reference_preferred=sum(r['margins']['intact']>1e-6 for r in values),
                       mean_intact_gap=float(np.mean([r['margins']['intact'] for r in values])),
                       cells=sum(r['cells'] for r in values),top1=sum(r['correct'] for r in values))
        for k in ('source','draft'):
            totals[m][k]=dict(positive_requests=sum(r[k]['log_gain']>1e-6 for r in values),
                              mean_reference_log_gain=float(np.mean([r[k]['log_gain'] for r in values])),
                              mean_attribution=float(np.mean([r[k+'_attribution'] for r in values])),
                              repairs=sum(r[k]['repairs'] for r in values),losses=sum(r[k]['losses'] for r in values))
    base.dump(PUB/'audit.json',dict(archive=base.sha(ARCHIVE),epoch_targets=epoch_positions,donor=donor,
              replay=replay,cpu_exact=sum(r['same'] for r in replay),totals=totals,paths=results,
              proof=base.sha(ROOT/'formal/span_list/PicturedEvidenceAttribution.lean'),
              protocol=base.sha(ROOT/'references/visual_training_breadth_source_test.md')))
    print('cpu',sum(r['same'] for r in replay),'donors',donor,totals,flush=True)


if __name__=='__main__':
    main()

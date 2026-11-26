"""Test a fixed visual writer on attested bilingual source edits and archive all audit replies."""
from hashlib import sha256
import json
from time import monotonic
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.qualify_paired_translation_teacher import DATA, PUB, ROOT, base, margins, population
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_visual_training_breadth import png
from scripts.train_visual_training_breadth import restore, MODEL
from scripts.train_visual_sequence_preference import path_scores
from scripts.write_glyph_draft_editor import pack, pictured_input
from scripts.write_covered_phrases import CoveredWriter

ARCHIVE = ROOT / 'publication/ilm-nature/evidence/paired_visual_answers.zip'


@torch.inference_mode()
def main():
    if (PUB/'writer.json').exists():
        raise SystemExit('Preserved writer result exists')
    selection, families = population()
    teacher = base.read(PUB/'teacher.json')
    assert teacher['selection'] == base.sha(PUB/'selection.json')
    torch.set_num_threads(3)
    start = monotonic()
    net = restore('broad','cpu')
    inputs = np.load(DATA/'pairs.npy',mmap_mode='r')
    ids = np.asarray([r['rows'] for r in selection['contrasts']])
    audit = set(teacher['audit'])
    names = base.read(base.BASE/'readback.json')
    scores = np.empty((len(ids),2,2,2),dtype=np.float64)
    counts = np.empty_like(scores, dtype=np.int32)
    records = []
    source_roundtrips = 0
    temp = ARCHIVE.with_suffix('.partial')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
        for t in (0,1):
            engine = CoveredWriter(t)
            for step in range(0,len(ids),16):
                group = np.arange(step,min(step+16,len(ids)))
                sources, targets = [], []
                for i in group:
                    for j in ids[i]:
                        source,target = inputs[j,1-t],inputs[j,t]
                        sources.append(net.palette[source[source!=0]])
                        targets.append(target[target!=0].tolist()+[1])
                drafts = [engine.translate(img)[0] for img in sources]
                optical = [net.addresses(img) for img in sources]
                for n, cells in enumerate(optical):
                    i, side = group[n//2], n%2
                    assert np.array_equal(net.palette[cells], sources[n])
                    assert ''.join(names[str(k)] for k in cells)==families[i][side][1-t]
                    assert ''.join(names[str(k)] for k in targets[n][:-1])==families[i][side][t]
                    source_roundtrips += 1
                source = pack(optical,'cpu')
                field = (255*net.marker(net.features,source).sigmoid()*source.ne(0).unsqueeze(-1)).round().byte()
                source = pictured_input(source,field)
                draft = pack([net.addresses(d) or [0] for d in drafts],'cpu')
                features = net.features,net.inv,net.local
                for cross in (0,1):
                    target = pack([targets[n^cross] for n in range(len(targets))],'cpu')
                    ll,width = path_scores(net.model,features,source,draft,target,t)
                    for n,(a,b) in enumerate(zip(ll.tolist(),width.tolist(),strict=True)):
                        i,side = group[n//2],n%2
                        scores[i,t,side,side^cross] = a/b
                        counts[i,t,side,side^cross] = b
                held = [n for n in range(len(sources)) if group[n//2] in audit]
                if held:
                    predictions = net.translate([sources[n] for n in held],t,[drafts[n] for n in held])
                    for n,(img,stop) in zip(held,predictions,strict=True):
                        i,side=int(group[n//2]),n%2
                        cells=net.addresses(img)
                        rec=dict(i=i,t=t,side=side,stop=bool(stop),cells=len(img),
                                 text=''.join(names[str(k)] for k in cells),exact=bool(stop and cells+[1]==targets[n]),
                                 source=families[i][side][1-t],ref=families[i][side][t])
                        for key,bitmap in [('source',sources[n]),('answer',img),('reference',net.palette[targets[n][:-1]])]:
                            blob=png(bitmap)
                            rec[key+'_sha256']=sha256(blob).hexdigest()
                            z.writestr(f'{i}/{t}/{side}/{key}.png',blob)
                        records.append(rec)
        z.writestr('records.json',json.dumps(records,ensure_ascii=False))
    assert len(records)==len(audit)*4
    assert np.isfinite(scores).all() and (counts>0).all()
    temp.replace(ARCHIVE)
    with ZipFile(ARCHIVE) as z:
        for r in records:
            prefix=f"{r['i']}/{r['t']}/{r['side']}/"
            for k in ('source','answer','reference'):
                assert sha256(z.read(prefix+k+'.png')).hexdigest()==r[k+'_sha256']
        # Verify a single isolated CPU decode per direction and source side on eight frames.
        replay=[]
        for r in records:
            if r['i'] not in teacher['audit'][:8]:continue
            prefix=f"{r['i']}/{r['t']}/{r['side']}/"
            length=np.count_nonzero(inputs[ids[r['i'],r['side']],1-r['t']])
            physical=unpack(z.read(prefix+'source.png'),length)
            out,stopped=net.translate([physical],r['t'])[0]
            replay.append(bool(stopped==r['stop'] and sha256(png(out)).hexdigest()==r['answer_sha256']))
    path=PUB/'writer-scores.npz'
    np.savez_compressed(path,scores=scores,lengths=counts)
    a=np.asarray(teacher['audit'])
    aligned=margins(scores)>0
    base.dump(PUB/'writer.json',dict(teacher=base.sha(PUB/'teacher.json'),archive=base.sha(ARCHIVE),
              arrays=base.sha(path),fit=base.sha(MODEL/'broad/fit.json'),roundtrips=source_roundtrips,
              audit_requests=len(records),exact=sum(r['exact'] for r in records),stopped=sum(r['stop'] for r in records),
              paired_both_directions=int(aligned[a].all(axis=(1,2)).sum()),positive_individual=int(aligned[a].sum()),
              replay=replay,seconds=monotonic()-start))
    print('audit: exact written',sum(r['exact'] for r in records),'of',len(records),'ranked',int(aligned[a].sum()),
          'both directions',int(aligned[a].all(axis=(1,2)).sum()),'replay',sum(replay),'/',len(replay),flush=True)


if __name__=='__main__':
    main()

"""Check an alias-only replay and judge the changed physical feedback on the frozen panel."""
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from ilm.visual_lm.phase_feedback import PhaseFeedbackWriter
from scripts.compare_training_breadth_transfer import ROOT,PUB,RAW,ARCHIVE,base,png,grids,blind_votes,gpu_owner,configure_cuda_cli

OUTPUT=ROOT/'publication/ilm-nature/evidence/phase_feedback_images.zip'


def generate():
    if (PUB/'phase-generation.json').exists():
        assert base.sha(OUTPUT)==base.read(PUB/'phase-generation.json')['archive']
        return
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        inputs=grids(z,rows)
    with ZipFile(RAW) as z:
        original=json.loads(z.read('records.json'))
    alphabet=np.load(base.BASE/'palette.npy')
    names=base.read(base.BASE/'readback.json')
    reverse={v.tobytes():names[str(i)] for i,v in enumerate(alphabet) if str(i) in names}
    records=[{} for _ in inputs]
    cost={}
    partial=OUTPUT.with_suffix('.partial')
    with ZipFile(partial,'w',ZIP_DEFLATED) as dest:
        for t,lang in enumerate(('en','zh')):
            order=[i for i,r in enumerate(rows) if r['t']==t]
            for control in (True,False):
                net=PhaseFeedbackWriter(lang,'cuda',pooled_control=control)
                cost[lang]=net.phase_cost
                for k in range(0,len(order),16):
                    ids=order[k:k+16]
                    for i,(pixels,stop,spans) in zip(ids,net.write([inputs[i] for i in ids]),strict=True):
                        packed=np.packbits(pixels.reshape(len(pixels),1024),axis=1)
                        blob=png(packed)
                        h=sha256(blob).hexdigest()
                        old=original[i]['visual']
                        if control:
                            records[i]['alias_same']=(h==old['png'] and stop==old['stop'] and spans==old['spans'])
                        else:
                            records[i]['phase']=dict(text=''.join(reverse[cell.tobytes()] for cell in packed),
                                                     cells=len(packed),stop=bool(stop),spans=spans,png=h)
                            dest.writestr(f'{i:04d}/phase.png',blob)
                del net
        dest.writestr('records.json',json.dumps(records,ensure_ascii=False))
    partial.replace(OUTPUT)
    base.dump(PUB/'phase-generation.json',dict(archive=base.sha(OUTPUT),original=base.sha(RAW),cost=cost,
              aliases_same=sum(r['alias_same'] for r in records),protocol=base.sha(ROOT/'references/phase_feedback_protocol.md')))


def assess():
    with ZipFile(OUTPUT) as z:
        variants=json.loads(z.read('records.json'))
        for i,r in enumerate(variants):
            assert sha256(z.read(f'{i:04d}/phase.png')).hexdigest()==r['phase']['png']
    with ZipFile(RAW) as z:parents=json.loads(z.read('records.json'))
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        inputs=grids(z,rows)
    teacher=base.read(PUB/'assessment.json')['comparator']
    answers=[dict(v,phase=r['phase'],broad=b['answers']['broad'],qwen=q,reference=dict(text=b['ref'],stop=True),
                  foil=dict(text=b['foil'],stop=True)) for r,v,b,q in zip(variants,parents,rows,teacher,strict=True)]
    votes=blind_votes(rows,answers)
    grades=[];counts=defaultdict(Counter)
    for i,r in enumerate(rows):
        grade={k:bool(a['stop'] and votes[i*2][k] and votes[i*2+1][k]) for k,a in answers[i].items()}
        grades.append(grade)
        for group in ('all',r['doc'],('en','zh')[r['t']]):counts[group].update(grade)
    diffs=[(counts[w]['phase']-counts[w]['visual'])/24 for w in dict.fromkeys(r['doc'] for r in rows)]
    ci=np.quantile([np.mean(p) for p in product(diffs,repeat=4)],[.025,.975]).tolist()
    cpu=[]
    for t,lang in enumerate(('en','zh')):
        group=[i for i in (0,1,24,25,48,49,72,73) if rows[i]['t']==t]
        net=PhaseFeedbackWriter(lang,'cpu')
        for i,(pixels,stop,ends) in zip(group,net.write([inputs[j] for j in group]),strict=True):
            raw=png(np.packbits(pixels.reshape(len(pixels),1024),axis=1))
            old=variants[i]['phase']
            cpu.append(dict(index=i,same=sha256(raw).hexdigest()==old['png'] and ends==old['spans'] and stop==old['stop']))
    base.dump(PUB/'phase.json',dict(archive=base.sha(OUTPUT),grades=grades,votes=votes,counts=counts,interval=ci,cpu=cpu,
              stopped=sum(v['phase']['stop'] for v in variants),
              changed=sum(v['phase']['png']!=p['visual']['png'] for v,p in zip(variants,parents,strict=True))))
    print('phase',dict(counts['all']),ci,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        generate()
        if not (PUB/'phase.json').exists():
            assess()

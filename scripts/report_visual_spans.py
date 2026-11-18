"""Recount samples and export original pixels and measured values for the alignment study."""
import io
import shutil
from collections import Counter
from zipfile import ZipFile

import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.prepare_edit_diversity import COHORTS
from scripts.train_visual_extractive_reader import save_page
from scripts.train_visual_spans import ARMS, ROOT, WORK, frozen, load, require, sha, tiles, write_new


def audit_samples(split,bank,rows):
    folder=WORK/'risk'/split
    report=load(folder/'report.json')
    require(sha(folder/'streams.zip')==report['streams_sha256'],'Sample archive changed')
    known={bytes(v):i for i,v in enumerate(bank)}
    total={}
    with ZipFile(folder/'streams.zip') as z:
        for task,name in enumerate(('translation','quote')):
            tally=Counter()
            for i,r in enumerate(rows):
                if not r['x']:
                    continue
                target=r['x'] if task else r['y']
                raw=bank[target].tobytes()
                with np.load(io.BytesIO(z.read(f'{name}/{i}.samples'))) as data:
                    ns,stops,pixels,anchors=[data[k] for k in ('lengths','stops','pixels','cursors')]
                require(len(ns)==len(stops)==32 and ns.sum()==len(pixels)==len(anchors),'Ragged samples inconsistent')
                offset=0
                samples_hit=[]
                for n,end in zip(ns,stops,strict=True):
                    chunk=pixels[offset:offset+n]
                    pos=anchors[offset:offset+n]
                    require(bool(np.all(np.diff(pos)>=0)) and bool(np.all(pos<=len(r['x']))),'Bad anchor order')
                    samples_hit.append(bool(end and chunk.tobytes()==raw))
                    offset+=n
                selected=z.read(f'{name}/{i}.bin')
                decoded=[known.get(bytes(v),-1) for v in np.frombuffer(selected,np.uint8).reshape(-1,128)]
                tally.update(exact=raw==selected,stopped_samples=int(stops.sum()),sample_cells=len(pixels),
                             edits=Levenshtein.distance(decoded,target),known_sample_cells=sum(bytes(v) in known for v in pixels),
                             proposal_exact=any(samples_hit[:16]),risk_draw_exact=any(samples_hit[16:]),
                             changed_proposal_exact=any(samples_hit[:16]) and r['x']!=target,
                             changed_any_sample_exact=any(samples_hit) and r['x']!=target)
            counts=report['counts'][name]
            for key,reported in [('exact','exact'),('edits','edits'),('sample_cells','sample_cells'),('stopped_samples','sample_stops')]:
                require(tally[key]==counts[reported],f'{name} failed independent {key} recount')
            total[name]=dict(tally)
    return total


def main():
    dest=ROOT/'references/evidence/visual_span_20261118'
    dest.mkdir()
    bank=tiles()
    sets={s:list(frozen(s)) for s in COHORTS}
    for s,rows in sets.items():
        write_new(dest/(s+'_sampling_audit.json'),audit_samples(s,bank,rows))
    for p in sorted(WORK.rglob('*.json')):
        if p.name!='rows.json':
            shutil.copy2(p,dest/'_'.join(p.relative_to(WORK).parts))
    table=[['Writer and selection rule','Discovery','Replication']]
    for label,arm in [('Local edits','local'),('Original editor','parent'),('Fixed alignment, one path','fixed'),
                      ('Summed alignments, one path','marginal'),('Summed alignments, next image','marginal_prefix'),
                      ('Summed alignments, edit risk','risk')]:
        row=[label]
        for s in COHORTS:
            p=load(WORK/('parent' if arm=='local' else arm)/s/'report.json')
            v=p.get('outcome',p.get('counts'))['local' if arm=='local' else 'translation']
            row.append(f"{v['exact']}   /   {100*v['cer']:.2f}%")
        table.append(row)
    accuracy=[]
    for key in ('assigned','reversed_assigned'):
        accuracy.append([round(100*d[key]/d['pairs'],2) for s in COHORTS
                         for d in [load(WORK/(s+'_exchange.json'))['marginal']]])
    rows=sets['discovery']
    scored=load(WORK/'marginal/discovery/rows.json')
    chosen={'changed':next(r['index'] for r in scored if r['translation']['exact'] and r['hard']),
            'capped':next(r['index'] for r in scored if not r['translation']['stopped']
                          and max(len(rows[r['index']]['x']),r['length'])<=14)}
    widths,examples={},{}
    for label,i in chosen.items():
        r=rows[i]
        imgs=dict(source=bank[r['x']],reference=bank[r['y']])
        examples[label]=dict(index=i,source=r['source'],reference=r['target'],chapter=r['g'],lengths={})
        for a in ('marginal','marginal_prefix','risk'):
            with ZipFile(WORK/a/'discovery/streams.zip') as z:
                imgs[a]=np.frombuffer(z.read(f'translation/{i}.bin'),np.uint8).reshape(-1,128)
        for key,img in imgs.items():
            examples[label]['lengths'][key]=len(img)
            display=img[:16]
            require(len(display)>0,'Selected display is empty')
            name=f'{label}_{key}'
            widths[name]=len(display)
            save_page(dest/(name+'.png'),display)
    write_new(dest/'figure.json',dict(table=table,span_accuracy_percent=accuracy,widths=widths,examples=examples))


if __name__=='__main__':
    main()

"""Blind and then score contextual readings of saved Chinese insertion images."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scripts.report_local_ink import RUN, require, roster, sha, sites_independent, write_new
from scripts.report_visual_phrases import png

FOLDER = RUN/'reading'
ARMS = ('source','local','reference')


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare():
    rows,bank = roster('fresh')
    forms = {bytes(bank[g]):ch for r in rows for field,ids in [('source','x'),('target','y')]
             for ch,g in zip(r[field],r[ids])}
    loc = list(sites_independent(rows))
    report = json.loads((RUN/'stream/fresh/report.json').read_text())
    runs = {}
    for arm in ARMS[:2]:
        path = RUN/'stream/fresh'/(arm+'.npz')
        require(sha(path)==report['arms'][arm]['pixels_sha256'],'Changed streams')
        with np.load(path) as z:
            runs[arm]=np.split(z['pixels'],z['offsets'][1:-1])
    FOLDER.mkdir()
    packet,mapping=[],[]
    seen=set()
    picked=Counter()
    for n in sorted(range(len(loc)),key=lambda n:key(f'wording-fresh-1/{n}')):
        i,c,j,k=loc[n]
        r=rows[i]
        stratum='single' if k-j==1 else 'multi'
        if n in (238,714) or r['g'] in seen or picked[stratum]>=24:
            continue
        picked[stratum]+=1
        seen.add(r['g'])
        phrase=[runs['source'][n],runs['local'][n],bank[r['y'][j:k]]]
        versions=sorted(set(p.tobytes() for p in phrase),key=lambda s: hashlib.sha256(n.to_bytes(4,'big')+s).digest())
        labels={}
        for v,data in enumerate(versions):
            raw=np.frombuffer(data,np.uint8).reshape(-1,128)
            string=''.join(forms.get(bytes(pic),'▧') for pic in raw)
            label=chr(65+v)
            if '▧' in string:
                png(FOLDER,f'{len(packet)+1}_{label}',raw)
            labels[label]=string
        packet.append(dict(number=len(packet)+1,classical=r['source'],before=r['target'][:j],after=r['target'][k:],versions=labels))
        mapping.append(dict(site=n,stratum=stratum,labels={arm:chr(65+versions.index(p.tobytes())) for arm,p in zip(ARMS,phrase)}))
        if len(packet)==48:
            break
    require(len(packet)==48,'Incomplete reading packet')
    write_new(FOLDER/'packet.json',packet)
    write_new(FOLDER/'mapping.json',mapping)
    write_new(FOLDER/'binding.json',dict(packet=sha(FOLDER/'packet.json'),mapping=sha(FOLDER/'mapping.json')))
    print('Frozen packet',sha(FOLDER/'packet.json'))


def score(grades):
    read=lambda n:json.loads((FOLDER/(n+'.json')).read_text())
    bind=read('binding')
    for name in ('packet','mapping'):
        require(bind[name]==sha(FOLDER/(name+'.json')), 'Changed reading packet')
    packet,mapping=read('packet'),read('mapping')
    results={a:Counter() for a in ARMS}
    per_item=[]
    for p,m,decision in zip(packet,mapping,grades.read_text().splitlines(),strict=True):
        letters=sorted(p['versions'])
        require(len(decision)==len(letters) and set(decision)<=set('AIU'),'Invalid reading')
        d=dict(zip(letters,decision,strict=True))
        per_item.append({a:d[label] for a,label in m['labels'].items()})
        for a,v in per_item[-1].items():
            results[a][v]+=1
    alternatives={a:sum(v[a]=='A' and m['labels'][a]!=m['labels']['reference'] for m,v in zip(mapping,per_item))
                  for a in ARMS[:2]}
    write_new(FOLDER/'result.json',dict(totals=results,accepted_nonidentical=alternatives,
              binding=bind,ratings_sha256=sha(grades),strata={'single':24,'multi':24},rows=per_item))
    print(results,alternatives)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score',type=Path)
    args=p.parse_args()
    if args.score:score(args.score)
    else:prepare()

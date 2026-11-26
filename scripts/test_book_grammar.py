"""Measure intact and scrambled visual book priors on published English minimal pairs."""
import json
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch

from scripts.evaluate_book_order_prior import next_logits, restore, ARMS, PUB, WORK, ROOT, base, gpu_owner, configure_cuda_cli

REV='3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
URL=f'https://raw.githubusercontent.com/alexwarstadt/blimp/{REV}/BLiMP.zip'
HASH='172a1a714d72e72a89e8384dd531483f3d0f8a9aee3d33dccaf7992d8cecbb73'


def prepare():
    path=ROOT/'data/raw/pg19-optical/blimp-3e56b06f.zip'
    if not path.exists():
        urlretrieve(URL,path)
    assert base.sha(path)==HASH
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    repertoire=set(base.read(base.BASE/'inventories.json')[0])
    glyph={}
    excluded={}
    cases=[]
    with ZipFile(path) as archive:
        for filename in sorted(n for n in archive.namelist() if n.startswith('data/') and n.endswith('.jsonl')):
            for line in archive.read(filename).splitlines():
                r=json.loads(line)
                two=[r['sentence_good'],r['sentence_bad']]
                for char in set(''.join(two))-glyph.keys():
                    glyph[char]=face.encode(char)[0]
                uid=r['UID']
                if any(len(s)>130 or not s or face.bad.intersection(s) or any(glyph[c] not in repertoire for c in s) for s in two):
                    excluded[uid]=excluded.get(uid,0)+1
                    continue
                cases.append((uid,r['linguistics_term'],int(r['pairID']),sorted(two[0])==sorted(two[1]),
                              [[glyph[c] for c in s] for s in two]))
    assert len(cases)+sum(excluded.values())==67000
    return cases,excluded


@torch.inference_mode()
def main():
    cases,excluded=prepare()
    rows=[x for r in cases for x in r[-1]]
    order=np.argsort([len(x) for x in rows],kind='stable')
    groups=np.array([r[0] for r in cases])
    terms=np.array([r[1] for r in cases])
    bags=np.array([r[3] for r in cases])
    fits={}
    trial={}
    with gpu_owner():
        for arm in ARMS:
            w=restore(arm,'prior','cuda')
            bank=w.model.optical(w.features)
            output=np.empty(len(rows))
            for j in range(0,len(order),192):
                ix=order[j:j+192]
                batch=[rows[i] for i in ix]
                padded=np.zeros((len(ix),max(map(len,batch))),dtype=np.int64)
                for k,row in enumerate(batch):padded[k,:len(row)]=row
                y=torch.as_tensor(padded,device='cuda')
                law=next_logits(w.model,bank,w.inv[0],y[:,:-1])
                score=law.gather(-1,w.local[0][y].clamp_min(0).unsqueeze(-1)).squeeze(-1)
                output[ix]=(score*y.ne(0)).double().sum(1).cpu().numpy()
            scores=output.reshape(-1,2)
            hit=scores[:,0]>scores[:,1]
            trial[arm]=scores
            fits[arm]=dict(all=int(hit.sum()),identical_bag=int(hit[bags].sum()),ties=int((scores[:,0]==scores[:,1]).sum()),
                by_term={t:[int(hit[terms==t].sum()),int((terms==t).sum())] for t in np.unique(terms)},
                by_paradigm={g:[int(hit[groups==g].sum()),int((groups==g).sum())] for g in np.unique(groups)})
            print(arm,fits[arm]['all'],fits[arm]['identical_bag'],flush=True)
    # Resample entire paradigms to account for repeated lexical frames.
    bootstrap={}
    for name,selected in [('all',np.ones(len(cases),dtype=bool)),('identical_bag',bags)]:
        _,ig=np.unique(groups[selected],return_inverse=True)
        delta=np.diff(-trial['intact'],axis=1).ravel()>0
        control=np.diff(-trial['scrambled'],axis=1).ravel()>0
        sizes=np.bincount(ig)
        picks=np.random.default_rng(762035).integers(len(sizes),size=(10000,len(sizes)))
        diff=np.bincount(ig,weights=(delta.astype(int)-control.astype(int))[selected])
        bootstrap[name]=np.quantile(diff[picks].sum(1)/sizes[picks].sum(1),[.025,.975]).tolist()
    table=PUB/'grammar_scores.npz'
    np.savez_compressed(table,uid=groups,pair=np.array([r[2] for r in cases]),bag=bags,**trial)
    base.dump(PUB/'grammar.json',dict(source=URL,sha256=HASH,score_file=base.sha(table),eligible=len(cases),
        excluded=excluded,identical_bag=int(bags.sum()),scores=fits,intervals=bootstrap,
        rule='Full sum of conditional log probabilities of pictured cells, no paragraph stop. Strict comparison; ties earn zero. Exploratory evaluation after pretraining.'))


if __name__=='__main__':
    configure_cuda_cli()
    if not (PUB/'grammar.json').exists():
        main()

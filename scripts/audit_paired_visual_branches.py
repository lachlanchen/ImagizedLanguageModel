"""Measure competing glyphs, exact free-prefix reach and the translation interaction bound."""
from collections import Counter
from zipfile import ZipFile
import json
import numpy as np
import torch
from scripts.measure_paired_visual_writer import ARCHIVE,PUB,base,restore,unpack,pack,pictured_input,CoveredWriter
from scripts.qualify_paired_translation_teacher import ROOT,margins


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    net=restore('broad','cpu')
    with ZipFile(ARCHIVE) as z:
        assert base.sha(ARCHIVE)==base.read(PUB/'writer.json')['archive']
        records={(r['i'],r['t'],r['side']):r for r in json.loads(z.read('records.json'))}
        ids=base.read(PUB/'teacher.json')['audit']
        provenance=base.read(PUB/'provenance.json')
        natural=set(provenance['natural']['audit'])
        reports=[]
        names=base.read(base.BASE/'readback.json')
        face=base.Ink(net.palette)
        for t in (0,1):
            engine=CoveredWriter(t)
            for i in ids:
                targets=[face.encode(records[i,t,s]['ref'])+[1] for s in (0,1)]
                pos=next(k for k,(a,b) in enumerate(zip(*targets)) if a!=b)
                branches=[target[pos] for target in targets]
                for side in (0,1):
                    r=records[i,t,side]
                    length=len(face.encode(r['source']))
                    img=unpack(z.read(f'{i}/{t}/{side}/source.png'),length)
                    source=pack([net.addresses(img)],'cpu')
                    source=pictured_input(source,(255*net.marker(net.features,source).sigmoid()).round().byte())
                    draft=pack([net.addresses(engine.translate(img)[0]) or [0]],'cpu')
                    target=pack([targets[side][:pos+1]],'cpu')
                    law=net.model(net.features,source,target,net.inv[t],t,draft,net.local[t])[0,pos]
                    l0,l1=law[net.local[t][branches]].tolist()
                    best=int(net.inv[t][law.argmax()])
                    free=net.addresses(unpack(z.read(f'{i}/{t}/{side}/answer.png'),r['cells']))+([1] if r['stop'] else [])
                    reached=len(free)>pos and free[:pos]==targets[side][:pos]
                    if reached:
                        assert free[pos]==best,(i,t,side,pos,free[pos],best)
                    reports.append(dict(i=i,t=t,side=side,prefix=pos,branches=branches,log_scores=[l0,l1],top=best,
                                   reached=reached,aligned=bool(best==branches[side]),third=best not in branches,
                                   pair_correct=bool((l0>l1) if side==0 else (l1>l0))))
        assert not face.bad
    whole=np.load(PUB/'writer-scores.npz')['scores']
    totals={}
    for k,group in [('corpus',natural),('constructed',set(ids)-natural)]:
        rows=[r for r in reports if r['i'] in group]
        stats={flag:sum(r[flag] for r in rows) for flag in ('reached','aligned','third','pair_correct')}
        stats['correct_but_third']=sum(r['pair_correct'] and r['third'] for r in rows)
        stats['wrong_alternative']=len(rows)-stats['aligned']-stats['third']
        stats['reached_aligned']=sum(r['reached'] and r['aligned'] for r in rows)
        ms=margins(whole[sorted(group)])
        individual=(ms>=.1).all(axis=-1)
        feasible=ms.sum(axis=-1)>=.2
        assert not np.any(individual & ~feasible)
        stats.update(row_decisions=len(rows),directions=individual.size,already_at_margin=int(individual.sum()),
                     offset_repairable=int((feasible & ~individual).sum()),interaction_too_small=int((~feasible).sum()))
        totals[k]=stats
    base.dump(PUB/'branches.json',dict(archive=base.sha(ARCHIVE),whole=base.sha(PUB/'writer-scores.npz'),
              proof=base.sha(ROOT/'formal/span_list/PairedTranslationChoice.lean'),
              protocol=base.sha(ROOT/'references/paired_visual_branch_protocol.md'),totals=totals,rows=reports))
    print(totals,flush=True)


if __name__=='__main__':
    main()

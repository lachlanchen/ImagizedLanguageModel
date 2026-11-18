"""Recount or repaint frozen visual answer intervals and verify their stored pixel streams."""
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import resource
import time
import zipfile

import numpy as np
import torch

from ilm.visual_lm.grounded_detail_writer import GroundedDetailWriter
from ilm.visual_lm.residual_ink_writer import ResidualInkWriter
from scripts.diagnose_visual_context_gradients import load,require,sha,write_new
from scripts.evaluate_grounded_detail_writer import roster,interval
from scripts.fit_grounded_detail_writer import OUT as BASE,TargetRasterizer, Rasterizer,guard,packed
from scripts.fit_unseen_detail_residual import OUT as RESIDUAL
from scripts.train_visual_extractive_reader import save_page


def main(repaint):
    start=time.monotonic()
    guard(start)
    evaldir=BASE/'evaluation'
    earlier=load(evaldir/'result.json')
    require(all(sha(evaldir/f)==h for f,h in earlier['files'].items()),'Saved outputs differ from original result')
    groups,_=roster(Rasterizer(),TargetRasterizer())
    records=load(evaldir/'scores.json')
    work=(RESIDUAL/'evaluation') if repaint else (BASE/'recount')
    work.mkdir(exist_ok=False)
    totals=defaultdict(Counter)
    cache={}
    counts=Counter()
    deltas=defaultdict(lambda:np.zeros(2))
    case_rows={}
    if repaint:
        fit=load(RESIDUAL/'fit.json')
        require(sha(RESIDUAL/'neural.pt')==fit['neural_sha256'] and sha(BASE/'writer.npz')==load(BASE/'fit.json')['checkpoint_sha256'],'Weights changed')
        painter=ResidualInkWriter.load(GroundedDetailWriter.load(BASE/'writer.npz'),RESIDUAL/'neural.pt')
    idx=0
    import contextlib
    with contextlib.ExitStack() as stack:
        archive=stack.enter_context(zipfile.ZipFile(evaldir/'emitted.zip'))
        output=stack.enter_context(zipfile.ZipFile(work/'emitted.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=2)) if repaint else None
        for gid,cases in groups:
            guard(start)
            for c in cases:
                r=records[idx]
                require(r['id']==c['label'] and r['group']==gid,'Question order')
                gold={y for _,y in c['refs']}
                source_gold={x for x,_ in c['refs']}
                condition_results={}
                for name,previous in r['predictions'].items():
                    a,b=previous['bounds']
                    source=c['p'][a*128:b*128]
                    if name=='blank_passage':
                        source=bytes(len(source))
                    require(0<=a<b<=len(c['p'])//128 and (source in source_gold)==previous['source_exact'],'Source alignment')
                    before=archive.read(f'{idx}/{name}_detail.bin')
                    coarse=archive.read(f'{idx}/{name}_coarse.bin')
                    require(len(before)==len(coarse)==len(source) and (before in gold)==previous['detail_exact'] and
                            (coarse in gold)==previous['coarse_exact'] and (source in gold)==previous['copy_exact'],
                            'Complete pixel recount')
                    counts['streams_recounted']+=2
                    value=before
                    if repaint:
                        words=[source[n:n+128] for n in range(0,len(source),128)]
                        absent=list(dict.fromkeys(w for w in words if w not in cache))
                        if absent:
                            raw=painter.paint(packed(b''.join(absent)))
                            cache.update((w,v.tobytes()) for w,v in zip(absent,raw,strict=True))
                        value=b''.join(cache[w] for w in words)
                        output.writestr(f'{idx}/{name}.bin',value)
                    success=value in gold
                    s=totals[name]
                    s['count']+=1;s['exact']+=success;s['differs_from_source']+=value!=source
                    s['exact_changed']+=success and value!=source
                    s['changed_vs_count']+=value!=before
                    if previous['source_exact']:
                        ref=next(y for x,y in c['refs'] if x==source)
                        pred=np.unpackbits(packed(value),axis=1);chosen=np.unpackbits(packed(ref),axis=1)
                        mass=(pred+chosen).sum(1)
                        s['selected_correctly']+=1
                        s['aligned_cells']+=len(pred)
                        s['aligned_ink_f1_sum']+=np.divide(2*(pred & chosen).sum(1),mass,out=np.ones(len(pred)),where=mass>0).sum()
                    condition_results[name]=success
                    if name=='normal':
                        deltas[gid]+=[int(success)-int(previous['detail_exact']),1]
                    if repaint and name=='normal' and c['label'] in earlier['examples']['pair']:
                        side='ab'[earlier['examples']['pair'].index(c['label'])]
                        save_page(work/f'pair_answer_{side}.png',packed(value))
                case_rows[c['label']]=dict(g=gid,partner=r['partner'],gold=gold,success=condition_results['normal'])
                idx+=1
    require(idx==len(records) and counts['streams_recounted']==2*sum(v['count'] for v in totals.values()),'Incomplete outputs')
    pairs=set()
    both=0
    for label,row in case_rows.items():
        other=row['partner']
        if other is None:
            continue
        pair=tuple(sorted([label,other]))
        if pair in pairs or row['gold'] & case_rows[other]['gold']:
            continue
        pairs.add(pair)
        both+=int(row['success'] and case_rows[other]['success'])
    if not repaint:
        require(all(v['exact']==earlier['scores'][k]['detail_exact'] for k,v in totals.items())
                and both==earlier['pairs']['both_complete_exact'], 'Original metrics differ')
    summary=dict(conditions={k:dict(v) for k,v in totals.items()},questions=idx,known_range_pairs=len(pairs),both_correct=both,
                 counts=dict(counts),normal_exact_minus_count=interval(list(deltas.values())),
                 source_report_sha256=sha(evaldir/'result.json'),elapsed_seconds=time.monotonic()-start,
                 rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,program_sha256=sha(Path(__file__)))
    if repaint:
        summary.update(emitted_sha256=sha(work/'emitted.zip'),physical_pass=fit['transfer_pass'],
                       preserves_normal_exact=totals['normal']['exact']>=earlier['scores']['normal']['detail_exact'])
    guard(start)
    write_new(work/'result.json',summary)
    print(idx,totals['normal']['exact'],both,counts['streams_recounted'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    main(args.residual)

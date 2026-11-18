"""Train one inverse-square-root-frequency arm and compare both frozen image writers."""
from collections import Counter
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import PicturedIdiom
from scripts.compare_passage_diversity import (
    DATA, FRESH, HINTS, JOINT, PREVIOUS, ROOT, OUT as DIVERSITY, corpora, fit, guard, infer,
    integrated_check, load, load_initial, load_split, local_counts, matching_pairs, paired_interval,
    physical_bank, prepare, require, sha, summarize, verified_split, write_new,
)

OUT = ROOT/'artifacts/expression_frequency_weight_v1'


def main():
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    require('4090' in torch.cuda.get_device_name(), 'Expected registered GPU')
    prior = load(DIVERSITY/'evaluation.json')
    require(prior['fresh']['arms']['diverse']['exact']==550 and not all(prior['criteria'].values()),'Missing prior result')
    train, augmented = corpora()
    before = verified_split(DIVERSITY/'fresh')
    bank = np.load(DIVERSITY/'pool/pixels.npy')
    OUT.mkdir()
    held = OUT/'fresh'
    held.mkdir()
    excluded = [verified_split(k) for k in (FRESH,HINTS/'confirmation',JOINT/'confirmation',PREVIOUS,DIVERSITY/'pool')]
    test, expanded = prepare(train,load_split(DATA/'validation.npz'),bank,buckets=range(104,112),destination=held,
                             excluded=[*excluded,before])
    initial = load_initial()
    features,bits,feedback,reachable = physical_bank(initial,bank)
    freq = Counter(map(bytes,augmented['target']))
    frequency = np.array([freq[a.tobytes()] for a in augmented['target']])
    weights = np.reciprocal(np.sqrt(frequency))
    weights = (weights/weights.mean()).astype(np.float32)
    require(abs(weights.mean()-1)<1.e-6 and len(weights)==142878,'Invalid normalized weights')
    np.save(OUT/'weights_by_row.npy',weights)
    x,y = [torch.as_tensor(augmented[k].astype(np.int32),device='cuda') for k in ('context','target')]
    torch.manual_seed(720)
    fit('reference',initial,x,y,None,features,bits,feedback,start,epochs=4,destination=OUT/'weighted',
        row_weights=torch.as_tensor(weights,device='cuda'))
    original_frequency = Counter(map(bytes, train['target']))
    result = {}
    for split, dev, px in [('previous',before,bank), ('fresh',test,expanded)]:
        folder = OUT/split if split=='fresh' else OUT/'previous'
        folder.mkdir(exist_ok=split=='fresh')
        pair = matching_pairs(dev['context'],dev['target'],dev['group'])
        hits,stats = {},{}
        for name in ('unweighted','weighted'):
            path = DIVERSITY/'diverse' if name=='unweighted' else OUT/name
            dest = folder/name
            if (split,name)==('previous','unweighted'):
                dest = DIVERSITY/'evaluation_fresh/diverse'
                require(sha(dest/'generated.npz')==prior['fresh']['arms']['diverse']['archive_sha256'],'Prior streams changed')
            else:
                infer(path/'weights.pt',dest,augmented,dev,px,start)
            hit,measurements = summarize(load_split(dest/'generated.npz'),dev,px,augmented,original_frequency,pair)
            hits[name],stats[name] = hit,measurements
            record = load(dest/'result.json')
            require(measurements['exact']==record['complete_exact'],'Physical recount differs')
            measurements.update(known=record['training_target_shape_hits'],emitted=record['emitted_cells'],
                                renderer=record['oracle_motor'],
                                archive_sha256=sha(dest/'generated.npz'))
        w,u = hits['weighted'],hits['unweighted']
        result[split] = dict(rows=len(dev['target']),groups=len(np.unique(dev['group'])),arms=stats,
                 local_original=int(local_counts(train,dev).sum()),local_combined=int(local_counts(augmented,dev).sum()),
                 gains=int((w&~u).sum()),losses=int((u&~w).sum()),interval_pp=paired_interval(w,u,dev['group']))
        print(split,{k:v['exact'] for k,v in stats.items()},flush=True)
    fresh = result['fresh']
    w,u = fresh['arms']['weighted'],fresh['arms']['unweighted']
    def rare(arm):
        return sum(row['correct'] for row in arm['arm_frequency'] if row['lo'] in (1,6))
    prev = result['previous']['arms']
    criteria = dict(interval=fresh['interval_pp'][0]>0, concentration=w['most_frequent_ten']<u['most_frequent_ten'],
                    rare=rare(w)>rare(u), hard_pairs=w['both_correct']>=u['both_correct'],ink=w['known']>=.99*w['emitted'],
                    preceding=prev['weighted']['exact']>prev['unweighted']['exact'])
    result['criteria'] = criteria
    if all(criteria.values()):
        state = torch.load(OUT/'weighted/weights.pt',weights_only=True,map_location='cpu')
        net = PicturedIdiom(state['landmarks']).cuda().eval()
        net.load_state_dict(state)
        result['historical'] = integrated_check(net,test,expanded,start,generated=held/'weighted/generated.npz')
    result.update(weights_sha256=sha(OUT/'weights_by_row.npy'),minimum_weight=float(weights.min()),
                  maximum_weight=float(weights.max()), distinct_target_expressions=len(freq),fit=load(OUT/'weighted/fit.json'),
                  registered_protocol_sha256=sha(ROOT/'references/expression_frequency_weight_protocol.md'),
                  program_sha256=sha(Path(__file__)),device=torch.cuda.get_device_name(),seconds=time.monotonic()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    guard(start)
    write_new(OUT/'evaluation.json',result)

if __name__=='__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

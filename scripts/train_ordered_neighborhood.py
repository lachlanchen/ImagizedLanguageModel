"""Train and evaluate matched independent visual writers with a local context branch."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import PicturedIdiom
from ilm.visual_lm.ordered_neighborhood import NeighborhoodWriter
from scripts.compare_passage_diversity import (
    DATA, FRESH, HINTS, JOINT, PREVIOUS, ROOT, OUT as DIVERSITY, corpora, fit, guard, infer, integrated_check,
    load, load_initial, load_split, local_counts, matching_pairs, paired_interval, physical_bank, prepare,
    require, sha, summarize, verified_split, write_new,
)
from scripts.test_expression_frequency_weight import OUT as WEIGHT

OUT = ROOT/'artifacts/ordered_neighborhood_v1'
ARMS = ('start','uniform','shuffled','aligned')


def train(start):
    original, combined = corpora()
    initial = load_initial()
    bank = np.load(DIVERSITY/'pool/pixels.npy')
    features,bits,feedback,_ = physical_bank(initial,bank)
    OUT.mkdir()
    rng_seed = 720
    torch.manual_seed(rng_seed)
    augmented = NeighborhoodWriter(initial['landmarks'])
    missing, extra = augmented.load_state_dict(initial,strict=False)
    require(set(missing)=={'neighborhood.'+k for k in augmented.neighborhood.state_dict()} and not extra,
            'Incomplete pre-existing model')
    augmented.eval()
    source = PicturedIdiom(initial['landmarks']).eval()
    source.load_state_dict(initial)
    probe = features[combined['context'][:16].astype(np.int32)].cpu()
    with torch.inference_mode():
        require(torch.equal(source.read(probe),augmented.read(probe)), 'Zero residual changed initial reading')
    state = augmented.state_dict()
    x,y = [torch.as_tensor(combined[k].astype(np.int32),device='cuda') for k in ('context','target')]
    require(len(x)==142878,'Wrong corpus')
    for arm in ('shuffled','aligned'):
        torch.manual_seed(rng_seed)
        fit('reference',state,x,y,None,features,bits,feedback,start,epochs=4,destination=OUT/arm,
            model_type=NeighborhoodWriter,shuffle_neighborhood=arm=='shuffled')
    write_new(OUT/'fits.json',dict(added_parameters=sum(v.numel() for v in augmented.neighborhood.parameters()),
              total_parameters=sum(v.numel() for v in augmented.parameters()), original_rows=len(original['target']),
              unique_rows=len(x),exposures=4*len(x),updates=4*((len(x)+191)//192),
              arms={k:load(OUT/k/'fit.json') for k in ('shuffled','aligned')}))


def evaluate(start):
    original,combined = corpora()
    previous = verified_split(WEIGHT/'fresh')
    bank = np.load(WEIGHT/'fresh/pixels.npy')
    held = OUT/'fresh'
    held.mkdir()
    exclude = [verified_split(k) for k in (FRESH,HINTS/'confirmation',JOINT/'confirmation',PREVIOUS,
                                           DIVERSITY/'fresh',DIVERSITY/'pool')]
    fresh,expanded = prepare(original,load_split(DATA/'validation.npz'),bank,buckets=range(112,124),
                             destination=held,excluded=[*exclude,previous])
    freq = Counter(map(bytes,original['target']))
    seen = set(map(bytes,combined['context'][:,20:28]))
    report = {}
    for name,test,pixels in [('previous',previous,bank),('fresh',fresh,expanded)]:
        folder = OUT/('evaluate_'+name)
        folder.mkdir()
        local = local_counts(combined,test)
        repeated = np.array([v.tobytes() in seen for v in test['context'][:,20:28]])
        pair = matching_pairs(test['context'],test['target'],test['group'])
        stats,hits = {},{}
        for arm,model in zip(ARMS,(JOINT/'reference',DIVERSITY/'diverse',OUT/'shuffled',OUT/'aligned')):
            path = folder/arm
            if name=='previous' and arm=='uniform':
                path = WEIGHT/'fresh/unweighted'
                require(sha(path/'generated.npz')==load(WEIGHT/'evaluation.json')['fresh']['arms']['unweighted']['archive_sha256'],
                        'Changed earlier uniform output')
            else:
                infer(model/'weights.pt',path,combined,test,pixels,start,
                      model_type=NeighborhoodWriter if arm in ('aligned','shuffled') else PicturedIdiom)
            stream = load_split(path/'generated.npz')
            hit,row = summarize(stream,test,pixels,combined,freq,pair)
            generation = load(path/'result.json')
            require(row['exact']==generation['complete_exact'],'Mismatch in physical recount')
            row.update(known=generation['training_target_shape_hits'],emitted=generation['emitted_cells'],
                       repeat_exact=int(hit[repeated].sum()),new_exact=int(hit[~repeated].sum()),
                       local_misses_correct=int(hit[~local].sum()),local_hits_correct=int(hit[local].sum()),
                       archive_sha256=sha(path/'generated.npz'))
            stats[arm],hits[arm] = row,hit
        a = hits['aligned']
        comparisons = {k:dict(gains=int((a&~v).sum()),losses=int((~a&v).sum()),
                               paired_interval_pp=paired_interval(a,v,test['group'])) for k,v in hits.items() if k!='aligned'}
        earlier = hits['uniform'] & ~local
        report[name] = dict(rows=len(a),groups=len(np.unique(test['group'])),eight_seen=int(repeated.sum()),
                  local_exact=int(local.sum()),local_eight=int(local[repeated].sum()),arms=stats,comparisons=comparisons,
                  uniform_only_vs_local=int(earlier.sum()),aligned_retained=int((earlier&a).sum()),
                  aligned_new_vs_uniform_and_local=int((~hits['uniform']&~local&a).sum()))
        print(name,{k:v['exact'] for k,v in stats.items()},report[name]['comparisons'],flush=True)
    f=report['fresh']
    a,u=f['arms']['aligned'],f['arms']['uniform']
    criteria=dict(interval=all(f['comparisons'][k]['paired_interval_pp'][0]>0 for k in ('uniform','shuffled')),
                  eight=a['repeat_exact']>u['repeat_exact'],new=a['new_exact']>u['new_exact'],
                  local_misses=a['local_misses_correct']>=u['local_misses_correct'], ink=a['known']>=.99*a['emitted'])
    report['criteria']=criteria
    if all(criteria.values()):
        state=torch.load(OUT/'aligned/weights.pt',weights_only=True,map_location='cpu')
        net=NeighborhoodWriter(state['landmarks']).cuda().eval()
        net.load_state_dict(state)
        report['historical']=integrated_check(net,fresh,expanded,start,generated=OUT/'evaluate_fresh/aligned/generated.npz')
    write_new(OUT/'evaluation.json',report)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['train','evaluate'])
    phase=parser.parse_args().phase
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        guard(start)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        require('4090' in torch.cuda.get_device_name(),'Expected registered GPU')
        (train if phase=='train' else evaluate)(start)
        guard(start)
        write_new(OUT/(phase+'_resources.json'),dict(seconds=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),program_sha256=sha(Path(__file__))))

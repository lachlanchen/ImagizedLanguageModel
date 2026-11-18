"""Generate complete answers from matched training and held-out phrase contexts."""
from collections import Counter
import fcntl
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from ilm.visual_lm.ordered_neighborhood import NeighborhoodWriter
from scripts.train_ordered_neighborhood import OUT, DIVERSITY, corpora, verified_split, load_split, guard, require, sha, load, write_new


def select_donors(train, test):
    counts, first = {}, {}
    for i,(x,y) in enumerate(zip(train['context'],train['target'])):
        key,word = x[20:28].tobytes(), y.tobytes()
        counts.setdefault(key,Counter())[word] += 1
        first.setdefault((key,word), i)
    rows, donors = [],[]
    for i,x in enumerate(test['context']):
        key = x[20:28].tobytes()
        if key in counts:
            rows.append(i)
            donors.append(first[key,counts[key].most_common(1)[0][0]])
    return np.array(rows),np.array(donors)


@torch.inference_mode()
def main():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    final=load(OUT/'evaluation.json')['fresh']
    _,train=corpora()
    test=verified_split(OUT/'fresh')
    bank=np.load(OUT/'fresh/pixels.npy')
    rows,donors=select_donors(train,test)
    x,source=test['context'][rows],train['context'][donors]
    target=test['target'][rows]
    gold=bank[target]
    agree=(train['target'][donors]==target).all(1)
    require(np.array_equal(x[:,20:28],source[:,20:28]) and len(rows)==final['eight_seen'] and
            int(agree.sum())==final['local_eight'],'Retrieval differs from prior local baseline')
    blank=np.zeros_like(x)
    blank[:,20:28]=x[:,20:28]
    dest=OUT/'transfer'
    dest.mkdir()
    np.savez(dest/'sources.npz',rows=rows,donors=donors,same_gold=agree)
    report={}
    for arm in ('uniform','shuffled','aligned'):
        folder=DIVERSITY/'diverse' if arm=='uniform' else OUT/arm
        path=folder/'weights.pt'
        require(sha(path)==load(folder/'fit.json')['weights_sha256'],'Weights changed')
        state=torch.load(path,weights_only=True,map_location='cpu')
        model_type=PicturedIdiom if arm=='uniform' else NeighborhoodWriter
        net=model_type(state['landmarks']).cuda().eval()
        net.load_state_dict(state)
        features=visual_input(torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool(),net.landmarks)
        oldpath=OUT/'evaluate_fresh'/arm/'generated.npz'
        require(sha(oldpath)==final['arms'][arm]['archive_sha256'],'Changed production')
        old=load_split(oldpath)
        # Reuse the recorded full-size production batches for held-out inputs. Independent
        # donor and mask conditions have exactly the same selected rows and order.
        conditions={'held':(old['packed'][rows],old['length'][rows],old['stopped'][rows])}
        for case,contexts in [('donor',source),('local',blank),('held_replay',x)]:
            chunks=[net.generate(features[torch.as_tensor(block.astype(np.int32),device='cuda')])
                    for block in np.array_split(contexts,(len(contexts)+127)//128)]
            ink,n,stop=(np.concatenate(v) for v in zip(*chunks))
            ink[np.arange(ink.shape[1])[None,:]>=n[:,None]]=0
            conditions[case]=(ink,n,stop)
        matches={}
        archive={}
        for case,(ink,n,stop) in conditions.items():
            archive.update({case+'_packed':ink,case+'_length':n,case+'_stopped':stop})
            matches[case]=stop&(n==4)&(ink[:,:4]==gold).all((1,2))
        p,n,s=conditions['held'];q,m,t=conditions['held_replay']
        identical=(p==q).all((1,2))&(n==m)&(s==t)
        np.savez_compressed(dest/(arm+'.npz'),**archive)
        a,b,c=(matches[k] for k in ('donor','held','local'))
        report[arm]=dict(all_held=int(b.sum()),all_donor_vs_held_gold=int(a.sum()),all_local=int(c.sum()),
                  same_gold_held=int((b&agree).sum()),same_gold_donor=int((a&agree).sum()),same_gold_local=int((c&agree).sum()),
                  donor_and_held=int((a&b&agree).sum()),donor_only=int((a&~b&agree).sum()),held_only=int((~a&b&agree).sum()),
                  replay_identical=int(identical.sum()),replay_changed_hits=int((matches['held']!=matches['held_replay']).sum()),
                  output_sha256=sha(dest/(arm+'.npz')))
    write_new(dest/'result.json',dict(rows=len(rows),distinct_donors=len(set(donors)),same_gold=int(agree.sum()),
              same_gold_distinct_donors=len(set(donors[agree])),results=report,source_sha256=sha(dest/'sources.npz'),
              seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(report,flush=True)

if __name__=='__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

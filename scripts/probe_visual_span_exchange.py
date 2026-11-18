"""Score matched real inserted phrases in both target contexts with original or reversed sources."""
import argparse
import fcntl
import time
from collections import defaultdict

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_alignment import observed_prefix_logp
from scripts.evaluate_visual_spans import insert_runs
from scripts.prepare_edit_diversity import COHORTS
from scripts.train_visual_spans import ARMS, OUT, ROOT, WORK, frozen, guard, load, require, restore, sha, tensor_pad, tiles, write_new


def pairs(rows):
    waiting=defaultdict(list)
    used=set()
    selected=[]
    for i,r in enumerate(rows):
        for _,j,k in insert_runs(r['a']):
            if k-j>4 or set(r['x']) & set(r['y'][j:k]):
                continue
            before=r['y'][j-1] if j else -1
            after=r['y'][k] if k<len(r['y']) else -1
            bucket=waiting[(before,after,k-j)]
            for h,left,right in bucket:
                old=rows[h]
                if h in used or h==i or old['g']==r['g'] or old['y'][left:right]==r['y'][j:k]:
                    continue
                if set(old['x']) & set(r['y'][j:k]) or set(r['x']) & set(old['y'][left:right]):
                    continue
                selected.append(((h,left,right),(i,j,k)))
                used.update((i,h))
                break
            if i in used:
                break
            bucket.append((i,j,k))
    return selected


@torch.inference_mode()
def score(net,ink,rows,selected,start,*,through_phrase=False):
    features=visual_input(ink,net.landmarks)
    q=load(OUT/'preparation.json')['question_tiles'][0][0]
    buckets=defaultdict(list)
    for ix,pair in enumerate(selected):
        for side,(i,j,k) in enumerate(pair):
            h,u,v=pair[1-side]
            x,y=rows[i]['x'],rows[i]['y']
            other=y[:j]+rows[h]['y'][u:v]+y[k:]
            for reverse in (0,1):
                for foil,target in enumerate((y,other)):
                    observed = target[:k] if through_phrase else target
                    buckets[len(x)].append((ix,side,reverse,foil,x[::-1] if reverse else x,observed))
    values=np.empty((len(selected),2,2,2))
    for bucket in buckets.values():
        for first in range(0,len(bucket),8):
            group=bucket[first:first+8]
            x=torch.tensor([r[4] for r in group],device='cuda')
            y,lengths=tensor_pad([r[5] for r in group],x.device)
            question=torch.tensor(q,device='cuda')[None].expand(len(x),-1)
            code=ink[y][:,:,net.landmarks]
            action,glyph=net.score_grid(features[x],code,features[question],torch.full_like(lengths,len(q)))
            logp=net.marginal(action,glyph,ink[x][:,:,net.landmarks],code,lengths)
            if through_phrase:
                prefix,terminal=observed_prefix_logp(action,glyph,ink[x][:,:,net.landmarks],code,lengths)
                torch.testing.assert_close(terminal,logp,atol=.001,rtol=1e-6)
                logp=prefix
            for row,prob in zip(group,logp.tolist(),strict=True):
                values[row[:4]]=prob
            guard(start)
    margin=(values[:,:,:,0]-values[:,:,:,1]).sum(1)
    diff=margin[:,0]-margin[:,1]
    weights=np.random.default_rng(1901).integers(len(diff),size=(2000,len(diff)))
    return dict(pairs=len(selected),multiglyph_pairs=sum(a[2]-a[1]>1 for a,b in selected),
                assigned=int((margin[:,0]>0).sum()),reversed_assigned=int((margin[:,1]>0).sum()),
                mean_order_effect_nats=float(diff.mean()),pair_bootstrap_ci_nats=np.quantile(diff[weights].mean(1),[.025,.975]).tolist(),
                mean_margin_nats=margin.mean(0).tolist(),pair_margins_nats=margin.tolist())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-phrase',action='store_true',help='Score only the observed prefix through the exchanged phrase')
    args=parser.parse_args()
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        start=time.monotonic()
        guard(start)
        ink=torch.from_numpy(np.unpackbits(tiles(),axis=1)).cuda().bool()
        for split in COHORTS:
            rows=list(frozen(split))
            selected=pairs(rows)
            result=dict(pairs=selected)
            for a in ARMS:
                path=WORK/a/'editor.safetensors'
                require(sha(path)==load(WORK/a/'fit.json')['checkpoint_sha256'],'Editor changed')
                result[a]=score(restore(path).eval(),ink,rows,selected,start,through_phrase=args.through_phrase)
            suffix='_prefix_exchange.json' if args.through_phrase else '_exchange.json'
            if args.through_phrase:
                require(load(WORK/(split+'_exchange.json'))['pairs']==[[list(u),list(v)] for u,v in selected],
                        'Prefix probe changed phrase pairs')
            write_new(WORK/(split+suffix),result)
            print(split,[(a,result[a]['pairs'],result[a]['assigned'],result[a]['pair_bootstrap_ci_nats']) for a in ARMS])

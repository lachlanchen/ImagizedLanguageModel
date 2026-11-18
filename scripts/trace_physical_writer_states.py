"""Locate losses of the canonical path in the actual physical beam."""
from collections import Counter
import fcntl
import time

import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.idiom_ink import GROUPS, STOP, PicturedIdiom, nibbles
from scripts.measure_physical_writer_states import OUT,PARENT,guard,load,require,sha,write_new


@torch.inference_mode()
def main():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    receipt=load(OUT/'result.json')
    require(not receipt['passed'] and sha(OUT/'states.npz')==receipt['states_sha256'],'Expected failed state qualification')
    require(sha(OUT/'fitted.npz')==receipt['results']['fitted']['outputs_sha256'],'Fitted pixels changed')
    path=PARENT/'diverse/weights.pt'
    require(sha(path)==receipt['parent_sha256'],'Checkpoint changed')
    state=torch.load(path,weights_only=True,map_location='cpu')
    net=PicturedIdiom(state['landmarks']).cuda().eval()
    net.load_state_dict(state)
    with np.load(OUT/'states.npz') as a:
        words,counts,reachable=(a[k] for k in ('words','counts','reachable'))
        states=torch.as_tensor(a['fitted'],device='cuda')
    with np.load(OUT/'fitted.npz') as a:
        old,n,stop,correct,original_score=(a[k] for k in ('packed','length','stopped','correct','logp'))
    bank=np.load(PARENT/'pool/pixels.npy')
    images=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    bits=images[:,net.landmarks].float()
    feedback=(torch.cat([net.motor(b) for b in bits.split(128)])[:,net.landmarks]>0).float()
    ranks,pruned,winner_score,source_score,replay=[],[],[],[],[]
    for i in range(0,len(words),128):
        guard(start)
        z=states[i:i+128]
        ids=torch.as_tensor(words[i:i+len(z)].astype(np.int32),device='cuda')
        symbol=nibbles(bits[ids]).flatten(1)
        labels=F.pad(symbol,(0,1),value=STOP)
        steps=torch.arange(labels.shape[1],device=z.device)
        prev=F.pad(symbol,(1,0),value=17)
        feed=F.pad(feedback[ids],(0,0,1,0))[:,steps//GROUPS]
        seq=net.inputs(prev,(steps%GROUPS).expand_as(labels),feed,z[:,None].expand(-1,len(steps),-1))
        h,_=net.causal(seq,z[None].contiguous())
        logits=net.next_pattern(h).float()
        logits[:,steps%GROUPS!=0,STOP]=-torch.inf
        gold=logits.gather(-1,labels[...,None])
        ranks.append((1+(logits>gold).sum(-1)).cpu().numpy())
        source_score.append(logits.log_softmax(-1).gather(-1,labels[...,None]).sum(1).flatten().cpu().numpy())
        alive=torch.zeros((len(z),4),device=z.device,dtype=torch.bool)
        alive[:,0]=True
        first=torch.full((len(z),),-1,device=z.device)
        winning=None

        def observe(t,parent,sym,scores):
            nonlocal alive,winning
            winning=scores[:,0]
            if t<=4*GROUPS:
                alive=alive.gather(1,parent)&(sym==labels[:,t,None])
                first.masked_fill_((first<0)&~alive.any(1),t)

        ink,length,stopped=net.generate_from_context(z,observe_step=observe)
        ink[np.arange(ink.shape[1])[None,:]>=length[:,None]]=0
        replay.append((ink==old[i:i+len(z)]).all((1,2))&(length==n[i:i+len(z)])&(stopped==stop[i:i+len(z)]))
        pruned.append(first.cpu().numpy())
        winner_score.append(winning.cpu().numpy())
    ranks,pruned,winner_score,source_score,replay=map(np.concatenate,(ranks,pruned,winner_score,source_score,replay))
    drift=float(np.max(np.abs(source_score-original_score)))
    require(replay.all() and drift<.001,'Physical replay or supplied path differs')
    dest=OUT/'trace'
    dest.mkdir()
    np.savez_compressed(dest/'paths.npz',ranks=ranks,first_pruned=pruned,winner_logp=winner_score,source_logp=source_score,
                        correct=correct,reachable=reachable,counts=counts)
    report={}
    for k,subset in [('all',np.ones_like(correct)),('rare',counts<=5),('canonical_reachable',reachable)]:
        mask=subset&~correct
        diff=winner_score[mask]-source_score[mask]
        lost=pruned[mask]
        have=mask&(pruned>=0)
        report[k]=dict(failures=int(mask.sum()),renderer_changed=int((mask&~reachable).sum()),
            winner_higher=int((diff>.0001).sum()),source_higher=int((diff<-.0001).sum()),ties=int((np.abs(diff)<=.0001).sum()),
            never_pruned=int((lost<0).sum()),lost_by_glyph=dict(Counter(str(min(5,1+int(t)//GROUPS)) for t in lost if t>=0)),
            ranks_at_loss=dict(Counter(str(v) for v in ranks[np.where(have)[0],pruned[have]])),
            fully_greedy=int((ranks[mask]==1).all(1).sum()))
    write_new(dest/'result.json',dict(exact_stream_replays=int(replay.sum()),max_nll_drift=drift,results=report,
                    trace_sha256=sha(dest/'paths.npz'),seconds=time.monotonic()-start))
    print(report,flush=True)

if __name__=='__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

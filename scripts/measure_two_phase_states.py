"""Compare one and two answer-informed states in the same frozen physical writer."""
import fcntl
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import GROUPS, PicturedIdiom
from scripts.measure_physical_writer_states import OUT as SINGLE, PARENT, ROOT, guard, load, require, sha, write_new, physical_score

OUT=ROOT/'artifacts/two_phase_writing_states_v1'


def main():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    single=load(SINGLE/'result.json')
    require(load(SINGLE/'trace/result.json')['results']['all']['winner_higher']==182 and
            sha(SINGLE/'states.npz')==single['states_sha256'],'Single-state diagnosis differs')
    OUT.mkdir()
    path=PARENT/'diverse/weights.pt'
    require(sha(path)==single['parent_sha256'],'Writer changed')
    state=torch.load(path,weights_only=True,map_location='cpu')
    net=PicturedIdiom(state['landmarks']).cuda().eval().requires_grad_(False)
    net.load_state_dict(state)
    with np.load(SINGLE/'states.npz') as z:
        centers=torch.as_tensor(z['centroids'],device='cuda')
        words,counts,reachable=(z[k] for k in ('words','counts','reachable'))
    with np.load(SINGLE/'fitted.npz') as a:
        earlier=a['correct']
    bank=np.load(PARENT/'pool/pixels.npy')
    images=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    bits=images[:,net.landmarks].float()
    feedback=torch.cat([(net.motor(b)[:,net.landmarks]>0).float() for b in bits.split(128)])
    ids=torch.tensor(words.astype(np.int32),device='cuda')
    fitted=[]
    net.causal.train()
    for i in range(0,len(words),128):
        guard(start)
        batch=ids[i:i+128]
        z=centers[i:i+len(batch)]
        if i==0:
            a=physical_score(net,z,bits[batch],feedback[batch])
            b=physical_score(net,z,bits[batch],feedback[batch],later_context=z)
            require(torch.equal(a,b),'Duplicated-state likelihood differs')
            require(all(np.array_equal(l,r) for l,r in zip(net.generate_from_context(z),
                          net.generate_from_context(z,later_context=z))), 'Duplicated-state decoding differs')
        params=torch.nn.Parameter(z.clamp(-.999999,.999999).atanh()[:,None].repeat(1,2,1))
        opt=torch.optim.Adam([params],lr=.08)
        for _ in range(100):
            first,second=params.tanh().unbind(1)
            score=physical_score(net,first,bits[batch],feedback[batch],later_context=second)
            opt.zero_grad(set_to_none=True)
            (-score.sum()/(4*GROUPS+1)).backward()
            require(torch.isfinite(params.grad).all().item(),'Nonfinite state gradient')
            opt.step()
        fitted.append(params.detach().tanh())
    net.causal.eval()
    z=torch.cat(fitted)
    np.savez_compressed(OUT/'states.npz',words=words,counts=counts,states=z.cpu().numpy())
    outs=[net.generate_from_context(block[:,0],later_context=block[:,1]) for block in z.split(128)]
    ink,n,stop=(np.concatenate(v) for v in zip(*outs))
    used=np.arange(ink.shape[1])[None,:]<n[:,None]
    ink[~used]=0
    correct=(ink[:,:4]==bank[words]).all((1,2))&(n==4)&stop
    with torch.no_grad():
        logp=torch.cat([physical_score(net,v[:,0],bits[t],feedback[t],later_context=v[:,1]) for v,t in zip(z.split(128),ids.split(128))])
    bins={}
    for name,mask in [('all',np.ones_like(correct)),('once_to_five',counts<=5),('six_to_twenty',(counts>=6)&(counts<=20)),
                      ('over_twenty',counts>=21),('reachable',reachable)]:
        bins[name]=dict(forms=int(mask.sum()),single=int(earlier[mask].sum()),exact=int(correct[mask].sum()),
                        gained=int((correct&~earlier&mask).sum()),lost=int((~correct&earlier&mask).sum()))
    np.savez_compressed(OUT/'generated.npz',packed=ink,length=n,stopped=stop,correct=correct,logp=logp.cpu().numpy())
    require(all(torch.equal(v.cpu(),state[k]) for k,v in net.state_dict().items()),'Writer changed')
    passed=all(bins[k]['exact']>=.9*bins[k]['forms'] for k in ('all','once_to_five'))
    guard(start)
    write_new(OUT/'result.json',dict(bins=bins,passed=passed,mean_path_nll=float(-logp.mean()),all_stopped=bool(stop.all()),
              states_sha256=sha(OUT/'states.npz'),streams_sha256=sha(OUT/'generated.npz'),seconds=time.monotonic()-start,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(bins,passed,flush=True)

if __name__=='__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

"""Measure what a frozen ink decoder can write from bounded continuous states."""
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import GROUPS, PicturedIdiom, visual_input
from scripts.compare_passage_diversity import OUT as PARENT, ROOT, corpora, guard, load, require, sha, write_new
from scripts.score_paired_physical_answers import physical_score

OUT = ROOT/'artifacts/physical_writer_states_v1'


def main():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    require('4090' in torch.cuda.get_device_name(),'Expected registered GPU')
    _,train=corpora()
    folder=PARENT/'diverse'
    require(sha(folder/'weights.pt')==load(folder/'fit.json')['weights_sha256'],'Changed uniform checkpoint')
    state=torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
    net=PicturedIdiom(state['landmarks']).cuda().eval().requires_grad_(False)
    net.load_state_dict(state)
    OUT.mkdir()
    bank=np.load(PARENT/'pool/pixels.npy')
    pictures=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    features=visual_input(pictures,net.landmarks)
    bits=pictures[:,net.landmarks].float()
    ink=torch.cat([net.motor(b)>0 for b in bits.split(128)])
    feedback=ink[:,net.landmarks].float()
    words,inv,counts=np.unique(train['target'],axis=0,return_inverse=True,return_counts=True)
    inv=inv.reshape(-1)
    target=torch.as_tensor(words.astype(np.int32),device='cuda')
    reachable=(ink==pictures).all(1)[target].all(1).cpu().numpy()
    centers=np.zeros((len(words),192),np.float64)
    with torch.no_grad():
        for i in range(0,len(inv),256):
            x=torch.as_tensor(train['context'][i:i+256].astype(np.int32),device='cuda')
            read=net.read(features[x])
            np.add.at(centers,inv[i:i+len(x)],read.cpu().numpy())
            if i==0:
                a=net.generate(features[x])
                b=net.generate_from_context(read)
                require(all(np.array_equal(l,r) for l,r in zip(a,b)), 'Production entry points disagree')
    centers=torch.tensor(centers/counts[:,None],device='cuda',dtype=torch.float32)
    optimized=[]
    for i in range(0,len(words),128):
        guard(start)
        idx=target[i:i+128]
        param=torch.nn.Parameter(centers[i:i+len(idx)].clamp(-.999999,.999999).atanh())
        opt=torch.optim.Adam([param],lr=.08)
        for _ in range(100):
            score=physical_score(net,param.tanh(),bits[idx],feedback[idx])
            # Sum independent example losses; the step for a state is invariant to batch size.
            loss=-score.sum()/(4*GROUPS+1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            require(torch.isfinite(param.grad).all().item(),'Nonfinite state gradient')
            opt.step()
        optimized.append(param.detach().tanh())
    fitted=torch.cat(optimized)
    np.savez_compressed(OUT/'states.npz',words=words,counts=counts,centroids=centers.cpu().numpy(),
                        fitted=fitted.cpu().numpy(),reachable=reachable)
    known=set(map(bytes,bank[np.unique(train['target'])]))
    statistics={}
    for name,z in [('centroid',centers),('fitted',fitted)]:
        chunks=[net.generate_from_context(b) for b in z.split(128)]
        physical,length,stop=(np.concatenate(v) for v in zip(*chunks))
        used=np.arange(physical.shape[1])[None,:]<length[:,None]
        physical[~used]=0
        hit=(physical[:,:4]==bank[words]).all((1,2))&(length==4)&stop
        with torch.no_grad():
            score=torch.cat([physical_score(net,b,bits[ids],feedback[ids]) for b,ids in zip(z.split(128),target.split(128))])
        np.savez_compressed(OUT/(name+'.npz'),packed=physical,length=length,stopped=stop,correct=hit,logp=score.cpu().numpy())
        bins={}
        for label,mask in [('once_to_five',counts<=5),('six_to_twenty',(counts>=6)&(counts<=20)),('over_twenty',counts>=21),
                           ('reachable',reachable),('all',np.ones_like(hit))]:
            bins[label]=dict(forms=int(mask.sum()),exact=int(hit[mask].sum()))
        statistics[name]=dict(bins=bins,mean_path_nll=float(-score.mean()),emitted=int(used.sum()),
               known_cells=sum(tile.tobytes() in known for tile in physical[used]),stopped=int(stop.sum()),
               outputs_sha256=sha(OUT/(name+'.npz')))
    require(all(torch.equal(v.cpu(),state[k]) for k,v in net.state_dict().items()),'Frozen parameters changed')
    bins=statistics['fitted']['bins']
    passed=all(bins[k]['exact']>=.9*bins[k]['forms'] for k in ('all','once_to_five'))
    guard(start)
    write_new(OUT/'result.json',dict(distinct_targets=len(words),training_passages=len(inv),canonical_reachable=int(reachable.sum()),
             results=statistics,passed=passed,verified_entry_points=256,states_sha256=sha(OUT/'states.npz'),
             parent_sha256=sha(folder/'weights.pt'),program_sha256=sha(Path(__file__)),seconds=time.monotonic()-start,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(statistics,passed,flush=True)

if __name__=='__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

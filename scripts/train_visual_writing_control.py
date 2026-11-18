"""Learn a shared bounded writing state from each supplied physical glyph."""
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn

from ilm.visual_lm.idiom_ink import GROUPS, PicturedIdiom, visual_input
from scripts.measure_physical_writer_states import OUT as SINGLE, PARENT, ROOT, guard, load, require, sha, write_new, physical_score

OUT=ROOT/'artifacts/shared_visual_writing_control_v1'


class GlyphControl(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(112),nn.Linear(112,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,192),nn.Tanh())
        nn.init.zeros_(self[-2].weight)
        nn.init.zeros_(self[-2].bias)


def recombine(words):
    rng=np.random.default_rng(821)
    mixed=np.column_stack([rng.permutation(col) for col in words.T])
    seen=set(map(bytes,words))
    keep=[]
    for row in mixed:
        key=row.tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(row)
    return np.stack(keep)


def main():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    require('4090' in torch.cuda.get_device_name(),'Registered GPU')
    previous=load(ROOT/'artifacts/two_phase_writing_states_v1/result.json')
    require(not previous['passed'] and previous['bins']['all']['exact']==3757,'Missing previous diagnosis')
    original=load(SINGLE/'result.json')
    require(sha(SINGLE/'states.npz')==original['states_sha256'],'Training inventory changed')
    path=PARENT/'diverse/weights.pt'
    require(sha(path)==original['parent_sha256'],'Frozen writer changed')
    state=torch.load(path,weights_only=True,map_location='cpu')
    net=PicturedIdiom(state['landmarks']).cuda().eval().requires_grad_(False)
    net.load_state_dict(state)
    OUT.mkdir()
    with np.load(SINGLE/'states.npz') as a:
        words,counts=a['words'],a['counts']
    mixed=recombine(words)
    bank=np.load(PARENT/'pool/pixels.npy')
    pics=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    features=visual_input(pics,net.landmarks)
    bits=pics[:,net.landmarks].float()
    rendered=torch.cat([net.motor(b)>0 for b in bits.split(128)])
    feedback=rendered[:,net.landmarks].float()
    reachable=(pics==rendered).all(1).cpu().numpy()
    ids=torch.tensor(words.astype(np.int32),device='cuda')
    np.savez_compressed(OUT/'sources.npz',observed=words,counts=counts,mixed=mixed)
    report={}
    for arm in ('shuffled','aligned'):
        begin=time.monotonic()
        rng=np.random.default_rng(820)
        donor_rng=np.random.default_rng(822)
        torch.manual_seed(820)
        controller=GlyphControl().cuda()
        opt=torch.optim.AdamW(controller.parameters(),lr=.001)
        net.causal.train()
        history=[]
        for epoch in range(100):
            order=rng.permutation(len(words))
            # Random cyclic reassignment of a random permutation bijectively uses every input once.
            if arm=='shuffled':
                donors=donor_rng.permutation(len(words))
                assignment=np.empty_like(donors)
                assignment[donors]=np.roll(donors,1)
            total=0.
            for i in range(0,len(order),128):
                rows=order[i:i+128]
                target=ids[rows]
                source=ids[assignment[rows]] if arm=='shuffled' else target
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=controller(features[source])
                    score=physical_score(net,z,bits[target],feedback[target])
                    loss=-score.mean()/(4*GROUPS+1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(controller.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                total+=loss.item()*len(rows)
            history.append(total/len(words))
            if (epoch+1)%10==0: guard(start)
        net.causal.eval()
        controller.eval()
        stage=OUT/arm
        stage.mkdir()
        torch.save(controller.state_dict(),stage/'controller.pt')
        fit_seconds=time.monotonic()-begin
        measures={}
        with torch.inference_mode():
            for split,rows in [('observed',words),('mixed',mixed)]:
                idx=torch.tensor(rows.astype(np.int32),device='cuda')
                chunks=[net.generate_from_context(controller(features[b])) for b in idx.split(128)]
                ink,length,stop=(np.concatenate(v) for v in zip(*chunks))
                used=np.arange(ink.shape[1])[None,:]<length[:,None]
                ink[~used]=0
                hit=(ink[:,:4]==bank[rows]).all((1,2))&(length==4)&stop
                full=reachable[rows].all(1)
                masks={'all':np.ones_like(hit),'reachable':full}
                if split=='observed':
                    masks.update(rare=counts<=5,middle=(counts>5)&(counts<=20),frequent=counts>20)
                bins={k:dict(forms=int(v.sum()),exact=int(hit[v].sum())) for k,v in masks.items()}
                np.savez_compressed(stage/(split+'.npz'),packed=ink,length=length,stopped=stop,exact=hit)
                measures[split]=dict(bins=bins,stopped=int(stop.sum()),emitted=int(used.sum()),
                                     sha256=sha(stage/(split+'.npz')))
        report[arm]=dict(fit_seconds=fit_seconds,history=history,controller_parameters=sum(p.numel() for p in controller.parameters()),
                         weights_sha256=sha(stage/'controller.pt'),measures=measures)
        print(arm,measures,flush=True)
    a=report['aligned']['measures']
    passed=all(a[s]['bins'][k]['exact']>=.9*a[s]['bins'][k]['forms'] for s,k in
               [('observed','all'),('observed','rare'),('mixed','all')]) and all(
               a[s]['bins']['all']['exact']>report['shuffled']['measures'][s]['bins']['all']['exact'] for s in a)
    require(all(torch.equal(v.cpu(),state[k]) for k,v in net.state_dict().items()),'Frozen model changed')
    guard(start)
    write_new(OUT/'result.json',dict(arms=report,passed=passed,source_sha256=sha(OUT/'sources.npz'),
             program_sha256=sha(Path(__file__)),seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

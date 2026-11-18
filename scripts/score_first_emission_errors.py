"""Measure whether suppressing the previous wrong drawing makes the recorded drawing competitive."""
import fcntl

import numpy as np
import torch

from ilm.visual_lm.first_emission import FirstEmission
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import signatures
from scripts.evaluate_sentence_choice import DIRECTORY, INTERACTION, ROOT, SPLITS, ConditionalInk, LocalInk, checkpoint, load, require, roster, sha, sites, write_new
from scripts.train_first_emission import ARMS, WORK
from scripts.train_local_ink import OUT, fields, tensors


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    net=checkpoint('source')
    pen=ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    local=LocalInk.load(DIRECTORY/'onset.npz')
    models={a:FirstEmission.restore(WORK/'fit'/(a+'.safetensors'),sha(INTERACTION),'cuda') for a in ARMS}
    results={}
    for split in SPLITS:
        rows,bank=roster(split)
        pixels=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
        features=visual_input(pixels,net.landmarks)
        bits=pixels[:,net.landmarks]
        sig=signatures(bits.cpu().numpy())
        q=features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
        loc=sites(rows)
        folder=WORK/'evaluation'/split
        first={}
        gold=bank[[rows[i]['y'][j] for i,c,j,k in loc]]
        for a in ('frozen',*ARMS):
            path=folder/(a+'.npz')
            require(sha(path)==load(folder/'report.json')['arms'][a]['pixels_sha256'],'Changed emitted image')
            with np.load(path) as z:
                first[a]=z['pixels'][z['offsets'][:-1]]
        scores={a:[] for a in first}
        for start in range(0,len(loc),64):
            part=loc[start:start+64]
            stop=start+len(part)
            block=[dict(x=rows[i]['x'],y=rows[i]['y'][:j]) for i,c,j,k in part]
            coords=[(b,c,j,k) for b,(i,c,j,k) in enumerate(part)]
            mem,init,hist,_,_=tensors(block,features,bits,net,q,False)
            b,c,j,_=torch.as_tensor(coords,device='cuda').T
            boundary=net.boundary(mem,b,c)
            ctx=fields(block,sig,coords)
            shift=torch.as_tensor(local.bias(ctx),device='cuda')
            cx=torch.as_tensor(ctx.astype('int64'),device='cuda')
            for arm in first:
                state=(boundary+hist[b,j]).tanh() if arm=='frozen' else models[arm](boundary,init,hist[b,j])
                logp=[]
                for pics in (gold,first['frozen'],first[arm]):
                    target=torch.as_tensor(np.unpackbits(pics[start:stop],axis=1),device='cuda')[:,net.landmarks].bool()
                    code=nibbles(target)
                    logits=net.draw.teacher(state,target)+shift
                    logits[:,1:]+=pen.teacher(cx,torch.ones_like(b,dtype=torch.bool),code)
                    logp.append(logits.log_softmax(-1).gather(-1,code[:,:,None]).double().sum((1,2)))
                scores[arm].extend(torch.stack(logp,1).cpu().tolist())
        results[split]={}
        errors=(gold!=first['frozen']).any(1)
        for arm,values in scores.items():
            ref,old,new=np.asarray(values).T
            recovered=(gold==first[arm]).all(1)
            beats=errors & (ref>old+.0001)
            results[split][arm]=dict(parent_errors=int(errors.sum()),gold_beats_old=int(beats.sum()),
                recovered_when_gold_beats_old=int((beats & recovered).sum()),
                gold_loses_new_while_beating_old=int((beats & (ref<new-.0001)).sum()),
                recovered=int((errors & recovered).sum()),
                new_beats_gold_on_errors=int(((~recovered)&(ref<new-.0001)).sum()))
        print(split,results[split],flush=True)
    write_new(WORK/'rival_scores.json',results)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

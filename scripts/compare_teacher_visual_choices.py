"""Score both photographed phrase candidates under authentic and exchanged sources."""
import fcntl
import json

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.first_emission import FirstEmission
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import signatures
from scripts.evaluate_sentence_choice import DIRECTORY, INTERACTION, ROOT, ConditionalInk, LocalInk, PhraseInk, checkpoint, load, require, sha, write_new
from scripts.probe_phrase_teacher import OUT, WORK, prompts
from scripts.train_first_emission import WORK as TRAINED
from scripts.train_local_ink import OUT as BASE, fields, tensors, tiles


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    cases = [c for c in prompts() if not c['reverse']]
    needed = {i for case in cases for i,c,j,k in case['coords']}
    with (WORK/'rows.jsonl').open() as file:
        rows = {i:json.loads(line) for i,line in enumerate(file) if i in needed}
    net = checkpoint('source')
    pen = ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    update = FirstEmission.restore(TRAINED/'fit/preference.safetensors',sha(INTERACTION),'cuda')
    stream = PhraseInk(*(LocalInk.load(DIRECTORY/(n+'.npz')) for n in ('onset','later','end')))
    pixels=torch.as_tensor(np.unpackbits(tiles(),axis=1),device='cuda').bool()
    bits=pixels[:,net.landmarks]
    sig=signatures(bits.cpu().numpy())
    feat=visual_input(pixels,net.landmarks)
    q=feat[load(BASE/'preparation.json')['question_tiles'][0][0]][None]
    margins={'frozen':[],'preference':[]}
    for case in cases:
        side=case['side']
        src,c,_,_=case['coords'][side ^ case['swap']]
        recipient,_,j,_=case['coords'][side]
        prefix=rows[recipient]['y'][:j]
        block=[]
        for other,_,a,z in case['coords']:
            block.append(dict(x=rows[src]['x'],y=prefix+rows[other]['y'][a:z]))
        k=len(block[0]['y'])
        require(k==len(block[1]['y']),'Unequal phrase length')
        steps=[(b,c,t,k) for b in (0,1) for t in range(j,k)]
        ctx=fields(block,sig,steps)
        bb,cc,t,kk=torch.as_tensor(steps,device='cuda').T
        mem,init,hist,y,_=tensors(block,feat,bits,net,q,False)
        at=net.boundary(mem,bb,cc)
        ref=bits[y[bb,t]]
        code=nibbles(ref)
        first=t==j
        shift=np.empty((len(t),12,16),np.float32)
        for mask,table in ((first,stream.onset),(~first,stream.later)):
            mask=mask.cpu().numpy()
            if mask.any():
                shift[mask]=table.bias(ctx[mask])
        bias=torch.as_tensor(shift,device='cuda')
        bias[:,1:]+=pen.teacher(torch.as_tensor(ctx.astype('int64'),device='cuda'),first,code)
        end=net.end((at+hist[bb,t+1]).tanh())[:,0]+torch.as_tensor(stream.update(ctx,ref.cpu().numpy()),device='cuda')
        stopping=-F.binary_cross_entropy_with_logits(end,(t+1==kk).float(),reduction='none').double()
        for arm in margins:
            state=(at+hist[bb,t]).tanh()
            if arm=='preference':
                state[first]=update(at[first],init[bb[first]],hist[bb[first],t[first]])
            logits=net.draw.teacher(state,ref)+bias
            drawing=logits.log_softmax(-1).gather(-1,code[:,:,None]).double().sum((1,2))
            score=(drawing+stopping).view(2,-1).sum(1)
            margins[arm].append(float(score[side]-score[1-side]))
    report={}
    for arm,scores in margins.items():
        array=np.asarray(scores).reshape(32,2,2)
        hits=array>.0001
        report[arm]=dict(authentic_exchanged_correct=hits.sum((0,1)).tolist(),both_recipients=hits.all(1).sum(0).tolist(),
                         ties=(np.abs(array)<=.0001).sum((0,1)).tolist(),margins_nats=scores)
    write_new(OUT/'visual.json',report)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

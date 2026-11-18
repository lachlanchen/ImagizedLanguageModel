"""Learn how many insertions to allow before advancing the source cursor."""
import fcntl
import time

import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F

from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from ilm.visual_lm.visual_edit_writer import COPY,DROP,INSERT,REPLACE,visual_input
from scripts.train_visual_edit_translation import OUT,ROOT,bank,guard,inputs,load,padded,require,sha,write_new


def fields(rows,prompts):
    xs,qs,ys=[],[],[]
    for i,r in enumerate(rows):
        x = r['x']
        c = np.zeros(len(x)+1,np.int32)
        cursor = 0
        for a in r['a']:
            if a==INSERT:
                c[cursor]+=1
            if a in (COPY,REPLACE,DROP):
                cursor+=1
        for quote in range(1+int(i%4==0)):
            xs.append(x)
            qs.append(prompts[quote][(i//4 if quote else i)%2])
            ys.append(c*0 if quote else c)
    x,n=padded(xs)
    q,m=padded(qs)
    y,_=padded(ys)
    return x,n,q,m,y


def run():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.manual_seed(5003)
    torch.set_float32_matmul_precision('high')
    out=OUT/'fertility'
    out.mkdir()
    prompts=load(OUT/'preparation.json')['question_tiles']
    x,n,q,m,y=fields(inputs('train'),prompts)
    all_counts=y[np.arange(y.shape[1])[None]<=n[:,None]]
    fixed=max(1,int(np.quantile(all_counts,.99,method='higher')))
    maximum=int(all_counts.max())
    parent=OUT/'motor_refined/editor.safetensors'
    state=load_file(str(parent))
    model=FertilityEditor(state['landmarks'],maximum).cuda().eval()
    missing,extra=model.load_state_dict(state,strict=False)
    require(not extra and all(s.startswith('fertility.') for s in missing),'Unexpected inherited state')
    model.requires_grad_(False)
    model.fertility.requires_grad_(True)
    ink=torch.from_numpy(np.unpackbits(bank(),axis=1)).cuda().bool()
    features=visual_input(ink,model.landmarks)
    dx,dn,dq,dm,dy=[torch.as_tensor(v,device='cuda') for v in (x,n,q,m,y)]
    rng=np.random.default_rng(5003)
    optimizer=torch.optim.AdamW(model.fertility.parameters(),lr=.001)
    history=[]
    for epoch in range(6):
        accum,count=0.,0
        for indices in np.array_split(rng.permutation(len(n)),(len(n)+127)//128):
            ix=torch.as_tensor(indices,device='cuda')
            width=int(n[indices].max())
            v=torch.arange(width,device='cuda')[None]<dn[ix,None]
            qvalid=torch.arange(q.shape[1],device='cuda')[None]<dm[ix,None]
            with torch.no_grad():
                mem,g=model.encode(features[dx[ix,:width]],v,features[dq[ix]],qvalid)
            logits=model.count_logits(mem,g)
            valid=torch.arange(width+1,device='cuda')[None]<=dn[ix,None]
            targets=dy[ix,:width+1][valid].long()
            loss=F.cross_entropy(logits[valid],targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            accum+=loss.item()*len(targets)
            count+=len(targets)
        history.append(accum/count)
        guard(start)
    result={k:v.cpu().contiguous() for k,v in model.state_dict().items()}
    require(all(torch.equal(result[k],v) for k,v in state.items()),'Frozen language or motor changed')
    save_file(result,str(out/'editor.safetensors'))
    write_new(out/'report.json',dict(history=history,train_positions=len(all_counts),maximum=maximum,
        constant_control=fixed,trainables=sum(p.numel() for p in model.fertility.parameters()),
        weights_sha256=sha(out/'editor.safetensors'),parent_sha256=sha(parent),seconds=time.monotonic()-start))
    print('Count training complete',maximum,fixed,history,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

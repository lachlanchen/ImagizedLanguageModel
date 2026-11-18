"""Improve only the renderer of the frozen sentence editor on its original training shapes."""
import fcntl
import time

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from scripts.evaluate_visual_edit_translation import OUT, ROOT, bank, guard, load, require, restore, sha, write_new


def main():
    began=time.monotonic()
    guard(began)
    torch.set_num_threads(2)
    torch.manual_seed(5129)
    torch.set_float32_matmul_precision('high')
    folder=OUT/'motor_refined'
    folder.mkdir()
    parent=OUT/'fit/editor.safetensors'
    original=load(OUT/'fit/report.json')
    require(sha(parent)==original['weights_sha256'],'Original model changed')
    net=restore(parent,'cuda').requires_grad_(False)
    unchanged={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('motor.')}
    ids=np.load(OUT/'train_shapes.npy')
    images=torch.as_tensor(np.unpackbits(bank()[ids],axis=1),device='cuda').bool()
    measurements=images[:,net.landmarks].float()
    net.motor.requires_grad_(True)
    opt=torch.optim.AdamW(net.motor.parameters(),lr=.0005,weight_decay=.0001)
    history=[]
    for step in range(24001):
        if step%400==0:
            with torch.no_grad():
                actual=torch.cat([net.motor(v)>0 for v in measurements.split(256)])
                hard=torch.nonzero((actual!=images).any(1)).flatten()
            history.append(dict(step=step,exact=len(ids)-len(hard)))
            guard(began)
            if step==0:
                require(history[-1]['exact']==original['motor']['known_exact'],'Changed initial rendering')
            if not len(hard) or step==24000:
                break
        uniform=torch.randint(len(ids),(192,),device='cuda')
        difficult=hard[torch.randint(len(hard),(64,),device='cuda')]
        index=torch.cat((uniform,difficult))
        logits=net.motor(measurements[index])
        loss=F.binary_cross_entropy_with_logits(logits,images[index].float(),reduction='none')
        objective=loss.mean()+loss.topk(16,dim=1).values.mean()
        opt.zero_grad(set_to_none=True)
        objective.backward()
        opt.step()
    state={k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()}
    require(all(torch.equal(v,state[k]) for k,v in unchanged.items()),'Non-motor tensor changed')
    save_file(state,str(folder/'editor.safetensors'))
    write_new(folder/'report.json',dict(history=history,shapes=len(ids),steps=step,parent_sha256=sha(parent),
        checkpoint_sha256=sha(folder/'editor.safetensors'),frozen_tensors=len(unchanged),seconds=time.monotonic()-began,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    print(history[0],history[-1],flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

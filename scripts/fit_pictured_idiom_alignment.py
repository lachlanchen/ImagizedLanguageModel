"""Add whole-answer physical supervision to the unchanged autonomous writing route."""
from collections import Counter,defaultdict
import fcntl
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS,PicturedIdiom,visual_input
from scripts import fit_pictured_idioms as first
from scripts.diagnose_visual_context_gradients import load,require,sha,write_new

ROOT,DATA,CONTROL=first.ROOT,first.DATA,first.OUT
OUT=ROOT/'artifacts/pictured_idiom_alignment_v1'
PROTOCOL=ROOT/'references/pictured_idiom_alignment_protocol.md'


def phrase_loss(context,answer_embedding,target):
    logits=F.normalize(context.float()) @ F.normalize(answer_embedding.float()).T /.07
    same=(target[:,None,:]==target[None,:,:]).all(-1)
    return (logits.logsumexp(1)-logits.masked_fill(~same,-torch.inf).logsumexp(1)).mean()


def local_keys(row):
    return (row[20:28].tobytes(),row[22:26].tobytes(),row[23:25].tobytes(),
            row[23:24].tobytes(),row[24:25].tobytes())


def local_counts(train,dev):
    counts=[defaultdict(Counter) for _ in range(5)]
    freq=Counter()
    for x,y in zip(train['context'],train['target']):
        word=y.tobytes();freq[word]+=1
        for table,key in zip(counts,local_keys(x)):table[key][word]+=1
    maps=[{k:v.most_common(1)[0][0] for k,v in t.items()} for t in counts]
    default=freq.most_common(1)[0][0]
    return np.array([next((m[k] for m,k in zip(maps,local_keys(x)) if k in m),default)==y.tobytes()
                     for x,y in zip(dev['context'],dev['target'])])


@torch.inference_mode()
def reading_and_controls(model,aux,features,pixels,train,dev):
    options=torch.as_tensor(np.unique(train['target'],axis=0).astype(np.int32),device='cuda')
    keys=F.normalize(aux(features[options].flatten(1)).float())
    chosen=[]
    for chunk in np.array_split(dev['context'],(len(dev['context'])+255)//256):
        state=model.read(features[torch.as_tensor(chunk.astype(np.int32),device='cuda')])
        chosen.append(options[(F.normalize(state.float())@keys.T).argmax(1)].cpu().numpy())
    selected=np.concatenate(chosen)
    local=local_counts(train,dev)
    condition={}
    for name,path in [('control',CONTROL),('aligned',OUT)]:
        with np.load(path/'generated.npz') as a:
            exact=(a['packed'][:,:4]==pixels[dev['target']]).all((1,2)) & a['stopped'] & (a['length']==4)
            condition[name]=exact
    grp,idx=np.unique(dev['group'],return_inverse=True);sizes=np.bincount(idx)
    residual=np.bincount(idx,weights=condition['aligned'].astype(int)-condition['control'].astype(int))
    rng=np.random.default_rng(419);interval=[]
    for _ in range(2000):
        s=rng.integers(len(grp),size=len(grp));interval.append(residual[s].sum()/sizes[s].sum())
    np.savez_compressed(OUT/'diagnostic_choices.npz',supplied_inventory_selection=selected,local_exact=local)
    return dict(options=len(options),supplied_inventory_exact=int((selected==dev['target']).all(1).sum()),
                supplied_inventory_exchanged_exact=int((np.roll(selected,-1,axis=0)==dev['target']).all(1).sum()),
                local_counts_exact=int(local.sum()),aligned_only=int((condition['aligned'] & ~condition['control']).sum()),
                control_only=int((condition['control'] & ~condition['aligned']).sum()),
                paired_improvement_95=np.quantile(interval,[.025,.975]).tolist())


def main():
    start=time.monotonic();first.guard(start)
    require(torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(),'Registered GPU')
    old=load(CONTROL/'result.json')
    for name,h in old['bindings'].items():require(sha(ROOT/name)==h,'Changed original source or output')
    coords_state=torch.load(CONTROL/'weights.pt',map_location='cpu',weights_only=True)
    OUT.mkdir(exist_ok=False)
    first.OUT=OUT  # The inherited archive writer is confined to this trial in this process.
    torch.set_num_threads(2);torch.manual_seed(419);torch.set_float32_matmul_precision('high')
    rng=np.random.default_rng(419)
    model=PicturedIdiom(coords_state['landmarks']).cuda()
    model.motor.load_state_dict({k.removeprefix('motor.'):v for k,v in coords_state.items() if k.startswith('motor.')})
    model.motor.requires_grad_(False)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(9419)
        aux=nn.Sequential(nn.Linear(4*112,384),nn.GELU(),nn.Linear(384,192)).cuda()
    opt=torch.optim.AdamW([p for module in (model,aux) for p in module.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    # Advance the data RNG exactly as the preceding stroke-only warmup.
    m=old['physical']
    for _ in range(m['steps']):rng.integers(m['distinct_training_shapes'],size=256)
    with np.load(DATA/'train.npz') as f:train={k:f[k] for k in f.files}
    px=np.load(DATA/'pixels.npy')
    images=torch.as_tensor(np.unpackbits(px,axis=1),device='cuda',dtype=torch.bool)
    features=visual_input(images,model.landmarks); measured=images[:,model.landmarks].float()
    x=torch.as_tensor(train['context'].astype(np.int32),device='cuda');y=torch.as_tensor(train['target'].astype(np.int32),device='cuda')
    # Reuse the same reader forward with the exact dropout realization used for writing.
    saved=[]
    handle=model.initial.register_forward_hook(lambda module,inputs,out:saved.append(out))
    fit_start=time.monotonic();metrics=[]
    for e in range(12):
        model.train();aux.train();total=np.zeros(2);count=0
        order=rng.permutation(len(y))
        for i in range(0,len(order),192):
            if i%3072==0:first.guard(start)
            idx=order[i:i+192];target=y[idx]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,labels=model.teacher_logits(features[x[idx]],measured[target])
                anchor=saved.pop().tanh()
                stop=logits[:,:,16:17].masked_fill((torch.arange(logits.size(1),device='cuda')%GROUPS!=0)[None,:,None],-torch.inf)
                ce=F.cross_entropy(torch.cat([logits[:,:,:16],stop],-1).flatten(0,1).float(),labels.flatten())
                cl=phrase_loss(anchor,aux(features[target].flatten(1)),target)
                loss=ce+.5*cl
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters())+list(aux.parameters()) if p.requires_grad],
                                           1.,error_if_nonfinite=True)
            opt.step();total+=np.array([float(ce.detach()),float(cl.detach())])*len(idx);count+=len(idx)
        metrics.append((total/count).tolist())
        print('epoch',e+1,'pixel and contrast losses',metrics[-1],flush=True)
    handle.remove();fit=time.monotonic()-fit_start
    torch.save(model.state_dict(),OUT/'weights.pt');torch.save(aux.state_dict(),OUT/'training_only_aux.pt')
    restored=PicturedIdiom(coords_state['landmarks']).cuda()
    restored.load_state_dict(torch.load(OUT/'weights.pt',weights_only=True))
    with np.load(DATA/'validation.npz') as f:dev={k:f[k] for k in f.files}
    result=first.compare(restored,features,images,train,dev,px,np.unique(train['target']),start)
    reading=reading_and_controls(restored.eval(),aux.eval(),features,px,train,dev)
    result.update(alignment=reading,aligned_pass=bool(result['passed'] and reading['paired_improvement_95'][0]>0),
                  fit_seconds=fit,seconds=time.monotonic()-start,epoch_losses=metrics,
                  inference_parameters=sum(p.numel() for p in restored.parameters()),aux_parameters=sum(p.numel() for p in aux.parameters()),
                  peak_vram=torch.cuda.max_memory_allocated(),host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),PROTOCOL,CONTROL/'result.json',
                   OUT/'weights.pt',OUT/'training_only_aux.pt',OUT/'diagnostic_choices.npz',OUT/'generated.npz']})
    first.guard(start);write_new(OUT/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()

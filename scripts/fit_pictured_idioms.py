"""Fit and score candidate-free pictured Chinese cloze answers on one GPU."""
from collections import Counter
import fcntl
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, PicturedIdiom, choose_landmarks, visual_input
from scripts.diagnose_visual_context_gradients import load,require,sha,write_new
from scripts.prepare_pictured_idioms import OUT as DATA
from scripts.prepare_visual_sequence_corpus import ROOT
from scripts.train_visual_extractive_reader import save_page

OUT=ROOT/'artifacts/pictured_idioms_v1'
PROTOCOL=ROOT/'references/pictured_idiom_protocol.md'


def guard(start):
    available=next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    require(int(available.split()[1])/1024**2>=24 and shutil.disk_usage(OUT.parent).free>=48*1024**3,'Shared machine reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2 and torch.cuda.max_memory_allocated()<8*1024**3,'Process or GPU limit')
    require(time.monotonic()-start<2100,'Time limit reached')


def learn_motor(model, images, rng, start):
    # Distinct target shapes receive equal weight, independently of prose frequency.
    bits=images[:,model.landmarks]
    opt=torch.optim.AdamW(model.motor.parameters(),lr=.002,weight_decay=.0001)
    for step in range(1,4001):
        idx=torch.as_tensor(rng.integers(len(images),size=256),device=images.device)
        logits=model.motor(bits[idx])
        errors=F.binary_cross_entropy_with_logits(logits,images[idx].float(),reduction='none')
        loss=errors.mean()+.5*errors.topk(16,dim=1).values.mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%250==0:
            guard(start)
            with torch.no_grad():
                generated=torch.cat([model.motor(b)>0 for b in bits.split(256)])
                exact=(generated==images).all(-1).sum().item()
            print('motor',step,exact,len(images),flush=True)
            if exact==len(images):
                break
    model.motor.requires_grad_(False)
    return dict(steps=step,known_exact=int(exact),distinct_training_shapes=len(images))


def compare(model, features, images, train, dev, pixels, train_types, start):
    x,y=dev['context'],dev['target']
    answers=[];length=[];stops=[]
    model.eval()
    for i in range(0,len(y),128):
        if i%2048==0: guard(start)
        # Only input features and the trained weights reach free generation.
        out,n,stop=model.generate(features[torch.as_tensor(x[i:i+128].astype(np.int32),device='cuda')])
        answers.append(out);length.extend(n);stops.extend(stop)
    answers=np.concatenate(answers);length=np.asarray(length);stops=np.asarray(stops)
    # Omit padding beyond learned termination from the archived physical answer stream.
    used=np.arange(8)[None,:]<length[:,None];answers[~used]=0
    np.savez_compressed(OUT/'generated.npz',packed=answers,length=length,stopped=stops)
    truth=pixels[y]
    same=(answers[:,:4]==truth).all(-1)
    matches=(stops & (length==4) & same.all(1))
    absent=~(y[:,:,None]==x[:,None,:]).any(-1)
    emitted=used[:,:4]
    other=np.roll(np.arange(len(y)),-1)
    swapped=stops[other] & (length[other]==4) & (answers[other,:4]==truth).all((1,2))
    with torch.no_grad():
        glyphs=torch.as_tensor(np.unique(y),device='cuda').long()
        motor_images=torch.cat([model.motor(images[ids][:,model.landmarks])>0 for ids in glyphs.split(256)])
        correct=(motor_images==images[glyphs]).all(-1).cpu().numpy()
    ids=glyphs.cpu().numpy()
    known=np.isin(ids,train_types)
    form_oracle=dict(known_shapes=int(known.sum()),known_exact=int(correct[known].sum()),
                     unseen_shapes=int((~known).sum()),unseen_exact=int(correct[~known].sum()))
    mode=Counter(map(tuple,train['target'])).most_common(1)[0][0]
    freq=(y==np.array(mode)[None]).all(1)
    groups,inv=np.unique(dev['group'],return_inverse=True)
    delta=np.bincount(inv,weights=matches.astype(int)-swapped.astype(int))
    sizes=np.bincount(inv);rng=np.random.default_rng(419);interval=[]
    for _ in range(2000):
        sample=rng.integers(len(groups),size=len(groups))
        interval.append(delta[sample].sum()/sizes[sample].sum())
    examples={}
    for name,mask in [('first',np.ones(len(y),bool)),('exact_absent',matches & absent.any(1)),('first_miss',~matches)]:
        if not mask.any():continue
        j=int(np.flatnonzero(mask)[0]);examples[name]=dict(index=j,tag=int(dev['tag'][j]),length=int(length[j]))
        for tile,seq in [('before',pixels[x[j,:24]]),('after',pixels[x[j,24:]]),('reference',truth[j]),('answer',answers[j,:length[j]])]:
            if len(seq): save_page(OUT/f'{name}_{tile}.png',seq)
    # Correct-sequence and readable-repertoire measures answer different questions.
    inventory=set(a.tobytes() for a in pixels[train_types]); inventory_hits=sum(
        cell.tobytes() in inventory for cell in answers[used])
    distinct=len(set(map(tuple,y[matches & absent.any(1)])))
    return dict(examples=len(y),stopped=int(stops.sum()),lengths={str(n):int((length==n).sum()) for n in np.unique(length)},
                complete_exact=int(matches.sum()),swapped_exact=int(swapped.sum()),frequency_baseline_exact=int(freq.sum()),
                unique_correct=int(len(set(map(tuple,y[matches])))),absent_cases=int(absent.any(1).sum()),
                correct_with_absent=int((matches & absent.any(1)).sum()),distinct_correct_with_absent=distinct,
                aligned_absent_cells=int(absent.sum()),aligned_absent_exact=int((absent & emitted & same).sum()),
                emitted_cells=int(used.sum()),training_target_shape_hits=int(inventory_hits),
                group_bootstrap_normal_minus_swapped_95=np.quantile(interval,[.025,.975]).tolist(),
                oracle_motor=form_oracle,examples_by_type=examples,
                passed=bool(distinct>=20 and np.quantile(interval,.025)>0 and correct[known].mean()>=.99))


def main():
    start=time.monotonic();guard(start)
    require(torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(),'Registered GPU')
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(419);rng=np.random.default_rng(419)
    torch.set_float32_matmul_precision('high')
    report=load(DATA/'report.json');require(sha(DATA/'pixels.npy')==report['pixels_sha256'],'Source pixels changed')
    require(all(sha(DATA/(s+'.npz'))==r['sha256'] for s,r in report['splits'].items()),'Prepared corpus changed')
    with np.load(DATA/'train.npz',allow_pickle=False) as archive:train={k:archive[k] for k in archive.files}
    pixels=np.load(DATA/'pixels.npy',allow_pickle=False)
    train_types=np.unique(train['target']).astype(np.int64)
    masks=np.unpackbits(pixels,axis=1)
    coords=choose_landmarks(masks[train_types])
    model=PicturedIdiom(coords).cuda()
    images=torch.as_tensor(masks,device='cuda',dtype=torch.bool)
    physical=learn_motor(model,images[train_types],rng,start)
    physical['distinct_codes']=len(np.unique(masks[train_types][:,coords],axis=0))
    write_new(OUT/'motor.json',physical)
    features=visual_input(images,model.landmarks)
    target_bits=images[:,model.landmarks].float()
    x=torch.as_tensor(train['context'].astype(np.int32),device='cuda')
    y=torch.as_tensor(train['target'].astype(np.int32),device='cuda')
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001,weight_decay=.01)
    losses=[];fit_start=time.monotonic()
    for epoch in range(12):
        model.train();total=0.;positions=0
        order=rng.permutation(len(y))
        for i in range(0,len(order),192):
            if i%3072==0:guard(start)
            idx=order[i:i+192]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,targets=model.teacher_logits(features[x[idx]],target_bits[y[idx]])
                at_boundary=torch.arange(logits.size(1),device='cuda')%GROUPS==0
                stop=logits[:,:,16:17].masked_fill(~at_boundary[None,:,None],-torch.inf)
                logits=torch.cat([logits[:,:,:16],stop],-1)
                loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())*targets.numel();positions+=targets.numel()
        losses.append(total/positions)
        print('epoch',epoch+1,'loss',losses[-1],'elapsed',round(time.monotonic()-start,1),flush=True)
    fit=time.monotonic()-fit_start
    torch.save(model.state_dict(),OUT/'weights.pt')
    final=PicturedIdiom(coords).cuda()
    final.load_state_dict(torch.load(OUT/'weights.pt',map_location='cuda',weights_only=True))
    with np.load(DATA/'validation.npz',allow_pickle=False) as archive:dev={k:archive[k] for k in archive.files}
    result=compare(final,features,images,train,dev,pixels,train_types,start)
    result.update(physical=physical,parameters=sum(p.numel() for p in final.parameters()),fit_seconds=fit,
        seconds=time.monotonic()-start,peak_vram=torch.cuda.max_memory_allocated(),peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        epoch_losses=losses,device=torch.cuda.get_device_name(),bindings={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__).resolve(),ROOT/'ilm/visual_lm/idiom_ink.py',PROTOCOL,DATA/'report.json',OUT/'weights.pt',OUT/'generated.npz')})
    guard(start);write_new(OUT/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

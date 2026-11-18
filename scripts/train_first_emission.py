"""Fit a first-glyph visual writer using genuine preceding free model errors."""
import copy
import fcntl
import hashlib
import json
import resource
import time

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.first_emission import FirstEmission
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import signatures
from scripts.prepare_first_emission_training import DEST as WORK
from scripts.train_sentence_choice import (DIRECTORY, INTERACTION, ROOT, WORK as PAIRED, ConditionalInk, LocalInk,
    OUT, checkpoint, fields, load, require, resources, sha, tensors, tiles, write_new)

ARMS = ('likelihood', 'preference')
GRID = (0., .25, .5, .75, 1.)
SEED = 6811


def joint_score(net, state, bits, prior):
    labels = nibbles(bits)
    return (net.draw.teacher(state,bits) + prior).log_softmax(-1).gather(-1,labels[:,:,None]).sum((1,2))


def first_loss(reference, emitted=None, different=None):
    loss = -reference.mean()
    if emitted is not None:
        # Empty masks should leave the genuine likelihood gradient intact.
        extra = F.softplus(emitted-reference)*different
        loss = loss + extra.sum()/different.sum().clamp_min(1)
    return loss


@torch.no_grad()
def observations(net, pen, onset_table, start):
    spec = load(WORK/'preparation.json')
    require(sha(WORK/'first_images.npz')==spec['first_images_sha256'] and
            sha(PAIRED/'sites.npz')==spec['source_pairs_sha256'] and sha(PAIRED/'rows.jsonl')==spec['rows_sha256']
            and sha(INTERACTION)==spec['parent_interaction_sha256'], 'Bound parent images or data changed')
    with np.load(PAIRED/'sites.npz') as z:
        coords = z['locations'][z['pairs'].ravel()]
        held = np.repeat(z['calibration'],2)
    with np.load(WORK/'first_images.npz') as z:
        require(np.array_equal(held,z['calibration']), 'Calibration sites changed')
        emitted = torch.as_tensor(np.unpackbits(z['pixels'],axis=1),device='cuda')[:,net.landmarks].bool()
        different = torch.as_tensor(~z['same_signature'],device='cuda')
    with (PAIRED/'rows.jsonl').open() as file:
        rows = [(r['x'],r['y']) for r in map(json.loads,file)]
    bank = tiles()
    images = torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    bits = images[:,net.landmarks]
    features = visual_input(images,net.landmarks)
    sig = signatures(bits.cpu().numpy())
    question = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    count = len(coords)
    states = torch.empty((count,3,128),device='cuda')
    priors = torch.empty((count,2,12,16),device='cuda')
    targets = torch.empty(count,device='cuda',dtype=torch.long)
    contexts = torch.empty((count,4),device='cuda',dtype=torch.long)
    for first in range(0,count,128):
        part = coords[first:first+128]
        window = slice(first,first+len(part))
        block = [dict(x=rows[i][0],y=rows[i][1][:j]) for i,c,j,k in part]
        memory,init,hist,_,_=tensors(block,features,bits,net,question,False)
        axis=torch.arange(len(part),device='cuda')
        _,cursor,pos,_=torch.as_tensor(part,device='cuda').long().T
        states[window]=torch.stack((net.boundary(memory,axis,cursor),init,hist[axis,pos]),1)
        targets[window]=torch.as_tensor([rows[i][1][j] for i,c,j,k in part],device='cuda')
        local=fields(block,sig,[(b,c,j,k) for b,(_,c,j,k) in enumerate(part)])
        ctx=torch.as_tensor(local.astype('int64'),device='cuda')
        contexts[window]=ctx
        shift=torch.as_tensor(onset_table.bias(local),device='cuda')
        for side,ref in enumerate((bits[targets[window]],emitted[window])):
            prior=priors[window,side]
            prior.copy_(shift)
            prior[:,1:]+=pen.teacher(ctx,torch.ones_like(axis,dtype=torch.bool),nibbles(ref))
        if first%8192==0:
            resources(start)
    require(torch.equal(different,(emitted!=bits[targets]).any(1)), 'Mined first signature changed')
    return states,priors,contexts,targets,bits,images,emitted,different,held


def run():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    resources(start)
    dest=WORK/'fit'
    dest.mkdir()
    net=checkpoint('source').requires_grad_(False)
    pen=ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda').requires_grad_(False)
    table=LocalInk.load(DIRECTORY/'onset.npz')
    states,priors,contexts,targets,bits,images,emitted,different,held=observations(net,pen,table,start)
    prepared=time.monotonic()-start
    base=FirstEmission().cuda()
    arms={a:copy.deepcopy(base) for a in ARMS}
    opts={a:torch.optim.AdamW(m.parameters(),lr=.0006,weight_decay=.01) for a,m in arms.items()}
    train,cal=np.flatnonzero(~held),np.flatnonzero(held)
    print('Prepared sites',len(train),len(cal),flush=True)
    history={a:[] for a in ARMS}
    cost=dict.fromkeys(ARMS,0.)
    schedule=hashlib.sha256()
    rng=np.random.default_rng(SEED)
    for epoch in range(8):
        order=rng.permutation(train)
        schedule.update(order.astype('<i4').tobytes())
        total=dict.fromkeys(ARMS,0.)
        for first in range(0,len(order),512):
            idx=torch.as_tensor(order[first:first+512],device='cuda')
            for arm,writer in arms.items():
                began=time.monotonic()
                state=writer(*states[idx].unbind(1))
                own=joint_score(net,state,bits[targets[idx]],priors[idx,0])
                rival=joint_score(net,state,emitted[idx],priors[idx,1]) if arm=='preference' else None
                loss=first_loss(own,rival,different[idx])
                opts[arm].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(writer.parameters(),2.,error_if_nonfinite=True)
                opts[arm].step()
                total[arm]+=loss.detach().item()*len(idx)
                cost[arm]+=time.monotonic()-began
        for arm in ARMS:
            history[arm].append(total[arm]/len(train))
        resources(start)
        print('epoch',epoch+1,{a:history[a][-1] for a in ARMS},flush=True)
    results={}
    with torch.no_grad():
        for arm,writer in arms.items():
            counts,nll=[],[]
            for scale in GRID:
                writer.scale.fill_(scale)
                correct=0
                loss=0.
                for i in range(0,len(cal),256):
                    idx=torch.as_tensor(cal[i:i+256],device='cuda')
                    state=writer(*states[idx].unbind(1))
                    ctx=contexts[idx]
                    local=ctx.cpu().numpy().astype('uint64')
                    choice=net.draw.choose(state,pattern_bias=torch.as_tensor(table.bias(local),device='cuda'),
                        transitions=pen.transitions(ctx,torch.ones_like(idx,dtype=torch.bool)))
                    picture=net.motor(choice)>0
                    picture[:,net.landmarks]=choice.bool()
                    correct+=int((picture==images[targets[idx]]).all(1).sum())
                    loss-=joint_score(net,state,bits[targets[idx]],priors[idx,0]).double().sum().item()
                counts.append(correct)
                nll.append(loss/len(cal))
            best=max(range(len(GRID)),key=lambda i:(counts[i],-nll[i]))
            writer.scale.fill_(GRID[best])
            path=dest/(arm+'.safetensors')
            save_file({k:v.detach().cpu() for k,v in writer.state_dict().items()},str(path),
                      metadata=dict(format='ilm-first-emission-v1',interaction_sha256=sha(INTERACTION)))
            results[arm]=dict(calibration_exact=counts,calibration_nats=nll,selected_scale=GRID[best],
                              epoch_loss=history[arm],optimizer_seconds=cost[arm],sha256=sha(path))
            print(arm,results[arm],flush=True)
    resources(start)
    write_new(dest/'fit.json',dict(arms=results,scales=GRID,epochs=8,training_sites=len(train),calibration_sites=len(cal),
        schedule_sha256=schedule.hexdigest(),parameters=sum(p.numel() for p in base.parameters()),
        preparation_seconds=prepared,seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,preparation=load(WORK/'preparation.json')))


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

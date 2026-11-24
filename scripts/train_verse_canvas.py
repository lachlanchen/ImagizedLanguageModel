"""Fit matched image- and byte-observation line writers on the same masked pages."""
import argparse
import hashlib
import json
import math
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.verse_canvas import VerseCanvas
from scripts.prepare_verse_canvas import corpus, PUB, MODEL
from scripts.prepare_dialogue_ink import dump, sha
from scripts.measure_writer_energy import snapshot,consumed
from scripts.train_span_choice import gpu_owner

SEED=350371
STEPS=10000
BATCH=96


def banks(data, mode, device):
    if mode=='byte':
        values=observations({'bytes':data['bytes']},'byte')
        return torch.tensor(values,device=device)[None].expand(3,-1,-1)
    values=np.stack([observations({'tiles':packed},'ink') for packed in data['tiles']])
    return torch.as_tensor(values,device=device)


def train(meta,data,mode,binding=False):
    name = mode+'-bound' if binding else mode
    dest=MODEL/f'{name}.safetensors'
    receipt=PUB/f'{name}-fit.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['weights']!=sha(dest) or old['data']!=meta['corpus']:
            raise ValueError('Incompatible fitted writer')
        return
    rng=np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    t0=monotonic()
    energy=snapshot()
    bank=banks(data,mode,'cuda')
    rows=data['train']
    by_size={n:np.flatnonzero(rows[:,1]==n) for n in (5,7)}
    table=torch.as_tensor(rows,device='cuda')
    writer=VerseCanvas(meta['train_shapes']).cuda()
    optimizer=torch.optim.AdamW(writer.parameters(),lr=.001,weight_decay=.03,fused=True)
    schedule=hashlib.sha256()
    touched=np.zeros(len(table),bool)
    num_hidden=0
    trace=[]
    for step in range(1,STEPS+1):
        length=5 if rng.random()<len(by_size[5])/len(rows) else 7
        selected=rng.choice(by_size[length],BATCH)
        touched[selected]=True
        size=rng.integers(1,length+1,size=BATCH)
        size[rng.random(BATCH)<.25]=length
        mask=rng.random((BATCH,length)).argsort(axis=1)<size[:,None]
        num_hidden+=int(size.sum())
        face=rng.integers(2,size=BATCH)
        for arr in (selected,mask,face):
            schedule.update(arr.tobytes())
        batch=table[torch.as_tensor(selected,device='cuda')].long()
        source=batch[:,2:2+length]
        truth=batch[:,9:9+length]
        masked=torch.as_tensor(mask,device='cuda')
        srcface=torch.as_tensor(face,device='cuda')[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            first_ink = bank[srcface,source]
            visible_ink = bank[0,truth.masked_fill(masked,0)]
            logits=writer(first_ink,visible_ink,~masked)
            labels=(truth-1).masked_fill(~masked,-100)
            loss=F.cross_entropy(logits.flatten(0,1),labels.flatten())
            if binding:
                # The same output rows encode observed and predicted canonical bitmaps.
                seen = torch.cat([writer.source(first_ink),writer.target(visible_ink)],dim=1)
                canonical = torch.cat([source-1,(truth-1).masked_fill(masked,-100)],dim=1)
                visible_logits=writer.ink(F.layer_norm(seen,(seen.shape[-1],))).float()
                loss=loss+.5*F.cross_entropy(visible_logits.flatten(0,1),canonical.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(writer.parameters(),1.)
        rate=min(1.,step/400)*(.1+.9*(1+math.cos(math.pi*step/STEPS))/2)
        optimizer.param_groups[0]['lr']=.001*rate
        optimizer.step()
        if step==1 or step%250==0:
            scalar=loss.detach().item()
            sec=monotonic()-t0
            peak=torch.cuda.max_memory_allocated()
            print(name,step,round(scalar,4),round(sec,2),peak,flush=True)
            if not math.isfinite(scalar) or sec>1200 or peak>8*2**30:
                raise RuntimeError('Numeric or resource limit exceeded')
            trace.append([step,scalar,sec])
    torch.cuda.synchronize()
    seconds=monotonic()-t0
    cost=consumed(energy,snapshot())
    temp=dest.with_suffix('.tmp')
    save_file({k:v.cpu().contiguous() for k,v in writer.state_dict().items()},str(temp))
    temp.replace(dest)
    dump(receipt,dict(data=meta['corpus'],weights=sha(dest),schedule=schedule.hexdigest(),
                      mode=mode,identity_supervision=binding,params=sum(p.numel() for p in writer.parameters()),seconds=seconds,
                      energy=cost,peak_allocated=torch.cuda.max_memory_allocated(),trace=trace,
                      distinct_pairs=int(touched.sum()),hidden_cells=num_hidden,exposures=STEPS*BATCH))


def main(binding=False):
    meta,data=corpus()
    with gpu_owner():
        torch.set_num_threads(2)
        for mode in ('ink','byte'):
            train(meta,data,mode,binding)
    names=[m+'-bound' if binding else m for m in ('ink','byte')]
    receipts={k:json.loads((PUB/f'{k}-fit.json').read_text()) for k in names}
    if len({r['schedule'] for r in receipts.values()})!=1:
        raise ValueError('Different training schedules')
    if binding:
        baseline=json.loads((PUB/'ink-fit.json').read_text())
        if baseline['schedule']!=receipts[names[0]]['schedule']:
            raise ValueError('Changed supervision schedule')
    dest=MODEL/('bound-writer.json' if binding else 'writer.json')
    dump(dest,dict(vocab=meta['train_shapes'],palette=sha(MODEL/'palette.npy'),
                                  weights={k:v['weights'] for k,v in receipts.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',action='store_true')
    main(parser.parse_args().binding)

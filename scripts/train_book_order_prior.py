"""Pretrain matched causal print decoders, then adapt both on identical bilingual ink."""
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from scripts.prepare_book_order_prior import PUB,WORK,RULE,ROOT,SEED,base
from scripts.train_open_news_writer import (
    parent,restore as old_news,prepare,corpus,atomic_save,pack,pictured_input,
    gpu_owner,configure_cuda_cli,MODEL as OLD_MODEL,SEED as NEWS_SEED,
)

MODEL=ROOT/'models/book_order_prior'

def bindings():
    return {str(p.relative_to(ROOT)):base.sha(p) for p in (RULE,PUB/'data.json',Path(__file__),
        ROOT/'scripts/prepare_book_order_prior.py',ROOT/'scripts/train_open_news_writer.py',
        ROOT/'ilm/visual_lm/glyph_draft_editor.py',ROOT/'ilm/visual_lm/ordered_translation.py',
        OLD_MODEL/'ink/fit.json',ROOT/'models/visual_training_breadth/broad/fit.json')}

def restore(arm,stage,device):
    if arm=='baseline':
        return parent('broad',device) if stage=='prior' else old_news('ink',device)
    if arm not in ('intact','scrambled') or stage not in ('prior','adapted'):
        raise ValueError((arm,stage))
    writer=parent('broad',device)
    p=MODEL/arm/stage
    receipt=base.read(p/'fit.json')
    assert receipt['bindings']==bindings() and receipt['weights']==base.sha(p/'weights.safetensors')
    writer.model.load_state_dict(load_file(str(p/'weights.safetensors')))
    writer.model.eval()
    return writer

def save(writer,arm,stage,history,start):
    p=MODEL/arm/stage
    p.mkdir(parents=True,exist_ok=True)
    weights=p/'weights.safetensors'
    atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in writer.model.state_dict().items()},f),weights)
    base.dump(p/'fit.json',dict(arm=arm,stage=stage,weights=base.sha(weights),bindings=bindings(),epochs=history,
        parameters=sum(p.numel() for p in writer.model.parameters()),
        trainables=sum(p.numel() for p in writer.model.parameters() if p.requires_grad),
        seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated()))

def neutral(net,x):
    zero=x.new_zeros((len(x),4,1,x.shape[-1]//4))
    return [(zero,zero)]*len(net.dec)

def next_logits(net,bank,inventory,previous):
    first=net.begin[0].expand(len(previous),1,-1)
    x=torch.cat((first,bank[previous]),1)
    h,_=net.decode(x,neutral(net,x),None)
    return net.distribution(h,net.prototypes(bank,inventory),None)

def budget(start):
    if monotonic()-start>1800 or torch.cuda.max_memory_allocated()>16*2**30:
        raise RuntimeError('Registered fit budget exceeded')

def pretrain(arm,rows):
    writer=parent('broad','cuda')
    net=writer.model.train()
    for name,p in net.named_parameters():
        p.requires_grad_(name=='end' or name.startswith(('final.','target.')) or
            (name.startswith('dec.') and '.cross.' not in name and '.cnorm.' not in name))
    variables=[p for p in net.parameters() if p.requires_grad]
    bank=net.optical(writer.features).detach()
    opt=torch.optim.AdamW(variables,lr=.00025,fused=True)
    order_rng=np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    start=monotonic();logs=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch,rate in enumerate((.00025,.00015,.00008),1):
        for g in opt.param_groups:g['lr']=rate
        order=order_rng.permutation(len(rows))
        total=0;running=0.;steps=0
        for k in range(0,len(order),128):
            y=rows[order[k:k+128]].long()
            n=int(y.ne(0).sum())
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=next_logits(net,bank,writer.inv[0],y[:,:-1])
            loss=F.nll_loss(logits.transpose(1,2),writer.local[0][y],ignore_index=-100,reduction='sum')
            (loss/n).backward()
            torch.nn.utils.clip_grad_norm_(variables,1,error_if_nonfinite=True)
            opt.step()
            running+=loss.detach().item();total+=n;steps+=1
            budget(start)
        logs.append(dict(epoch=epoch,positions=total,steps=steps,nll=running/total,schedule=sha256(order.astype('<i4').tobytes()).hexdigest()))
        print('prior',arm,epoch,logs[-1]['nll'],flush=True)
    save(writer,arm,'prior',logs,start)

def adapt(arm,news,extra,rehearsal):
    writer=restore(arm,'prior','cuda');net=writer.model.train()
    features=torch.cat((writer.features,extra.cuda()))
    historical,_,old_drafts,_=rehearsal
    opt=torch.optim.AdamW(net.parameters(),lr=.00010,fused=True)
    baseline=base.read(OLD_MODEL/'ink/fit.json')['epochs']
    rng=np.random.default_rng(NEWS_SEED);torch.manual_seed(NEWS_SEED)
    logs=[];start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for ep,expected in enumerate(baseline):
        for g in opt.param_groups:g['lr']=.00010 if ep<4 else .00005
        order=rng.permutation(len(news));hashes=sha256();total=0;running=0.;steps=0
        for j in range(0,len(order),16):
            ix=order[j:j+16];old=rng.integers(len(historical),size=16)
            hashes.update(ix.astype('<i4').tobytes());hashes.update(old.astype('<i4').tobytes())
            src,targets,drafts=map(list,zip(*(news[i] for i in ix)))
            for row,draft in zip(historical[old].cpu().numpy(),old_drafts[0][old].cpu().numpy(),strict=True):
                src.append(row[1][row[1]!=0].tolist());targets.append(row[0][row[0]!=0].tolist()+[1])
                drafts.append(draft[draft!=0].tolist() or [0])
            s,y,d=[pack(x,'cuda') for x in (src,targets,drafts)]
            with torch.no_grad():
                field=(255*writer.marker(features,s).sigmoid()*s.ne(0).unsqueeze(-1)).round().byte()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                law=net(features,pictured_input(s,field),y,writer.inv[0],0,d,writer.local[0])
            loss=F.nll_loss(law.transpose(1,2),writer.local[0][y],ignore_index=-100,reduction='sum')
            n=sum(map(len,targets))
            (loss/n).backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step()
            steps+=1;running+=loss.detach().item();total+=n;budget(start)
        result=dict(epoch=ep+1,steps=steps,positions=total,schedule=hashes.hexdigest(),nll=running/total)
        assert all(result[k]==expected[k] for k in ('steps','positions','schedule'))
        logs.append(result);print('adapt',arm,ep+1,round(running/total,4),flush=True)
    save(writer,arm,'adapted',logs,start)

def main():
    configure_cuda_cli()
    data=base.read(PUB/'data.json')
    assert data['protocol']==base.sha(RULE) and all(base.sha(WORK/k)==v for k,v in data['files'].items())
    with gpu_owner():
        with np.load(WORK/'train.npz') as source:
            for arm in ('intact','scrambled'):
                if not (MODEL/arm/'prior/fit.json').exists():
                    pretrain(arm,torch.as_tensor(source[arm],device='cuda'))
                else:restore(arm,'prior','cpu')
        ds,extra,_=prepare(parent('broad','cpu'));rehearsal=corpus()
        for arm in ('intact','scrambled'):
            if not (MODEL/arm/'adapted/fit.json').exists():adapt(arm,ds,extra,rehearsal)
            else:restore(arm,'adapted','cpu')
        a,b=[base.read(MODEL/m/'prior/fit.json')['epochs'] for m in ('intact','scrambled')]
        assert all(x[k]==y[k] for x,y in zip(a,b,strict=True) for k in ('positions','steps','schedule'))

if __name__=='__main__':main()

"""Fit matched visual writers to verified whole-answer preferences from training sources."""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from ilm.visual_lm.glyph_draft_editor import GlyphDraftEditor
from scripts.prepare_visual_sequence_preference import SEED,ROOT,WORK,PUB,MODEL,RULE,base,previous
from scripts.train_glyph_draft_editor import corpus,restore as parent_weights,atomic_save,batches
from scripts.train_full_image_roles import loaded,EventMarker,pictured_input,material,gpu_owner
from scripts.train_ordered_translation import pack
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.measure_writer_energy import snapshot,consumed

MARGIN=.20
BETA=4.


def binding():
    files=[Path(__file__),RULE,PUB/'selection.json',PUB/'mined.json',PUB/'verification.json',PUB/'confirmed.json',
           ROOT/'scripts/prepare_visual_sequence_preference.py',previous.MODEL/'editor/fit.json']
    return {str(p.relative_to(ROOT)):base.sha(p) for p in files}


def restore(arm,device):
    folder=MODEL/arm
    record=base.read(folder/'fit.json')
    path=folder/'weights.safetensors'
    assert record['weights']==base.sha(path) and record['binding']==binding()
    net=GlyphDraftEditor()
    net.load_state_dict(load_file(str(path)))
    return net.to(device).eval()


def stopped_targets(ids,lengths):
    return F.pad(ids,(0,1)).scatter_(1,torch.as_tensor(lengths[:,None],device=ids.device).long(),1)


def path_scores(net,optical,source,draft,target,t):
    features,inventory,local=optical
    scores=net(features,source,target,inventory[t],t,draft,local[t])
    desired=local[t][target]
    valid=target.ne(0)
    terms=scores.gather(-1,desired.clamp_min(0).unsqueeze(-1)).squeeze(-1)*valid
    return terms.sum(-1),valid.sum(-1)


def fit(arm,data,fields,optical,rows,approved):
    if (MODEL/arm/'fit.json').exists():
        restore(arm,'cpu')
        return
    pairs,lens,drafts,widths=data
    torch.manual_seed(SEED)
    net=parent_weights('editor','cuda').train()
    opt=torch.optim.AdamW(net.parameters(),lr=.00008,fused=True)
    start=monotonic()
    energy=snapshot()
    torch.cuda.reset_peak_memory_stats()
    history=[]

    def inputs(idx,t):
        s=1-t
        n=lens[idx,s].max()
        src=pictured_input(pairs[idx,s,:n],fields[idx,s,:n])
        target=stopped_targets(pairs[idx,t,:lens[idx,t].max()].long(),lens[idx,t])
        draft=drafts[t][idx,:max(1,widths[idx,t].max())].long()
        return src,draft,target

    for epoch,lr in enumerate((.00008,.00004)):
        opt.param_groups[0]['lr']=lr
        rng=np.random.default_rng(SEED+epoch)
        order=batches(lens,epoch)
        perm={t:rng.permutation(approved[t]) for t in (0,1)}
        cursor=[0,0]
        schedule=sha256()
        npos=pref_cells=neg_cells=preferences=wins=0
        train_loss=delta_sum=0.
        for step,idx in enumerate(order):
            schedule.update(idx.astype('<i4').tobytes())
            t=step%2
            chosen=perm[t][(cursor[t]+np.arange(min(48,len(perm[t]))))%len(perm[t])]
            cursor[t]+=len(chosen)
            schedule.update(chosen.astype('<i4').tobytes())
            batch=[rows[k] for k in chosen]
            selected=np.asarray([r['i'] for r in batch])
            ordinary=int(lens[idx].sum())+2*len(idx)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=0.
                for direction in (0,1):
                    src,ds,tgt=inputs(idx,direction)
                    ll,_=path_scores(net,optical,src,ds,tgt,direction)
                    loss=loss-ll.sum()/ordinary
                src,ds,pos=inputs(selected,t)
                neg=pack([r['cells']+[1] for r in batch],'cuda')
                lp,np_=path_scores(net,optical,src,ds,pos,t)
                ln,nn_=path_scores(net,optical,src,ds,neg,t)
                positive=lp/np_
                gap=positive-ln/nn_
                rank=F.softplus(BETA*(MARGIN-gap)).mean()/BETA
                loss=loss-.5*positive.mean()+(.5 if arm=='preference' else 0.)*rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            npos+=ordinary
            preferences+=len(batch)
            pref_cells+=int(np_.sum())
            neg_cells+=int(nn_.sum())
            delta_sum+=gap.detach().sum().item()
            wins+=int((gap.detach()>0).sum())
            train_loss+=loss.item()
            if monotonic()-start>1800 or torch.cuda.max_memory_allocated()>15*2**30:
                raise RuntimeError('Registered training resource ceiling')
        assert npos==lens.sum()+2*len(pairs)
        rec=dict(updates=len(order),ordinary_cells=npos,preferences=preferences,positive_cells=pref_cells,negative_cells=neg_cells,
                 schedule=schedule.hexdigest(),training_loss=train_loss/len(order),mean_margin=delta_sum/preferences,positive_margins=wins)
        history.append(rec)
        print(arm,'epoch',epoch+1,rec,flush=True)
    elapsed=monotonic()-start
    cost=consumed(energy,snapshot())
    folder=MODEL/arm
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'weights.safetensors'
    atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},f),path)
    base.dump(folder/'fit.json',dict(arm=arm,epochs=history,weights=base.sha(path),binding=binding(),seconds=elapsed,
              energy=cost,peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),beta=BETA,margin=MARGIN))


def main():
    configure_cuda_cli()
    with gpu_owner():
        verification=base.read(PUB/'verification.json')
        assert verification['emissions']==base.sha(WORK/'emissions.json')==base.read(PUB/'mined.json')['hash']
        rows=base.read(WORK/'emissions.json')
        eligible=[i for i,r in enumerate(rows) if r['stop'] and r['ref']!=r['free'] and
                  all(v[str(i)]==[True,False] for v in verification['votes'])]
        assert eligible==verification['admitted']
        final=base.read(PUB/'confirmed.json')
        assert final['previous']==base.sha(PUB/'verification.json')
        eligible=[i for i in eligible if final['votes'][str(i)]==[True,False]]
        assert eligible==final['admitted']
        approved=[[i for i in eligible if rows[i]['t']==t] for t in (0,1)]
        assert all(approved)
        data=corpus()
        p=data[0]
        optical=material('cuda')
        fields=torch.empty((*p.shape,3),dtype=torch.uint8,device='cuda')
        marker=loaded(EventMarker(),'marker','cuda')
        with torch.inference_mode():
            for t in (0,1):
                for j in range(0,len(p),256):
                    src=p[j:j+256,t].long()
                    fields[j:j+256,t]=(255*marker(optical[0],src).sigmoid()*src.ne(0).unsqueeze(-1)).round().byte()
        for arm in ('likelihood','preference'):
            fit(arm,data,fields,optical,rows,approved)
        a,b=[base.read(MODEL/m/'fit.json')['epochs'] for m in ('likelihood','preference')]
        for x,y in zip(a,b,strict=True):
            for k in ('schedule','updates','ordinary_cells','preferences','positive_cells','negative_cells'):
                assert x[k]==y[k]


if __name__=='__main__':
    main()

"""Compare the visible-ink objective with frozen base writers on fresh reserved poets."""
import json
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from scripts.prepare_verse_canvas import PUB,WORK,MODEL,corpus
from scripts.evaluate_verse_canvas import cases,generate,aligned
from scripts.write_verse_canvas import load,write
from scripts.train_verse_canvas import banks
from scripts.prepare_dialogue_ink import dump,sha
from scripts.measure_writer_energy import consumed,snapshot
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.portable_writer import save_grid,read_image

ARMS=('ink','ink-bound','byte','byte-bound')


@torch.inference_mode()
def optical(bank,net,vocab):
    source=bank[:,:vocab+1]
    keep=(source[2,1:].abs().sum(-1)>0).nonzero().flatten()+1
    original=net.source(source) if net is not None else source
    features=F.normalize(original.float(),dim=-1)
    hits=0
    direct=0
    for sel in keep.split(128):
        q=features[2,sel]
        score=torch.maximum(q@features[0,1:].T,q@features[1,1:].T)
        hits+=int((score.argmax(1)+1==sel).sum())
        if net is not None:
            raw=original[2,sel]
            logits=net.ink(F.layer_norm(raw,(raw.shape[-1],)))
            direct+=int((logits.argmax(1)+1==sel).sum())
    return dict(visible=len(keep),nearest=hits,shared_head=direct if net else None)


def summarize(output,rows,mask,select):
    truth=rows[:,9:]
    correct=(output==truth)&mask
    row_exact=(correct==mask).all(1)
    return dict(lines=int(select.sum()),cells=int(mask[select].sum()),hits=int(correct[select].sum()),
                exact=int(row_exact[select].sum()),
                repeated=int(sum(len(set(p[:n]))<n for p,n in zip(output[select],rows[select,1]))))


def main():
    meta,data=corpus()
    report=PUB/'confirmation.json'
    binding=dict(original=sha(MODEL/'writer.json'),bound=sha(MODEL/'bound-writer.json'),corpus=meta['corpus'])
    if report.exists():
        existing=json.loads(report.read_text())
        if (existing['archive']!=sha(PUB/'confirmation.npz') or
                any(existing[k]!=v for k,v in binding.items() if k in existing)):
            raise ValueError('Saved confirmation differs from the release')
        print('Verified existing confirmation',existing['archive'],flush=True)
        return
    ix,rows,donor,gaps=cases(data,subset='dev',limit=None,seed=350372)
    authors={k:set(data[k][:,0]) for k in ('train','dev','test')}
    if authors['dev'] & (authors['train']|authors['test']):
        raise ValueError('Confirmation author collision')
    same=np.arange(len(ix))
    full=np.arange(7)<rows[:,1,None]
    truth=rows[:,9:]
    covered=(data['tiles'][2].any(axis=1)[rows[:,2:9]]|~full).all(1)
    path=PUB/'confirmation.npz'
    output={}
    recovered=path.exists()
    if recovered:
        with np.load(path) as existing:
            for k,expected in [('ix',ix),('donor',donor),('gaps',gaps),('covered',covered)]:
                if not np.array_equal(existing[k],expected):
                    raise ValueError('Different confirmation examples')
            output={k:existing[k] for k in existing if k not in ('ix','donor','gaps','covered')}
    probes={}
    cost={}
    timing=WORK/'confirmation-timing.json'
    with gpu_owner():
        torch.set_num_threads(2)
        for arm in ARMS:
            clock=monotonic()
            start=snapshot()
            mode=arm.split('-')[0]
            bank=banks(data,mode,'cuda')
            net,_=load(MODEL,arm,'cuda')
            if arm=='ink':
                probes['raw']=optical(bank,None,meta['train_shapes'])
            if mode=='ink':
                probes[arm]=optical(bank,net,meta['train_shapes'])
            if recovered:
                continue
            for name,mask in [('blank',full),('gap',gaps)]:
                for cond,peer,face in [('original',same,0),('exchanged',donor,0),('droid',same,2)]:
                    if cond=='droid' and mode=='byte':
                        continue
                    output[f'{arm}_{name}_{cond}']=generate(net,bank,rows,peer,mask,face,'confidence')
            cost[arm]=dict(seconds=monotonic()-clock,energy=consumed(start,snapshot()))
    if not recovered:
        dump(timing,dict(binding=binding,cost=cost))
    elif timing.exists():
        timing_record=json.loads(timing.read_text())
        if timing_record['binding']!=binding:
            raise ValueError('Cost record belongs to different weights or data')
        cost=timing_record['cost']
    count=aligned(data['train'],rows)
    output.update(aligned_blank_original=count,aligned_gap_original=np.where(gaps,count,truth))
    summary={}
    all_rows=np.ones(len(rows),bool)
    for k,pred in output.items():
        _,task,condition=k.split('_')
        summary[k]=summarize(pred,rows,gaps if task=='gap' else full,covered if condition=='droid' else all_rows)
    _,author_index=np.unique(rows[:,0],return_inverse=True)
    number=author_index.max()+1
    sample=np.random.default_rng(350373).integers(number,size=(5000,number))
    comparison={}
    for task,mask in [('blank',full),('gap',gaps)]:
        for cond in ('original','droid'):
            sel=covered if cond=='droid' else all_rows
            bound=output['ink-bound_'+task+'_'+cond]
            rivals=['ink_'+task+'_'+cond]
            if cond=='original':
                rivals+=['ink-bound_'+task+'_exchanged','byte-bound_'+task+'_original','aligned_'+task+'_original']
            for other in rivals:
                ours=(bound==truth)&mask
                theirs=(output[other]==truth)&mask
                delta=(ours.sum(1)-theirs.sum(1))*sel
                denom=mask.sum(1)*sel
                boot=np.bincount(author_index,weights=delta)[sample].sum(1)/np.bincount(author_index,weights=denom)[sample].sum(1)
                comparison[other]=dict(delta=float(delta.sum()/denom.sum()),ci=np.quantile(boot,[.025,.975]).tolist(),
                    ours_only_exact=int(((ours==mask).all(1)&~(theirs==mask).all(1)&sel).sum()),
                    theirs_only_exact=int(((theirs==mask).all(1)&~(ours==mask).all(1)&sel).sum()))
    if not recovered:
        np.savez_compressed(path,ix=ix,donor=donor,gaps=gaps,covered=covered,**output)
    reps=json.loads((WORK/'readback.json').read_text())
    key='ink-bound_gap_original'
    passing=(output[key]==truth).all(1)
    contrast=passing&~(output['ink-bound_gap_exchanged']==truth).all(1)
    picks={'first':0,'miss':int(np.flatnonzero(~passing)[0])}
    if contrast.any():
        picks['restore']=int(np.flatnonzero(contrast)[0])
    exact=(output['ink-bound_blank_original']==truth).all(1)
    if exact.any():
        picks['complete']=int(np.flatnonzero(exact)[0])
    examples={}
    torch.set_num_threads(2)
    for name,i in picks.items():
        n=int(rows[i,1])
        seqs=dict(source=rows[i,2:2+n],reference=truth[i,:n],template=np.where(gaps[i,:n],0,truth[i,:n]),
                  blank=output['ink-bound_blank_original'][i,:n],restored=output[key][i,:n],
                  exchanged=output['ink-bound_gap_exchanged'][i,:n],base=output['ink_blank_original'][i,:n])
        images=WORK/'confirmation'/name
        images.mkdir(exist_ok=True,parents=True)
        for keyname,seq in seqs.items():
            save_grid(data['tiles'][0,seq].tobytes(),images/(keyname+'.png'))
        src=read_image(images/'source.png',None)
        for keyname,template in [('blank',np.zeros_like(src)),('restored',read_image(images/'template.png',None))]:
            actual=write(MODEL,src,template,'confidence','cpu',binding=True)
            assert np.array_equal(actual,read_image(images/(keyname+'.png'),None))
        examples[name]=dict(index=i,poet=reps['authors'][rows[i,0]],
                         **{k:''.join(reps['shapes'][j] if j else '□' for j in seq) for k,seq in seqs.items()})
    dump(PUB/'confirmation.json',dict(summary=summary,comparisons=comparison,probes=probes,
             examples=examples,authors=int(number),donors_same_author=int((rows[:,0]==rows[donor,0]).sum()),
             archive=sha(path),**binding,
             held_face_coverage=int(covered.sum()),cost=cost,from_saved_outputs=recovered))
    print(summary,probes,examples,flush=True)


if __name__=='__main__':
    main()

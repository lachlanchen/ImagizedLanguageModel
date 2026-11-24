"""Evaluate the registered held books on original, exchanged and short physical histories."""
from collections import defaultdict
import time
import json
import numpy as np
import torch
from ilm.visual_lm.book_predict import Predictor
from scripts.measure_writer_energy import consumed,snapshot
from scripts.prepare_book_prefix import corpus,WORK,MODEL,PUB
from scripts.prepare_dialogue_ink import dump,sha
from scripts.package_book_prefix import ARMS
from scripts.train_span_choice import gpu_owner


def held(meta,stream):
    starts, works, docs = [], [], []
    for split, work, doc, _, a, b in meta['records']:
        if split != 'test':
            continue
        for start in range(a+32,b-15,16):
            starts.append(start)
            works.append(work)
            docs.append(doc)
    starts = np.asarray(starts)
    work = np.asarray(works)
    donor = np.zeros(len(work),int)
    for name in sorted(set(works)):
        rows = np.flatnonzero(work==name)
        other = np.flatnonzero(work!=name)
        donor[rows] = np.resize(other,len(rows))
    prefix = stream[starts[:,None]+np.arange(-32,0)]
    exchanged = prefix[donor].copy()
    exchanged[:,-2:] = prefix[:,-2:]
    return starts,work,np.asarray(docs),prefix,exchanged,stream[starts[:,None]+np.arange(16)]


def summarize(found,truth,nll,known):
    result = {}
    for mode,pred in found.items():
        match = pred == truth
        result[mode] = dict(correct=int(match.sum()), total=match.size,
                            exact8=int(match[:,:8].all(1).sum()),exact16=int(match.all(1).sum()),
                            adjacent_repeat=int((pred[:,1:] == pred[:,:-1]).sum()),
                            unique=len(np.unique(pred,axis=0)), blocks=len(truth))
    result['common_nll'] = float(nll[known].mean())
    result['known'] = int(known.sum())
    return result


def run(arm,meta,data,held_data,device):
    starts,works,docs,prefix,exchanged,truth = held_data
    clock = time.monotonic()
    initial = snapshot()
    model = Predictor(MODEL,arm,data,device)
    found = defaultdict(list)
    logs = []
    for a in range(0,len(truth),48):
        rows = slice(a,a+48)
        src, target = prefix[rows],truth[rows]
        context = np.concatenate([src,target[:,:-1]],axis=1)
        mass = model.probabilities(context,31)
        if not torch.allclose(mass.sum(-1), torch.ones_like(mass[:,:,0]),atol=2e-5,rtol=0):
            raise ValueError('Unnormalized next image probabilities')
        found['teacher'].append(mass.argmax(-1).cpu().numpy().astype(np.uint16))
        obs = torch.as_tensor(target.astype(np.int64),device=device)
        prob = mass.gather(-1,obs.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        logs.append(-np.log(prob,where=prob>0,out=np.full_like(prob,-np.inf)))
        del mass
        found['free'].append(model.generate(src,16))
        found['swapped'].append(model.generate(exchanged[rows],16))
        found['two'].append(model.generate(src,16,2))
    pred = {k:np.concatenate(v) for k,v in found.items()}
    nll = np.concatenate(logs)
    known = truth <= meta['train_shapes']
    if not np.isfinite(nll[known]).all():
        raise ValueError('Nonfinite common likelihood')
    duration = time.monotonic()-clock
    energy = consumed(initial,snapshot())
    groups = {}
    for work in ['all',*sorted(set(works))]:
        rows = np.ones(len(works),bool) if work=='all' else works==work
        groups[work] = summarize({k:v[rows] for k,v in pred.items()},truth[rows],nll[rows],known[rows])
    path = PUB/f'{arm}-predictions.npz'
    np.savez_compressed(path,starts=starts, truth=truth, nll=nll, **pred)
    out = dict(groups=groups,seconds=duration,energy=energy,predictions=sha(path))
    print(arm,groups['all'],round(duration,2),flush=True)
    return out,pred


def main():
    meta,data = corpus()
    h = held(meta,data['stream'])
    starts,works,docs,prefix,_,truth = h
    all_pred, scores = {},{}
    with gpu_owner():
        torch.set_num_threads(2)
        for arm in (*ARMS,'counts'):
            scores[arm], all_pred[arm] = run(arm,meta,data,h,'cuda')
    boot = {}
    volumes, doc_index = np.unique(docs,return_inverse=True)
    volume_work = np.array([works[np.flatnonzero(docs==d)[0]] for d in volumes])
    sizes = np.bincount(doc_index)*16
    rng = np.random.default_rng(194671)
    strata = [np.flatnonzero(volume_work==w) for w in sorted(set(works))]
    samples = np.column_stack([rng.choice(s,(5000,len(s))) for s in strata])
    choices = [('free_prefix',('ink-hybrid','free'),('ink-hybrid','swapped')),
               ('free_two',('ink-hybrid','free'),('ink-hybrid','two')),
               ('teacher_counts',('ink-hybrid','teacher'),('counts','teacher')),
               ('teacher_neural',('ink-hybrid','teacher'),('ink-neural','teacher')),
               ('teacher_byte',('ink-hybrid','teacher'),('byte-hybrid','teacher'))]
    for name,(arm,mode),(other,control) in choices:
        x = (all_pred[arm][mode]==truth).sum(1)-(all_pred[other][control]==truth).sum(1)
        by_doc = np.bincount(doc_index,weights=x)
        delta = by_doc[samples].sum(1)/sizes[samples].sum(1)
        boot[name] = dict(estimate=x.sum()/truth.size, ci=np.quantile(delta,[.025,.975]).tolist())
    novel = truth > meta['train_shapes']
    seen = (truth[:,:,None]==prefix[:,None,:]).any(-1)
    reachable = ~novel|seen
    result = dict(models=scores, bootstrap=boot, volumes=len(volumes),blocks=len(starts),
                  volumes_by_work={k:int((volume_work==k).sum()) for k in sorted(set(works))},
                  reference_repeats=int((truth[:,1:]==truth[:,:-1]).sum()),positions=truth.size,
                  novel=int(novel.sum()), novel_in_prefix=int((novel&seen).sum()),
                  reachable=int(reachable.sum()), wholly_reachable=int(reachable.all(1).sum()))
    dump(PUB/'results.json',result)
    print(boot,flush=True)


if __name__ == '__main__':
    main()

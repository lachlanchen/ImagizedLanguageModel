"""Generate, archive and score complete insertion images at all held-out aligned sites."""
import fcntl
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import visual_input
from scripts.probe_visual_span_exchange import pairs
from scripts.train_visual_phrases import (ARMS, OUT, ROOT, RUN, WORK, frozen, guard, initialize, insert_runs,
    load, require, sha, tensor_pad, tensors, tiles, training_rows, write_new)

COHORTS = ('discovery', 'replication')


def sites(rows):
    return [(i,c,j,k) for i,r in enumerate(rows) for c,j,k in insert_runs(r['a'])]


def keys(row, cursor, pos):
    x, y = row['x'], row['y']
    prev = tuple(y[max(0,pos-2):pos])
    before = x[cursor-1] if cursor else -1
    after = x[cursor] if cursor < len(x) else -1
    return ((0,before,after,prev), (1,before,after,prev[-1:]), (2,before,after),
            (3,before,prev[-1:]), (4,after,prev[-1:]), (5,prev), (6,before),
            (7,after), (8,prev[-1:]), (9,))


def local_counts(train, cohorts):
    queried = {key for rows in cohorts.values() for i,c,j,k in sites(rows) for key in keys(rows[i],c,j)}
    counts = defaultdict(Counter)
    types = set()
    for row in train:
        for c,j,k in insert_runs(row['a']):
            phrase = tuple(row['y'][j:k])
            types.add(phrase)
            for key in keys(row,c,j):
                if key in queried:
                    counts[key][phrase] += 1
    # Require two observations at each context level before using it.
    modes = {key: table.most_common(1)[0][0] for key,table in counts.items() if table.total() >= 2}
    return modes, types


def checkpoint(arm):
    net = initialize()
    if arm in ARMS:
        path = RUN/arm/'writer.safetensors'
        require(sha(path) == load(RUN/arm/'fit.json')['checkpoint_sha256'], 'Phrase checkpoint changed')
        net.load_state_dict(load_file(str(path)))
    return net.cuda().eval()


@torch.inference_mode()
def produce(net, arm, rows, bank, qids, began):
    ink = torch.as_tensor(np.unpackbits(bank,axis=1), device='cuda').bool()
    bits = ink[:,net.landmarks]
    features = visual_input(ink,net.landmarks)
    q = features[qids][None]
    results, ends = [], []
    for first in range(0,len(rows),24):
        block = [r for r in rows[first:first+24] if r['x']]
        locations = sites(block)
        if not locations:
            continue
        mem, initial, hist, _, lengths = tensors(block,features,bits,net,q,arm=='prefix')
        for start in range(0,len(locations),128):
            b,c,j,_ = torch.tensor(locations[start:start+128],device='cuda').T
            at = net.boundary(mem,b,c,ablate=arm=='prefix',original=arm=='original')
            emitted,n,ended = net.complete(at,hist[b,j],initial[b],c==lengths[b],original=arm=='original')
            mask = emitted.cpu().numpy()
            for pic,count in zip(mask,n.tolist(),strict=True):
                results.append(np.packbits(pic[:count],axis=1))
            ends.extend(ended.tolist())
        guard(began)
    return results, ends


@torch.inference_mode()
def exchange(net, arm, rows, qids, bank):
    selected = pairs(rows)
    cursor = {(i,j,k): c for i,c,j,k in sites(rows)}
    ink = torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    bits = ink[:,net.landmarks]
    features = visual_input(ink,net.landmarks)
    probs = np.zeros((len(selected),2,2,2))
    requests = []
    for p,pair in enumerate(selected):
        for side,site in enumerate(pair):
            i,j,k = site
            h,u,v = pair[1-side]
            x,prefix = rows[i]['x'], rows[i]['y'][:j]
            c = cursor[site]
            for rev in (0,1):
                for foil,phrase in enumerate((rows[i]['y'][j:k],rows[h]['y'][u:v])):
                    requests.append(((p,side,rev,foil), x[::-1] if rev else x, len(x)-c if rev else c,
                                     prefix,phrase))
    for first in range(0,len(requests),48):
        batch = requests[first:first+48]
        x,n = tensor_pad([r[1] for r in batch],features.device)
        y,_ = tensor_pad([r[3]+r[4] for r in batch],features.device)
        mem,_,hist = net.contexts(features[x],n,bits[y],features[qids][None].expand(len(x),-1,-1),ablate=arm=='prefix')
        ix,c,j,last = torch.tensor([(b,r[2],len(r[3])+a,a+1==len(r[4])) for b,r in enumerate(batch)
                                    for a in range(len(r[4]))],device='cuda').T
        draw,end = net.losses(net.boundary(mem,ix,c,ablate=arm=='prefix'),hist,ix,j,bits[y[ix,j]])
        score = -draw - F.binary_cross_entropy_with_logits(end,last.float(),reduction='none')
        for r,value in zip(batch,score.split([len(r[4]) for r in batch]),strict=True):
            probs[r[0]] = float(value.sum())
    margins = (probs[:,:,:,0]-probs[:,:,:,1]).sum(1)
    if arm=='prefix':
        require(np.allclose(margins[:,0],margins[:,1],atol=.0001), 'Source reached the ablated scorer')
    return dict(pairs=len(selected),assigned=(margins>0).sum(0).tolist(),margin_nats=margins.tolist())


def chapter_interval(rows, delta):
    chapters,indices = np.unique(rows,return_inverse=True)
    counts = np.bincount(indices)
    sums = np.bincount(indices,weights=delta)
    sampled = np.random.default_rng(3129).integers(len(chapters),size=(4000,len(chapters)))
    return (100*np.quantile(sums[sampled].sum(1)/counts[sampled].sum(1),[.025,.975])).tolist()


def recount(path, rows, bank, train_types, pair_sites):
    answers = []
    flags = defaultdict(list)
    locations = sites(rows)
    index = {(i,j,k):idx for idx,(i,c,j,k) in enumerate(locations)}
    with np.load(path) as archive:
        pixels, edges, stops = (archive[k] for k in ('pixels','offsets','stopped'))
        require(edges[0]==0 and edges[-1]==len(pixels) and len(stops)==len(locations)
                and len(edges)==len(stops)+1 and np.all(np.diff(edges)>0), 'Raster archive coverage')
        for p,(i,c,j,k) in enumerate(locations):
            target = rows[i]['y'][j:k]
            drawn = pixels[edges[p]:edges[p+1]]
            correct = bool(stops[p]) and np.array_equal(drawn,bank[target])
            answers.append(correct)
            flags['first_glyph'].append(np.array_equal(drawn[0],bank[target[0]]))
            flags['correct_length'].append(len(drawn)==len(target))
            flags['novel'].append(tuple(target) not in train_types)
            flags['multi'].append(len(target)>1)
            flags['absent'].append(set(target).isdisjoint(rows[i]['x']))
        exact = np.asarray(answers)
        stats = dict(sites=len(exact),exact=int(exact.sum()),stopped=int(stops.sum()),emitted_cells=len(pixels),
                     first_glyph_exact=sum(flags['first_glyph']),correct_length=sum(flags['correct_length']),
                     both_exchange=sum(answers[index[a]] and answers[index[b]] for a,b in pair_sites))
        for name in ('novel','multi','absent'):
            mask = np.asarray(flags[name])
            stats[name] = [int(exact[mask].sum()),int(mask.sum())]
        stats['novel_types_exact'] = len({tuple(rows[i]['y'][j:k]) for s,(i,c,j,k) in enumerate(locations)
                                          if exact[s] and flags['novel'][s]})
        return exact, stats


def run():
    began = time.monotonic()
    guard(began)
    require(load(RUN/'source/fit.json')['schedule']==load(RUN/'prefix/fit.json')['schedule'],'Unmatched fits')
    bank = tiles()
    rows = {s: list(frozen(s)) for s in COHORTS}
    lookup, train_types = local_counts(training_rows(),rows)
    qids = load(OUT/'preparation.json')['question_tiles'][0][0]
    for split, cohort in rows.items():
        directory = RUN/split
        directory.mkdir()
        locations = sites(cohort)
        selected = pairs(cohort)
        recorded = load(WORK/(split+'_exchange.json'))['pairs']
        require(recorded==[[list(a),list(b)] for a,b in selected], 'Phrase exchange roster changed')
        report,exact = {},{}
        for arm in ('original','local',*ARMS):
            scores = None
            if arm=='local':
                drawn = [bank[list(next(lookup[key] for key in keys(cohort[i],c,j) if key in lookup))]
                         for i,c,j,k in locations]
                stops = np.ones(len(drawn),bool)
            else:
                net = checkpoint(arm)
                drawn,stops = produce(net,arm,cohort,bank,qids,began)
                if arm in ARMS:
                    scores = exchange(net,arm,cohort,qids,bank)
                    write_new(directory/(arm+'_exchange.json'),scores)
                del net
            path = directory/(arm+'.npz')
            np.savez_compressed(path,pixels=np.concatenate(drawn),offsets=np.cumsum([0,*map(len,drawn)]),
                                stopped=np.asarray(stops,bool))
            exact[arm],r = recount(path,cohort,bank,train_types,selected)
            if scores:
                r['assigned_original_and_reverse'] = scores['assigned']
            r['pixels_sha256'] = sha(path)
            report[arm] = r
            print(split,arm,r,flush=True)
        groups = [cohort[i]['g'] for i,c,j,k in locations]
        improvements = {a: chapter_interval(groups,exact['source'].astype(int)-exact[a]) for a in ('original','local','prefix')}
        passed = (min(v[0] for v in improvements.values())>0 and report['source']['novel_types_exact']>0 and
                  report['source']['stopped']>=.99*len(locations))
        write_new(directory/'report.json',dict(arms=report,chapter_gain_pp=improvements,qualifies=bool(passed),
                  train_insertion_types=len(train_types),unrendered_sentences=sum(not r['x'] for r in cohort)))
        guard(began)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

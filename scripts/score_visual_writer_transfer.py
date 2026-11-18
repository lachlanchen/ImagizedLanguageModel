"""Recount complete image streams and compare decisions on the same passages."""
from collections import defaultdict
import json

import numpy as np

from scripts.check_pictured_idioms import raw_cases
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_visual_writer_distillation import DATA, OUT, PRIOR, ROOT
from scripts.train_visual_extractive_reader import save_page

NAMES = ('published', 'original_targets', 'distilled_targets')
PUBLIC = ROOT/'references/evidence/writer_transfer_20261118'


def matching_pairs(contexts, targets, source_group):
    buckets = defaultdict(list)
    for i, row in enumerate(contexts):
        buckets[row[23:25].tobytes()].append(i)
    pairs = []
    for rows in buckets.values():
        while rows:
            i = rows.pop(0)
            for pos, j in enumerate(rows):
                if source_group[i] != source_group[j] and not np.array_equal(targets[i], targets[j]):
                    pairs.append((i, rows.pop(pos)))
                    break
    return np.asarray(pairs).T


def paired_interval(left, right, group):
    ids, inv = np.unique(group, return_inverse=True)
    counts = np.bincount(inv)
    delta = np.bincount(inv, weights=left.astype(int)-right.astype(int))
    rng = np.random.default_rng(613)
    samples = []
    for _ in range(2000):
        selected = rng.integers(len(ids), size=len(ids))
        samples.append(delta[selected].sum()/counts[selected].sum()*100)
    return np.quantile(samples, [.025,.975]).tolist()


def score(split, training_targets):
    fresh = split == 'fresh'
    base = OUT/'fresh' if fresh else DATA
    with np.load(base/('contexts.npz' if fresh else 'validation.npz')) as a:
        x,y,group,tags = (a[k] for k in ('context','target','group','tag'))
    with np.load((base if fresh else PRIOR)/'diagnostic_choices.npz') as a:
        teacher = (a['supplied_inventory_selection'] == y).all(1)
        local = a['local_exact']
    px = np.load(base/'pixels.npy')
    truth = px[y]
    absent = ~(y[:,:,None] == x[:,None,:]).any(2)
    unseen = np.array([row.tobytes() not in training_targets for row in y])
    i,j = matching_pairs(x,y,group)
    stats = dict(n=len(x), groups=len(np.unique(group)), all_absent=int(absent.all(1).sum()),
                 unseen=int(unseen.sum()), pairs=len(i), reader=int(teacher.sum()), local=int(local.sum()), arms={})
    hits, counts = {}, {}
    for name in NAMES:
        path = base/name if fresh else (PRIOR if name=='published' else OUT/name)
        with np.load(path/'generated.npz') as a:
            ink,n,stop = a['packed'],a['length'],a['stopped']
        hit = (ink[:,:4]==truth).all((1,2)) & stop & (n==4)
        if not hits:
            original = hit.copy()
        hits[name] = hit
        reported = load(path/'result.json')
        require(int(hit.sum())==reported['complete_exact'],'Archived output disagrees')
        counts[name] = (ink,n,stop)
        stats['arms'][name] = dict(exact=int(hit.sum()), exchanged=reported['swapped_exact'],
            distinct=reported['unique_correct'], reader_correct=int((hit & teacher).sum()),
            recovered_lost_reader=int((hit & teacher & ~original).sum()), reader_wrong=int((hit & ~teacher).sum()),
            local_wrong=int((hit & ~local).sum()), all_absent=int((hit & absent.all(1)).sum()),
            unseen=int((hit & unseen).sum()), pairs_both=int((hit[i] & hit[j]).sum()),
            pairs_changed=int(((n[i]!=n[j])|(stop[i]!=stop[j])|(ink[i]!=ink[j]).any((1,2))).sum()),
            gain=int((hit & ~original).sum()), loss=int((~hit & original).sum()), stopped=int(stop.sum()),
            emitted=reported['emitted_cells'], known=reported['training_target_shape_hits'],
            archive_sha256=sha(path/'generated.npz'))
    treatment = hits['distilled_targets']
    for name in NAMES[:2]:
        counts_name = stats['arms']['distilled_targets']
        counts_name['paired_pp_vs_'+name] = paired_interval(treatment,hits[name],group)
    meet = stats['arms']['distilled_targets']
    stats['transfer_pass'] = bool(all(meet['paired_pp_vs_'+name][0]>0 and
                                 meet['reader_correct']>stats['arms'][name]['reader_correct'] for name in NAMES[:2])
                                 and meet['known']>=.99*meet['emitted'])
    if not fresh:
        require(len(i)==8900 and stats['arms']['published']['exact']==134 and int(teacher.sum())==930,
                'Original benchmark changed')
        cases = raw_cases()
        samples = {}
        for name,mask in [('gain',treatment & ~original),('miss',~treatment & teacher)]:
            if not mask.any():
                continue
            row = int(np.flatnonzero(mask)[0])
            before,after,word = cases[int(tags[row])]
            img,n,_ = counts['distilled_targets']
            for label,tiles in [('before',px[x[row,12:24]]),('after',px[x[row,24:36]]),
                                ('output',img[row,:n[row]]),('reference',truth[row])]:
                if len(tiles):
                    save_page(PUBLIC/f'{name}_{label}.png',tiles)
            samples[name] = dict(row=row, tag=int(tags[row]), before=before,after=after,
                                 reference=word, produced_cells=int(n[row]))
        stats['samples'] = samples
    return stats


def main():
    PUBLIC.mkdir()
    with np.load(DATA/'train.npz') as train:
        training = {row.tobytes() for row in train['target']}
    result = {name:score(name,training) for name in ('development','fresh')}
    result['fit'] = {name:load(OUT/name/'fit.json') for name in NAMES[1:]}
    result['teacher'] = load(OUT/'teacher.json')
    result['execution'] = load(OUT/'execution.json')
    result['fresh_execution'] = load(OUT/'fresh/execution.json')
    result['fresh_preparation'] = load(OUT/'fresh/preparation.json')
    write_new(PUBLIC/'result.json',result)
    for name in ('development','fresh'):
        print(name,json.dumps(result[name],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

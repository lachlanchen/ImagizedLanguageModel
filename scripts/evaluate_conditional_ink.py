"""Evaluate physical group interactions against unchanged whole-phrase controls."""
import fcntl
import time
from collections import defaultdict

import numpy as np
import torch

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.local_ink import signatures
from scripts.evaluate_stream_ink import COHORTS, PRIOR, RUN, generate, records, roster, save_streams
from scripts.evaluate_visual_phrases import chapter_interval, keys, local_counts, pairs, recount
from scripts.train_conditional_ink import (DIRECTORY, WORK, guard, load, loss_at, nll, observations,
                                          require, sha, sites, training_rows, write_new)
from scripts.train_local_ink import ROOT, fields


def inputs(split):
    if split != 'confirmation':
        return roster(split)
    directory = WORK / 'confirmation'
    doc = load(directory/'receipt.json')
    require(sha(directory/'rows.jsonl') == doc['rows_sha256']
            and sha(directory/'pixels.npy') == doc['bank_sha256'], 'Confirmation roster changed')
    return list(records(directory/'rows.jsonl')), np.load(directory/'pixels.npy')


def control(split, name):
    if split == 'confirmation':
        return WORK/'results'/split/(name + '.npz')
    doc = load(RUN/'stream'/split/'report.json')
    previous = 'source' if name == 'marginal' else 'local'
    path = RUN/'stream'/split/(previous+'.npz')
    if previous == 'local' and split in COHORTS:
        path = PRIOR/split/'local.npz'
    require(sha(path) == doc['arms'][previous]['pixels_sha256'], 'Prior comparison changed')
    return path


@torch.inference_mode()
def run():
    from safetensors import safe_open
    from scripts.infer_visual_phrase import MODELS
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    directory = WORK/'results'
    directory.mkdir()
    path = WORK/'transitions.safetensors'
    require(sha(path) == load(WORK/'fit.json')['checkpoint_sha256'], 'Fit changed')
    model = ConditionalInk.restore(path, sha(DIRECTORY/'manifest.json'), 'cuda')
    training = training_rows()
    confirm, _ = inputs('confirmation')
    lookup, types = local_counts(training, {'confirmation': confirm})
    del training
    with safe_open(MODELS/'editor.safetensors', framework='np') as weights:
        landmarks = weights.get_tensor('landmarks')

    for split in (*COHORTS, 'fresh', 'confirmation'):
        rows, bank = inputs(split)
        where = sites(rows)
        grouped = defaultdict(list)
        context = fields(rows, signatures(np.unpackbits(bank, axis=1)[:, landmarks]), where)
        for index, key in enumerate(context):
            grouped[tuple(key)].append(index)
        ambiguous = [n for group in grouped.values()
                     if len({tuple(rows[where[i][0]]['y'][where[i][2]:where[i][3]]) for i in group}) > 1
                     for n in group]
        target_groups = [rows[i]['g'] for i, _, _, _ in where]
        selected = pairs(rows)
        dest = directory/split
        dest.mkdir()
        data = observations(rows, bank, started)
        teacher = np.zeros(2)
        for indices in torch.arange(len(data[0]), device='cuda').split(512):
            base, delta, target = loss_at(model, data, indices)
            teacher += [float(nll(z,target).double().sum()) for z in (base,base+delta)]
        teacher /= len(data[0])
        del data
        arms, hits = {}, {}
        for name in ('marginal', 'local', 'interaction'):
            seconds = peak = None
            if name == 'interaction' or split == 'confirmation':
                path = dest / (name+'.npz')
                if name == 'local':
                    out = [bank[list(next(lookup[k] for k in keys(rows[i],c,j) if k in lookup))]
                           for i,c,j,_ in where]
                    stops = np.ones(len(out), bool)
                else:
                    out, stops, _, seconds, peak = generate(rows, 'source', bank, started, stream=True,
                                                   interaction=model if name=='interaction' else None)
                save_streams(path, out, stops)
            else:
                path = control(split,name)
            hits[name], stat = recount(path, rows, bank, types, selected)
            stat.update(ambiguous_exact=int(hits[name][ambiguous].sum()), seconds=seconds,
                        peak_gpu_mib=peak, pixels_sha256=sha(path))
            arms[name] = stat
        gains = {n:chapter_interval(target_groups,hits['interaction'].astype(int)-hits[n])
                 for n in ('marginal','local')}
        cand = arms['interaction']
        passes = (min(g[0] for g in gains.values()) > 0 and cand['novel_types_exact'] > 0
                  and cand['stopped'] >= .99*len(where))
        write_new(dest/'report.json', dict(arms=arms, chapter_gain_pp=gains, qualifies=bool(passes),
                  ambiguous_sites=len(ambiguous), exchange_pairs=len(selected), teacher_cells=sum(k-j for i,c,j,k in where),
                  teacher_transition_nll=teacher.tolist(), sentences=len(rows), rendered=sum(bool(r['x']) for r in rows)))
        print(split,{k:v['exact'] for k,v in arms.items()},'NLL',teacher.tolist(),'gate',passes,flush=True)
        guard(started)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

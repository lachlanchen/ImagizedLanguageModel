"""Generate on both matched insertion cohorts and one newly reserved sentence cohort."""
import fcntl
import time

import numpy as np
import torch
from scripts.evaluate_local_ink import COHORTS, PRIOR, ROOT, RUN, chapter_interval, frozen, generate, guard, load, pairs, recount, require, sha, sites, tiles, training_rows, write_new
from scripts.evaluate_visual_phrases import local_counts, produce, checkpoint
from scripts.train_local_ink import OUT
from scripts.prepare_edit_diversity import records


def roster(name):
    if name in COHORTS:
        return list(frozen(name)),tiles()
    d = RUN/'fresh_roster'
    receipt = load(d/'receipt.json')
    require(sha(d/'rows.jsonl')==receipt['rows_sha256'] and sha(d/'pixels.npy')==receipt['bank_sha256'],
            'Independent roster changed')
    return list(records(d/'rows.jsonl')),np.load(d/'pixels.npy')


def save_streams(path,images,stop):
    np.savez_compressed(path,pixels=np.concatenate(images),offsets=np.cumsum([0,*map(len,images)]),
                        stopped=np.asarray(stop,bool))


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    start = time.monotonic()
    guard(start)
    train = training_rows()
    fresh,_ = roster('fresh')
    lookup,types = local_counts(train,{'fresh':fresh})
    del train
    from scripts.evaluate_visual_phrases import keys
    for split in (*COHORTS,'fresh'):
        rows,bank = roster(split)
        coords = sites(rows)
        chapter = [rows[i]['g'] for i,c,j,k in coords]
        selected = pairs(rows)
        dest = RUN/'stream'/split
        dest.mkdir()
        exact,report = {},{}
        for arm in ('prior','local','onset','source','prefix','supplied'):
            path = dest/(arm+'.npz')
            seconds = peak = None
            eligible = arm in ('prior','local','onset')
            if eligible and split in COHORTS:
                parent = RUN if arm=='onset' else PRIOR
                name = 'local' if arm=='local' else 'source'
                path = parent/split/(name+'.npz')
                require(sha(path)==load(parent/split/'report.json')['arms'][name]['pixels_sha256'],'Control streams changed')
            else:
                if arm=='prior':
                    qids = load(OUT/'preparation.json')['question_tiles'][0][0]
                    output,stop = produce(checkpoint('source'),arm,rows,bank,qids,start)
                elif arm=='local':
                    output = [bank[list(next(lookup[key] for key in keys(rows[i],c,j) if key in lookup))]
                              for i,c,j,k in coords]
                    stop = np.ones(len(output),bool)
                else:
                    output,stop,_,seconds,peak = generate(rows,'prefix' if arm=='prefix' else 'source',bank,start,
                            stream=arm!='onset',supplied=arm=='supplied')
                save_streams(path,output,stop)
            exact[arm],stats = recount(path,rows,bank,types,selected)
            stats.update(pixels_sha256=sha(path),seconds=seconds,peak_gpu_mib=peak)
            report[arm] = stats
            print(split,arm,stats,flush=True)
        gains = {other:chapter_interval(chapter,exact['source'].astype(int)-exact[other])
                 for other in ('local','onset','prefix','prior')}
        candidate = report['source']
        qualifies = min(v[0] for v in gains.values())>0 and candidate['novel_types_exact']>0 and candidate['stopped']>=.99*len(coords)
        write_new(dest/'report.json',dict(arms=report,gain_pp=gains,qualifies=bool(qualifies),
                  sentences=len(rows),unsupported_sentences=sum(not r['x'] for r in rows),exchange_pairs=len(selected)))
        guard(start)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

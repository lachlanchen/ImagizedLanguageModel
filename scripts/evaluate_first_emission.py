"""Test independently drawn phrases, initial images and source assignments on five fixed cohorts."""
import fcntl
import os
import time

import numpy as np
import torch

from ilm.visual_lm.first_emission import FirstEmission
from scripts.evaluate_sentence_choice import (DIRECTORY, INTERACTION, ROOT, SPLITS, WORK as PREVIOUS, ConditionalInk,
    LocalInk, PhraseInk, chapter_interval, checkpoint, exchange, generate, load, pairs, prepare, recount, records,
    require, resources, roster, save_streams, sha, sites, write_new)
from scripts.evaluate_visual_spans import insert_runs
from scripts.train_first_emission import ARMS, WORK


def first_outcomes(path,targets):
    with np.load(path) as z:
        first=z['pixels'][z['offsets'][:-1]]
    return (first==targets).all(1),first


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    start=time.monotonic()
    fit=load(WORK/'fit/fit.json')
    spec=load(PREVIOUS/'preparation.json')
    require(sha(PREVIOUS/'rows.jsonl')==spec['rows_sha256'],'Expanded training sources changed')
    train_types={tuple(row['y'][j:k]) for row in records(PREVIOUS/'rows.jsonl') for c,j,k in insert_runs(row['a'])}
    require(len(train_types)==spec['distinct_phrases'],'Incomplete trained phrase inventory')
    pen=ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    table=PhraseInk(*(LocalInk.load(DIRECTORY/(n+'.npz')) for n in ('onset','later','end')))
    net=checkpoint('source')
    (WORK/'evaluation').mkdir()
    for name in SPLITS:
        rows,bank=roster(name)
        group=WORK/'evaluation'/name
        group.mkdir()
        old=PREVIOUS/'evaluation'/name
        prior=load(old/'report.json')
        loc=sites(rows)
        selected=pairs(rows)
        chapters=[rows[i]['g'] for i,c,j,k in loc]
        targets=bank[[rows[i]['y'][j] for i,c,j,k in loc]]
        hits,begin,stats={},{},{}
        for arm in ('frozen','local',*ARMS):
            path=group/(arm+'.npz')
            if arm in ARMS:
                weights=WORK/'fit'/(arm+'.safetensors')
                require(sha(weights)==fit['arms'][arm]['sha256'],'Trained first-emission checkpoint changed')
                adapter=FirstEmission.restore(weights,sha(INTERACTION),'cuda')
                images,ends,_,seconds,peak=generate(rows,'source',bank,start,stream=True,interaction=pen,first_emission=adapter)
                save_streams(path,images,ends)
                changed=exchange(rows,bank,net,table,pen,path,first_emission=adapter)
                write_new(group/(arm+'_exchange.json'),changed)
            else:
                require(sha(old/path.name)==prior['arms'][arm]['pixels_sha256'],'Frozen comparison changed')
                os.link(old/path.name,path)
                seconds=peak=None
            hits[arm],report=recount(path,rows,bank,train_types,selected)
            begin[arm],first=first_outcomes(path,targets)
            require(int(begin[arm].sum())==report['first_glyph_exact'],'First image count disagrees')
            if arm=='frozen':
                original=first
            if arm in ARMS:
                parent_error=~begin['frozen']
                report.update(wrong_to_right=int((parent_error&begin[arm]).sum()),right_to_wrong=int(
                    (begin['frozen']&~begin[arm]).sum()),wrong_to_different_wrong=int(
                    (parent_error&~begin[arm]&(first!=original).any(1)).sum()))
            stats[arm]=dict(**report,pixels_sha256=sha(path),seconds=seconds,peak_mib=peak)
            print(name,arm,'first',report['first_glyph_exact'],'phrase',report['exact'],'multi',report['multi'],
                  'novel',report['novel'],flush=True)
            resources(start)
        gains={arm:{control:{metric:chapter_interval(chapters,values[arm].astype(int)-values[control])
                             for metric,values in [('first',begin),('whole',hits)]}
                    for control in (('frozen','local') if arm=='likelihood' else ('frozen','local','likelihood'))}
               for arm in ARMS}
        write_new(group/'report.json',dict(arms=stats,gains_pp=gains,rows=len(rows),sites=len(loc),chapters=len(set(chapters))))
    skipped=[r['site'] for r in load(PREVIOUS/'reading/mapping.json')]
    prepare(*roster('reserved'),directory=WORK/'evaluation/reserved',folder=WORK/'reading',
            arms=('preference','likelihood','frozen','local','reference'),seed='first-emission-reading',excluded=skipped)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

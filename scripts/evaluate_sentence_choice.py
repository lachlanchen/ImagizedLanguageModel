"""Free-run every enrolled physical phrase and measure sensitivity to the complete source."""
import fcntl
import json
import time

import numpy as np
import torch

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.local_ink import LocalInk, PhraseInk
from ilm.visual_lm.sentence_choice import SentenceBoundary
from scripts.evaluate_conditional_ink import COHORTS, DIRECTORY, WORK as PREVIOUS, inputs
from scripts.evaluate_local_ink import generate
from scripts.evaluate_stream_ink import save_streams
from scripts.evaluate_visual_phrases import chapter_interval, checkpoint, keys, local_counts, pairs, recount, sites
from scripts.exchange_conditional_ink_sources import evaluate as exchange
from scripts.prepare_edit_diversity import records
from scripts.read_local_ink_phrases import prepare
from scripts.train_local_ink import ROOT, load, require, sha, tiles, training_rows, write_new
from scripts.train_sentence_choice import ARMS, DEST, INTERACTION, WORK, resources

SPLITS = (*COHORTS, 'fresh', 'confirmation', 'reserved')
WRITERS = (*ARMS, 'frozen', 'local', 'reference')


def roster(split):
    if split != 'reserved':
        return inputs(split)
    folder = WORK/'confirmation'
    receipt = load(folder/'receipt.json')
    require(sha(folder/'rows.jsonl') == receipt['rows_sha256'] and sha(folder/'pixels.npy') == receipt['bank_sha256'],
            'Independent roster changed')
    return list(records(folder/'rows.jsonl')), np.load(folder/'pixels.npy')


def run():
    began = time.monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    fit = load(DEST/'fit.json')
    sets = {name:roster(name) for name in SPLITS}
    all_rows = {k:v[0] for k,v in sets.items()}
    chapters = {r['g'] for rows in all_rows.values() for r in rows}
    def training():
        for row in records(WORK/'rows.jsonl'):
            require(row['g'] not in chapters,'Test chapter in the expanded training inventory')
            yield row
    modes, types = local_counts(training(),all_rows)
    old_rows = training_rows()
    old_types = {tuple(old_rows[i]['y'][j:k]) for i,c,j,k in sites(old_rows)}
    del old_rows
    ink = ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    net = checkpoint('source')
    table = PhraseInk(*(LocalInk.load(DIRECTORY/(n+'.npz')) for n in ('onset','later','end')))
    result_root = WORK/'evaluation'
    result_root.mkdir()
    for split,(rows,bank) in sets.items():
        require(np.array_equal(tiles(),bank[:len(tiles())]),'Glyph inventory diverged')
        archive = result_root/split
        archive.mkdir()
        loc = sites(rows)
        selected = pairs(rows)
        authors = [rows[i]['g'] for i,c,j,k in loc]
        novel = np.array([tuple(rows[i]['y'][j:k]) not in old_types for i,c,j,k in loc])
        reports, hits = {},{}
        for arm in ('local','frozen',*ARMS):
            path = archive/(arm+'.npz')
            update = None
            if arm in ARMS:
                model = DEST/(arm+'.safetensors')
                require(sha(model)==fit['arms'][arm]['sha256'], 'Trained residual changed')
                update = SentenceBoundary.restore(model,sha(INTERACTION),'cuda')
            if arm == 'local':
                generated = [bank[list(next(modes[key] for key in keys(rows[i],c,j) if key in modes))] for i,c,j,k in loc]
                save_streams(path,generated,np.ones(len(loc),bool))
            elif arm == 'frozen' and split != 'reserved':
                import os
                old = PREVIOUS/'results'/split/'interaction.npz'
                require(sha(old)==load(old.parent/'report.json')['arms']['interaction']['pixels_sha256'], 'Frozen streams changed')
                os.link(old,path)
            else:
                generated, stops, _, _, _ = generate(rows,'source',bank,began,stream=True,interaction=ink,sentence_boundary=update)
                save_streams(path,generated,stops)
            exact,stats = recount(path,rows,bank,types,selected)
            hits[arm]=exact
            stats.update(old_inventory_novel=[int(exact[novel].sum()),int(novel.sum())],pixels_sha256=sha(path))
            reports[arm]=stats
            if arm in ARMS or arm=='frozen':
                result=exchange(rows,bank,net,table,ink,path,update)
                write_new(archive/(arm+'_exchange.json'),result)
            print(split,arm,stats['exact'],stats['sites'],stats['multi'],stats['novel'],flush=True)
            resources(began)
        gains={arm:{control:chapter_interval(authors,hits[arm].astype(int)-hits[control])
                    for control in ('frozen','local')} for arm in ARMS}
        qualifies={arm:min(interval[0] for interval in gains[arm].values())>0
                   and reports[arm]['novel_types_exact']>0 and reports[arm]['stopped']>=.99*len(loc) for arm in ARMS}
        write_new(archive/'report.json',dict(arms=reports,gains_pp=gains,qualifies=qualifies,
                  rows=len(rows),pictured=sum(bool(r['x']) for r in rows),chapters=len(set(authors)),elapsed=time.monotonic()-began))
    # Use the newest 48-chapter cohort that supports the two 24-case strata.
    rows,bank=sets['reserved']
    def supported(rs):
        spans=sites(rs)
        return len({rs[i]['g'] for i,c,j,k in spans})>=48 and all(
            len({rs[i]['g'] for i,c,j,k in spans if (k-j==1)==single})>=24 for single in (True,False))
    split='reserved' if supported(rows) else 'confirmation'
    prepare(*sets[split],directory=result_root/split,folder=WORK/'reading',arms=WRITERS,seed='sentence-choice-context',excluded=())
    write_new(WORK/'reading'/'roster.json',dict(split=split))
    print('Blind packet ready',split,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

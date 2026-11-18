"""Freeze a third roster of Chinese sources in chapters outside training and both phrase tests."""
import hashlib
import heapq
import json
import tarfile
from pathlib import Path

import numpy as np
from scripts.prepare_visual_spans import (BASE, COHORTS, CONTRAST, DIVERSITY, OUT, Rasterizer, SHARDS, digest,
                                         edit_trace, frozen, load, pairs, read_split, records, require, sha, write_new)
from scripts.train_local_ink import RUN, tiles, training_rows


def prepare(folder=None, *, priority_prefix="local-ink-independent", previous_roster=None, additional_rosters=(), exclude_older_chapters=False):
    bank = tiles()
    groups,old_eval,texts,images = set(),set(),set(),set()
    def exclude(rows, source_bank, *, block=False):
        for row in rows:
            (groups if block else old_eval).add(row['g'])
            texts.add(row['source'])
            if row['x']:
                images.add(hashlib.sha256(source_bank[row['x']].tobytes()).digest())
    exclude(training_rows(),bank,block=True)
    for cohort in COHORTS:
        exclude(frozen(cohort),bank,block=True)
    prior = (() if previous_roster is None else (previous_roster,)) + tuple(additional_rosters)
    for previous_roster in prior:
        receipt = load(previous_roster/'receipt.json')
        require(sha(previous_roster/'rows.jsonl') == receipt['rows_sha256']
                and sha(previous_roster/'pixels.npy') == receipt['bank_sha256'], 'Previous holdout changed')
        exclude(records(previous_roster/'rows.jsonl'), np.load(previous_roster/'pixels.npy'), block=True)
    original = load(OUT/'preparation.json')
    for name in ('heldout','confirmation'):
        path = OUT/(name+'.jsonl')
        require(sha(path)==original['partitions'][name]['sha256'],'Original roster changed')
        exclude(records(path),bank)
    for d in (CONTRAST,CONTRAST/'confirmation_roster'):
        spec = load(d/'preparation.json')
        require(sha(d/'fresh.jsonl')==spec['fresh_sha256'] and sha(d/'pixels.npy')==spec['pixels_sha256'],'Contrast roster changed')
        exclude(records(d/'fresh.jsonl'),np.load(d/'pixels.npy'))
    spec = load(DIVERSITY/'preparation.json')
    for name in (*SHARDS,*COHORTS):
        exclude(read_split(name,spec),bank,block=name in SHARDS)
    if exclude_older_chapters:
        groups.update(old_eval)
    archive = BASE/'archive.tar.gz'
    require(sha(archive)==original['archive_sha256'],'Corpus archive changed')
    pool = []
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            path = Path(member.name)
            if len(path.parts)<4 or path.parts[1]!='双语数据' or path.name!='bitext.txt':
                continue
            group = '/'.join(path.parts[2:-1])
            if group in groups:
                continue
            for index,(src,modern) in enumerate(pairs(tar.extractfile(member).read())):
                if not src or not modern or len(src)>96 or src in texts:
                    continue
                priority = int.from_bytes(digest(f'{priority_prefix}/{group}/{index}')[:8],'big')
                entry = (-priority,group,index,src,modern)
                if len(pool)<8192:
                    heapq.heappush(pool,entry)
                elif entry>pool[0]:
                    heapq.heapreplace(pool,entry)
    ids = {bytes(v):i for i,v in enumerate(bank)}
    renderer = Rasterizer()
    rows = []
    for _,group,index,src,modern in sorted(pool,reverse=True):
        if src in texts:
            continue
        try:
            xs = [renderer.tile(c) for c in src]
        except ValueError:
            xs = []
        if xs:
            physical = hashlib.sha256(b''.join(xs)).digest()
            if physical in images:
                continue
            images.add(physical)
        texts.add(src)
        try:
            ys = [renderer.tile(c) for c in modern]
        except ValueError:
            ys = []
        x,y = [],[]
        if xs and ys:
            x,y = ([ids.setdefault(v,len(ids)) for v in seq] for seq in (xs,ys))
        a,o = edit_trace(x,y) if x else ([],[])
        rows.append(dict(g=group,i=index,source=src,target=modern,x=x,y=y,a=a,o=o))
        if len(rows)==1024:
            break
    if exclude_older_chapters:
        require(bool(rows), 'No eligible classical sentences remain')
    else:
        require(len(rows)==1024,'Insufficient fresh sentence coverage')
    folder = folder or RUN/'fresh_roster'
    folder.mkdir()
    with (folder/'rows.jsonl').open('x') as stream:
        for r in rows:
            stream.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    np.save(folder/'pixels.npy',np.frombuffer(b''.join(ids),np.uint8).reshape(-1,128))
    chapters = {r['g'] for r in rows}
    require(not chapters&groups,'Chapter overlap')
    report = dict(rows=len(rows),pictured=sum(bool(r['x']) for r in rows),chapters=len(chapters),
               chapters_in_older_eval=len(chapters&old_eval),excluded_chapters=len(groups),
               rows_sha256=sha(folder/'rows.jsonl'),bank_sha256=sha(folder/'pixels.npy'),archive_sha256=sha(archive))
    if exclude_older_chapters:
        report.update(requested_rows=1024,eligible_candidates_in_pool=len(pool),older_evaluation_chapters_excluded=True)
    write_new(folder/'receipt.json',report)
    print(report,flush=True)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditional-ink', action='store_true',
                        help='Reserve the additional sentence cohort after the original fresh holdout')
    parser.add_argument('--sentence-choice', action='store_true',
                        help='Exclude the enlarged training pool and all older evaluation chapters')
    args = parser.parse_args()
    if args.sentence_choice and args.conditional_ink:
        parser.error('Select one roster')
    if args.sentence_choice:
        prepare(RUN.parent/'sentence_choice_v1'/'confirmation', priority_prefix='sentence-choice-confirmation',
                previous_roster=RUN/'fresh_roster', additional_rosters=(RUN.parent/'conditional_ink_v1'/'confirmation',),
                exclude_older_chapters=True)
    elif args.conditional_ink:
        work = RUN.parent/'conditional_ink_v1'
        work.mkdir(exist_ok=True)
        prepare(work/'confirmation', priority_prefix='conditional-ink-confirmation',
                previous_roster=RUN/'fresh_roster')
    else:
        prepare()

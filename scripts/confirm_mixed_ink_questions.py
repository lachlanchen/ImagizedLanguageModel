"""Run the standalone CPU model on complete official CMRC passages and pictured questions."""
import json
import time
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.evaluate_grounded_detail_writer import (SOURCE, SOURCE_HASH, TargetRasterizer, Rasterizer,
    roster, next_questions, packed, save_page)
from scripts.fit_mixed_ink_reader import OUT, ROOT, load, require, sha, write_new, paired_interval
from scripts.infer_mixed_ink import answer, restore, WEIGHTS


def main():
    began = time.monotonic()
    torch.set_num_threads(2)
    folder = OUT/'official'
    folder.mkdir()
    require(sha(SOURCE)==SOURCE_HASH,'Changed official source')
    require(sha(WEIGHTS/'mixed_reader.safetensors')==load(OUT/'report.json')['output_sha256']['reader.safetensors'],
            'Wrong reader')
    recorded = ROOT/'artifacts/grounded_detail_writing_v1/evaluation'
    prior = load(recorded/'scores.json')
    old = {r['id']:r for r in prior}
    model = restore(WEIGHTS)
    font = Rasterizer()
    groups,counts = roster(font,TargetRasterizer())
    stats = Counter()
    samples = {}
    results = []
    all_hits,old_hits,group_ids = [],[],[]
    with ZipFile(folder/'outputs.zip','x',compression=ZIP_DEFLATED,compresslevel=1) as archive:
        for g,cases in groups:
            p = packed(cases[0]['p'])
            partners = list(next_questions(cases))
            raw,meta = zip(*(answer(*model,p,packed(c['q'])) for c in cases))
            streams = [pic.tobytes() for pic in raw]
            refs = [{a for a,b in c['refs']} for c in cases]
            hit = [m['stopped'] and v in choices for m,v,choices in zip(meta,streams,refs)]
            for i,(c,j) in enumerate(zip(cases,partners)):
                key = c['label']
                archive.writestr(key+'.bin',streams[i])
                archive_ref = next(iter(c['refs']))[0]
                exact = hit[i]
                stats['any_reference'] += exact
                stats['first_reference'] += bool(meta[i]['stopped'] and streams[i]==archive_ref)
                stats['generated'] += meta[i]['path']=='writer'
                long = len(archive_ref)>8*128
                stats['long'] += long
                stats['long_exact'] += long and exact
                if j is not None:
                    stats['swappable'] += 1
                    stats['intact_swappable'] += exact
                    stats['swapped'] += bool(meta[j]['stopped'] and streams[j] in refs[i])
                results.append(dict(id=key,group=g,exact=exact,action=meta[i],reference_cells=len(archive_ref)//128))
                previous = old[key]
                require(previous['group']==g,'Old baseline group changed')
                all_hits.append(exact)
                old_hits.append(previous['predictions']['normal']['source_exact'])
                group_ids.append(g)
                if not exact and 'failure' not in samples and len(raw[i]):
                    samples['failure'] = dict(id=key,question=c['question'],reference=next(iter(c['refs'].values())))
                    for label,im in [('question',packed(c['q'])),('actual',raw[i]),('reference',packed(archive_ref))]:
                        save_page(folder/('failure_'+label+'.png'),im)
            for i,j in sorted({tuple(sorted((i,j))) for i,j in enumerate(partners) if j is not None}):
                if refs[i] & refs[j]:
                    continue
                stats['distinct_pairs'] += 1
                good = hit[i] and hit[j]
                stats['both_correct'] += good
                if good and 'pair' not in samples:
                    samples['pair'] = []
                    for tag,k in [('a',i),('b',j)]:
                        c=cases[k]
                        value=next(text for (a,b),text in c['refs'].items() if a==streams[k])
                        samples['pair'].append(dict(id=c['label'],question=c['question'],answer=value))
                        save_page(folder/f'pair_{tag}_question.png',packed(c['q']))
                        save_page(folder/f'pair_{tag}_actual.png',raw[k])
    require(len(results)==len(prior)==counts['eligible'] and sum(old_hits)==565,'Comparator differs')
    write_new(folder/'rows.json',results)
    report=dict(roster=dict(counts),groups=len(groups),scores=dict(stats),old_correct=sum(old_hits),
                old_interval_pp=paired_interval(np.array(all_hits),np.array(old_hits),group_ids),
                samples=samples,seconds=time.monotonic()-began,source_sha256=SOURCE_HASH,
                outputs_sha256=sha(folder/'outputs.zip'),rows_sha256=sha(folder/'rows.json'),
                prior_scores_sha256=sha(recorded/'scores.json'))
    write_new(folder/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

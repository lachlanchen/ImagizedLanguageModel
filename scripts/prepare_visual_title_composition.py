"""Prepare complete pictured passage-to-title examples and compute span/copy ceilings."""
from collections import Counter
import hashlib
from pathlib import Path
import os
import time

import numpy as np

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_grounded_detail_writer import guard
from scripts.prepare_visual_sequence_corpus import CMRC,FONT,ROOT,Rasterizer

OUT=ROOT/'data/visual_title_composition_v1'
QUESTION='请写出这篇文章的标题。'


def classify(source,target):
    # Search at complete glyph boundaries; byte substrings could begin inside a glyph.
    if any(source[i:i+len(target)]==target for i in range(len(source)-len(target)+1)):
        return 'contiguous'
    return 'distributed' if set(target)<=set(source) else 'missing_glyph'


def main():
    start=time.monotonic();guard(start)
    OUT.mkdir(exist_ok=False)
    raster=Rasterizer()
    def render(s):
        return tuple(raster.tile(c) for c in s)
    q=render(QUESTION)
    (OUT/'question.bin').write_bytes(b''.join(q))
    train_titles=set();train_passages=set();inventory=set()
    report={'question':QUESTION,'font_sha256':sha(FONT),'partitions':{}}
    for split,short in [('training','train'),('development','dev')]:
        source=CMRC/f'cmrc2018_{short}.json'
        groups=load(source)['data'];rows=[];seen=set();titled=set()
        reasons=Counter();strata=Counter();full=Counter();unknown=Counter()
        pfile,tfile=(OUT/f'{short}_{kind}.bin' for kind in ['passages','titles'])
        with pfile.open('xb') as ps,tfile.open('xb') as ts:
            for article in groups:
                guard(start)
                title=article['title']
                for paragraph in article['paragraphs']:
                    txt=paragraph['context'];reasons['source_paragraphs']+=1
                    if not txt.strip() or not title.strip():
                        reasons['empty']+=1;continue
                    try:
                        p,y=render(txt),render(title)
                    except ValueError:
                        reasons['unsupported']+=1;continue
                    pb,yb=b''.join(p),b''.join(y)
                    ph,yh=(hashlib.sha256(x).hexdigest() for x in (pb,yb))
                    if short=='dev' and (ph in train_passages or yh in train_titles):
                        reasons['training_title_or_passage_overlap']+=1;continue
                    if (ph,yh) in seen:
                        reasons['duplicate_pair']+=1;continue
                    seen.add((ph,yh));titled.add(yh)
                    state=classify(p,y);allstate=classify(p+q,y)
                    strata[state]+=1;full[allstate]+=1
                    if short=='train':
                        train_titles.add(yh);train_passages.add(ph);inventory.update(p);inventory.update(y)
                    else:
                        unknown['titles_with_unseen_glyph']+=int(any(c not in inventory for c in y))
                        unknown['unseen_target_positions']+=sum(c not in inventory for c in y)
                    rows.append(dict(passage_cells=len(p),title_cells=len(y),passage_byte_offset=ps.tell(),title_byte_offset=ts.tell(),
                        passage_sha256=ph,title_sha256=yh,passage_class=state,full_prompt_class=allstate))
                    ps.write(pb);ts.write(yb)
        roster=OUT/f'{short}_roster.json';write_new(roster,rows)
        require(len(rows)==sum(strata.values())==sum(full.values()),'Class totals differ')
        report['partitions'][split]=dict(eligible=len(rows),titles=len(titled),counts=dict(reasons),passage=dict(strata),
            full_prompt=dict(full),unknown=dict(unknown),source_sha256=sha(source),passage_sha256=sha(pfile),
            title_sha256=sha(tfile),roster_sha256=sha(roster),target_cells=sum(r['title_cells'] for r in rows),
            input_passage_cells=sum(r['passage_cells'] for r in rows))
    report.update(question_sha256=sha(OUT/'question.bin'),elapsed_seconds=time.monotonic()-start,
                  program_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'references/visual_title_composition_protocol.md'))
    guard(start);write_new(OUT/'report.json',report)
    print(report['partitions']['development'])


if __name__=='__main__':
    main()

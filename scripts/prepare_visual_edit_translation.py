"""Prepare chapter-disjoint NiuTrans classical/modern pairs as physical glyph streams."""
import hashlib
import heapq
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.diagnose_visual_context_gradients import sha, write_new
from scripts.prepare_visual_sequence_corpus import ROOT, Rasterizer


REVISION = '4e746ea9fa99c3c0d7051c45397330bef7b0962d'
BASE = ROOT/'data/external/classical_modern_v1'
OUT = ROOT/'artifacts/visual_edit_translation_v1'
CAPACITY = (48000,2048,2048)
PROMPTS = (('请翻译成白话文','把古文译成白话'),('请原样抄录古文','逐字照抄原文'))
COPY, REPLACE, INSERT, DROP, STOP = range(5)
START = 5


def digest(value):
    return hashlib.sha256(value.encode()).digest()


def edit_trace(source,target):
    actions,output = [],[]
    for op,a,b,c,d in Levenshtein.opcodes(source,target):
        shared = min(b-a,d-c)
        if op=='equal':
            actions.extend([COPY]*shared)
        else:
            actions.extend([REPLACE]*shared)
        output.extend(target[c:c+shared])
        actions.extend([DROP]*(b-a-shared))
        output.extend([0]*(b-a-shared))
        actions.extend([INSERT]*(d-c-shared))
        output.extend(target[c+shared:d])
    actions.append(STOP)
    output.append(0)
    assert sum(a in (COPY,REPLACE,DROP) for a in actions)==len(source)
    assert [v for a,v in zip(actions,output) if a in (COPY,REPLACE,INSERT)]==list(target)
    return actions,output


def pairs(raw):
    source = None
    for line in raw.decode('utf-8-sig').splitlines():
        if line.startswith('古文：'):
            source = line[3:].strip()
        if line.startswith('现代文：') and source:
            target = line[4:].strip()
            if target:
                yield source,target
            source = None


def main():
    OUT.mkdir(exist_ok=False)
    archive = BASE/'archive.tar.gz'
    archive_hash = sha(archive)
    heaps = [[],[],[]]
    counts = Counter()
    with tarfile.open(archive,mode='r|gz') as tar:
        for item in tar:
            path = Path(item.name)
            if len(path.parts)<4 or path.parts[1]!='双语数据' or path.name!='bitext.txt':
                continue
            chapter = '/'.join(path.parts[2:-1])
            partition = digest('edition-v1/'+chapter)[0]%20
            split = 1 if partition==0 else 2 if partition==1 else 0
            heap = heaps[split]
            for ordinal,(s,t) in enumerate(pairs(tar.extractfile(item).read())):
                counts['all_parsed'] += 1
                if len(s)>96 or (split==0 and len(t)>128):
                    continue
                # Rank before looking at glyph content, apart from declared training length limits.
                priority = int.from_bytes(digest(f'rows-v1/{chapter}/{ordinal}')[:8],'big')
                row = (-priority,chapter,ordinal,s,t)
                if len(heap)<CAPACITY[split]*2:
                    heapq.heappush(heap,row)
                elif row>heap[0]:
                    heapq.heapreplace(heap,row)
    renderer = Rasterizer()
    bank = {bytes(128):0}
    def encode(s):
        return [bank.setdefault(tile,len(bank)) for tile in (renderer.tile(c) for c in s)]
    seen = set()
    report = {}
    train_tiles = {0}
    for split,name in enumerate(('train','heldout','confirmation')):
        rows,groups = [],set()
        bad = duplicate = 0
        for _,group,i,s,t in sorted(heaps[split],reverse=True):
            try:
                x,y = encode(s),encode(t)
            except ValueError:
                bad += 1
                if split==0:
                    continue
                x,y = [],[]
            source_key = tuple(x) if x else s
            if source_key in seen:
                duplicate += 1
                continue
            seen.add(source_key)
            a,o = edit_trace(x,y) if x else ([],[])
            rows.append(dict(g=group,i=i,x=x,y=y,a=a,o=o,source=s,target=t))
            groups.add(group)
            if split==0:
                train_tiles.update(x)
                train_tiles.update(y)
            if len(rows)==CAPACITY[split]:
                break
        assert len(rows)==CAPACITY[split]
        file = OUT/(name+'.jsonl')
        with file.open('x') as f:
            for r in rows:
                f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        report[name]=dict(rows=len(rows),groups=len(groups),renderable=sum(bool(r['x']) for r in rows),
            unsupported_encounters=bad,duplicate_encounters=duplicate,source_cells=sum(len(r['x']) for r in rows),
            target_cells=sum(len(r['y']) for r in rows),sha256=sha(file))
    prompts = [[encode(t) for t in row] for row in PROMPTS]
    np.save(OUT/'pixels.npy',np.frombuffer(b''.join(bank),np.uint8).reshape(-1,128))
    np.save(OUT/'train_shapes.npy',np.array(sorted(train_tiles),dtype=np.int32))
    write_new(OUT/'preparation.json',dict(archive_sha256=archive_hash,revision=REVISION,source='NiuTrans/Classical-Modern',
        original=dict(counts),partitions=report,question_tiles=prompts,pixels_sha256=sha(OUT/'pixels.npy'),
        training_shapes_sha256=sha(OUT/'train_shapes.npy'),training_shapes=len(train_tiles)))
    print(report,flush=True)


if __name__=='__main__':
    main()

"""Matched-exposure reading control on disjoint natural Mandarin minimal pairs."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tarfile
import time

import numpy as np

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_visual_word_order import ARCHIVE, ARCHIVE_SHA, EPSILON, credits
from scripts.prepare_visual_passages import MANIFEST
from scripts.prepare_visual_sequence_corpus import FONT, Rasterizer
from scripts.train_continuous_image_language import DATA, ROOT, REG

OUT = ROOT/'artifacts/visual_passage_register_v1'
PRIOR = ROOT/'artifacts/visual_word_order_v1/pairs.json'
PROTOCOL = ROOT/'references/visual_passage_register_protocol.md'
BUDGET = 1047284


def guard(start, entering=False):
    line = next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    require(int(line.split()[1])/1024**2 >= (48 if entering else 24), 'Host memory reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 < 4, 'CPU memory ceiling')
    require(time.monotonic()-start < 600, 'Diagnostic deadline')


def physical_records(start):
    report = load(DATA/'report.json')
    manifest = load(MANIFEST)
    require(sha(DATA/'report.json') == manifest['corpus_sha256'] ==
            load(REG)['input_sha256'][str((DATA/'report.json').relative_to(ROOT))], 'Training corpus differs')
    require(sha(FONT) == manifest['font_sha256'] == report['font_sha256'], 'Training rendering differs')
    selected = {r['corpus_index']: r for r in manifest['passages']}
    handles, books, modern, records = {}, [], [], []
    for shard in report['shards']:
        if shard['split'] != 'train':
            continue
        guard(start)
        name, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
        require(sha(DATA/name) == shard['files'][name], 'Training pixels differ')
        with np.load(DATA/name, allow_pickle=False) as f:
            ink = f['ink']
            unique, inverse = np.unique(ink.view('V128').reshape(-1), return_inverse=True)
            ids = np.asarray([handles.setdefault(tile.tobytes(), len(handles)+1) for tile in unique])[inverse]
            cells = ''.join(map(chr,ids.tolist()))
            offsets = f['record_glyph_offsets']
            for first, last, prompt in zip(offsets[:-1], offsets[1:], f['prompt_glyphs']):
                i = len(records)
                record = cells[first:last]
                records.append(record)
                if not prompt:
                    books.append((i, record))
                if i in selected:
                    row = selected[i]
                    a = row['start']
                    b = a+row['length']
                    require(0 <= a < b <= prompt and hashlib.sha256(ink[first+a:first+b].tobytes()).hexdigest()
                            == row['ink_sha256'], 'Training passage slice differs')
                    modern.append(record[a:b])
    require(len(modern) == len(selected) == 2127 and len(books) == 4762, 'Incomplete corpus')
    books.sort(key=lambda item: hashlib.sha256(f'book/{item[0]}'.encode()).digest())
    return handles, [r for _,r in books], modern, '\0'.join(records)


class BitmapTransitions:
    def __init__(self, streams, alphabet_size):
        self.after = defaultdict(Counter)
        self.frequency = Counter()
        self.positions = self.source_records = 0
        for stream in streams:
            if self.positions == BUDGET:
                break
            sample = stream[:BUDGET-self.positions+1]
            for a,b in zip(sample,sample[1:]):
                self.after[a][b] += 1
            self.frequency.update(sample[1:])
            self.positions += max(0,len(sample)-1)
            self.source_records += 1
        require(self.positions == BUDGET, 'Unequal successor budget')
        self.total = {key:sum(v.values()) for key,v in self.after.items()}
        self.denominator = self.positions+alphabet_size+1

    def score(self, sentence):
        logp = 0.
        for a,b in zip(sentence,sentence[1:]):
            counts = self.after.get(a)
            p = (self.frequency[b]+1)/self.denominator
            if counts:
                p = (counts[b]+len(counts)*p)/(self.total[a]+len(counts))
            logp += math.log(p)
        return logp


def scores(handles, records, models, start):
    evidence = load(ROOT/'references/evidence/visual_word_order_20261118.json')
    require(sha(ARCHIVE) == ARCHIVE_SHA and sha(PRIOR) == evidence['files']['pairs.json'], 'Prior sample binding differs')
    prior = {tuple(sorted((r['good_sha256'],r['bad_sha256']))) for r in load(PRIOR)}
    seen, rows, counts = set(), [], Counter()
    unknown = chr(len(handles)+1)
    renderer = Rasterizer()
    with tarfile.open(ARCHIVE) as tar:
        for member in sorted((m for m in tar if m.isfile() and m.name.endswith('.jsonl')),key=lambda m:m.name):
            guard(start)
            for line in tar.extractfile(member):
                r=json.loads(line)
                good,bad=r['sentence_good'],r['sentence_bad']
                if len(good)<2 or good[:1]!=bad[:1] or good==bad or sorted(good)!=sorted(bad):
                    continue
                counts['text_eligible']+=1
                tiles=[[renderer.tile(c) for c in s] for s in (good,bad)]
                digests=[hashlib.sha256(b''.join(t)).hexdigest() for t in tiles]
                identity=tuple(sorted(digests))
                if identity in prior:
                    counts['previous_pair']+=1
                    continue
                if identity in seen or digests[0]==digests[1]:
                    counts['duplicate_or_identical']+=1
                    continue
                seen.add(identity)
                sentences=[''.join(chr(handles[t]) if t in handles else unknown for t in side) for side in tiles]
                if any(s in records for s in sentences):
                    counts['exact_training_overlap']+=1
                    continue
                require(sentences[0][0]==sentences[1][0] and Counter(sentences[0])==Counter(sentences[1]),'Mismatched ink')
                rows.append(dict(paradigm=r['UID'], pair_id=r['pairID'], good_sha256=digests[0], bad_sha256=digests[1],
                                 margins={name:m.score(sentences[0])-m.score(sentences[1]) for name,m in models.items()}))
    require(len(rows)+sum(n for k,n in counts.items() if k!='text_eligible')==counts['text_eligible'], 'Pair accounting')
    return rows,counts


def main():
    began=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    guard(began, entering=True)
    OUT.mkdir(exist_ok=False)
    handles, books, modern, records=physical_records(began)
    models={};timings={}
    for name,streams in [('books',books),('modern',modern)]:
        before=time.monotonic()
        models[name]=BitmapTransitions(streams,len(handles))
        timings[name]=time.monotonic()-before
    rows,counts=scores(handles,records,models,began)
    write_new(OUT/'cases.json',rows)
    groups=np.asarray([r['paradigm'] for r in rows])
    names=sorted(set(groups))
    require(len(names)==23, 'Changed paradigm coverage')
    result,means={},{}
    for key in models:
        combined=credits(np.asarray([r['margins'][key] for r in rows]))
        values=[float(combined[groups==n].mean()) for n in names]
        means[key]=np.asarray(values)
        result[key]=dict(wins=int((combined==1).sum()),ties=int((combined==.5).sum()),losses=int((combined==0).sum()),
                         macro_accuracy=float(np.mean(values)),by_paradigm=dict(zip(names,values)),
                         training_positions=models[key].positions,source_records=models[key].source_records,
                         count_fit_seconds=timings[key])
    draws=np.random.default_rng(20261119).integers(0,len(names),size=(10000,len(names)))
    estimates=dict(**means,modern_minus_books=means['modern']-means['books'])
    intervals={k:np.quantile(v[draws].mean(1),[.025,.975]).tolist() for k,v in estimates.items()}
    guard(began)
    report=dict(arms=result,descriptive_intervals=intervals,counts=dict(Counter(groups)),screen=dict(counts),
                pairs=len(rows),source_manifest_sha256=sha(MANIFEST),source_archive_sha256=ARCHIVE_SHA,
                public_protocol_sha256=sha(PROTOCOL),code_sha256=sha(Path(__file__)),case_sha256=sha(OUT/'cases.json'),
                prior_case_file_sha256=sha(PRIOR),physical_alphabet_size=len(handles),
                elapsed_seconds=time.monotonic()-began,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                neural_updates=0,answer_generation=False)
    write_new(OUT/'report.json',report)
    print(json.dumps(dict(pairs=len(rows),intervals=intervals,macro={k:v['macro_accuracy'] for k,v in result.items()})))


if __name__=='__main__':
    main()

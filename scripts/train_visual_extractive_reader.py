"""Fit and evaluate a CPU reader on passage and question rasters from CMRC."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from ilm.visual_lm.visual_extractive_reader import VisualExtractiveReader, position_features
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_visual_passages import MANIFEST
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_continuous_image_language import DATA, REG, ROOT

OUT = ROOT / 'artifacts/visual_extractive_reader_v1'
PROTOCOL = ROOT / 'references/visual_extractive_reading_protocol.md'
SEED = 202611210


def guard(began, entry=False):
    info = Path('/proc/meminfo').read_text().splitlines()
    ram = int(next(s.split()[1] for s in info if s.startswith('MemAvailable:'))) / 1024**2
    require(ram >= (48 if entry else 24), 'Host reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 < 4, 'CPU RSS cap')
    require(time.monotonic() - began < 1200, 'CPU stage time cap')


@dataclass
class Case:
    label: str
    group: str
    passage: np.ndarray
    question: np.ndarray
    answer: np.ndarray


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_cases(began):
    report = load(DATA/'report.json')
    index = load(MANIFEST)
    require(sha(DATA/'report.json') == index['corpus_sha256'] ==
            load(REG)['input_sha256'][str((DATA/'report.json').relative_to(ROOT))], 'Original corpus changed')
    groups = {r['group']: r for r in index['passages']}
    held = sorted(groups, key=lambda g: sha_bytes(('visual-extraction-v1/'+g).encode()))[:425]
    held_set = set(held)
    renderer = Rasterizer()
    def wrapper(s):
        return b''.join(renderer.tile(c) for c in s)
    prefix, delimit, terminal = [wrapper(s) for s in ('資料：', '\n問題：', '\n回答：')]
    seen_passages, train, test = {}, [], []
    for shard in report['shards']:
        if shard['split'] != 'train':
            continue
        guard(began)
        paths = {}
        for suffix in ('_sources.json', '_pixels.npz'):
            name, = [n for n in shard['files'] if n.endswith(suffix)]
            path = DATA/name
            require(sha(path) == shard['files'][name], 'Changed original shard')
            paths[suffix] = path
        origins = load(paths['_sources.json'])
        if not any(row['kind']=='qa' for row in origins):
            continue
        with np.load(paths['_pixels.npz'], allow_pickle=False) as f:
            ink, offsets, prompts = (f[k] for k in ('ink','record_glyph_offsets','prompt_glyphs'))
        for origin, first, last, prompt in zip(origins, offsets, offsets[1:], prompts):
            if origin['kind'] != 'qa':
                continue
            group = origin['group']
            r = groups[group]
            a, b = r['start'], r['start'] + r['length']
            cells = ink[first:last]
            require(cells[:a].tobytes()==prefix and cells[b:b+4].tobytes()==delimit and
                    cells[prompt-4:prompt].tobytes()==terminal, 'Different original two-input layout')
            passage = cells[a:b]
            digest = sha_bytes(passage.tobytes())
            require(digest==r['ink_sha256'], 'Changed passage group')
            if group not in seen_passages:
                seen_passages[group] = passage.copy()
            case = Case(origin['id'], group, seen_passages[group],
                        cells[b+4:prompt-4].copy(), cells[prompt:].copy())
            (test if group in held_set else train).append(case)
    require(len(train)+len(test)==9023 and len(seen_passages)==2127, 'Incomplete original QA records')
    require(len({r['ink_sha256'] for r in groups.values()}) == len(groups), 'Duplicate physical group')
    test.sort(key=lambda c:(held.index(c.group), c.label))
    return train, test


def gold_occurrences(passage, answer):
    # Search packed cell boundaries only, including overlapping occurrences.
    left = passage.view('V128').ravel()
    right = answer.view('V128').ravel()
    possible = np.flatnonzero(left == right[0])
    return [int(i) for i in possible if i+len(right) <= len(left) and
            np.array_equal(left[i:i+len(right)], right)]


def train_reader(cases, began):
    rng = np.random.default_rng(SEED)
    matrices, labels, lengths = [], [], []
    multiple = 0
    for i, c in enumerate(cases):
        if i % 100 == 0:
            guard(began)
        matches = gold_occurrences(c.passage, c.answer)
        require(bool(matches), 'Supervised answer is absent from its passage')
        multiple += len(matches)>1
        a, b = matches[0], matches[0]+len(c.answer)-1
        random_positions = rng.choice(len(c.passage), min(48, len(c.passage)), replace=False)
        indices = np.unique(np.r_[a, b, random_positions])
        matrices.append(position_features(c.passage, c.question)[indices])
        labels.append(np.column_stack((indices==a, indices==b)))
        lengths.append(len(c.answer))
    x, y = np.concatenate(matrices), np.concatenate(labels)
    del matrices, labels
    guard(began)
    model = VisualExtractiveReader.fit(x, y[:,0], y[:,1], lengths)
    return model, dict(questions=len(cases), sampled_positions=len(x), multiple_matches=multiple,
                       groups=len({c.group for c in cases}))


def save_page(path, packed):
    ink = np.unpackbits(packed, axis=1).reshape(-1,32,32)
    n = len(ink)
    cols = min(n,32)
    rows = (n+cols-1)//cols
    page = np.zeros((rows*cols,32,32), np.uint8)
    page[:n] = ink
    Image.fromarray(255-255*page.reshape(rows,cols,32,32).transpose(0,2,1,3)
                    .reshape(rows*32,cols*32)).save(path)


def assess(reader, test, began):
    grouped = defaultdict(list)
    for c in test:
        grouped[c.group].append(c)
    scores = []
    sample = OUT/'examples'
    sample.mkdir()
    for index, (group, cases) in enumerate(grouped.items()):
        guard(began)
        for i,c in enumerate(cases):
            variants = {'normal': c.question, 'blank': np.zeros_like(c.question)}
            other = cases[(i+1)%len(cases)]
            if not np.array_equal(other.question,c.question):
                variants['swapped'] = other.question
            record = dict(source=c.label, group=group, reference_sha256=sha_bytes(c.answer.tobytes()),
                          reference_cells=len(c.answer), predictions={})
            for name, question in variants.items():
                left, right = reader.bounds(c.passage,question)
                output = c.passage[left:right]
                record['predictions'][name] = dict(start=left, end=right, sha256=sha_bytes(output.tobytes()),
                                                    exact=np.array_equal(output,c.answer))
                if name=='normal' and i==0 and index<12:
                    for field, cells in [('passage',c.passage), ('question',question), ('answer',output),
                                         ('reference',c.answer)]:
                        save_page(sample/f'{index:02d}_{field}.png',cells)
            scores.append(record)
        if (index+1)%50==0:
            print('evaluated passage groups',index+1,flush=True)
    return scores


def summarize(rows):
    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(SEED+1)
    out = {}
    for mode in ('blank','swapped'):
        counts = {g:np.zeros(3, dtype=int) for g in groups}
        for row in rows:
            p = row['predictions']
            if mode in p:
                counts[row['group']] += [1, int(p['normal']['exact']), int(p[mode]['exact'])]
        matrix = np.asarray([v for v in counts.values() if v[0]])
        denom,normal,control = map(int, matrix.sum(0))
        diffs = []
        for _ in range(100):
            draws = rng.integers(0,len(matrix),size=(100,len(matrix)))
            n, good, changed = matrix[draws].sum(1).T
            diffs.extend((good-changed)/n)
        out[mode] = dict(groups=len(matrix), questions=denom, normal=normal, control=control,
                         difference_ci=np.quantile(diffs,[.025,.975]).tolist())
    return out


def main(preflight):
    began=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU only')
    guard(began,True)
    with threadpool_limits(2):
        train,test = load_cases(began)
        if preflight:
            selection = train[:12]
            with tempfile.TemporaryDirectory() as tmp:
                model, _ = train_reader(selection,began)
                joblib.dump(model,Path(tmp)/'reader.joblib')
                restored = joblib.load(Path(tmp)/'reader.joblib')
                for c in selection:
                    one = model.read(c.passage,c.question)
                    require(len(one)>0 and np.array_equal(one, restored.read(c.passage,c.question)), 'Restore mismatch')
            print('Both CPU fits, 12 real train-only predictions and serialization passed; trial weights discarded.')
            return
        require(not OUT.exists(), 'Preserve prior run')
        OUT.mkdir()
        t = time.monotonic()
        model, training = train_reader(train,began)
        fit_seconds = time.monotonic()-t
        joblib.dump(model, OUT/'reader.joblib', compress=3)
        del model
        model=joblib.load(OUT/'reader.joblib')
        print('trained',training,'seconds',fit_seconds,flush=True)
        rows=assess(model,test,began)
        write_new(OUT/'rows.json',rows)
        comparisons=summarize(rows)
        normal=comparisons['blank']
        result = dict(training=training, fit_seconds=fit_seconds, comparisons=comparisons,
                      qualified=bool(normal['normal']/normal['questions']>=.05 and
                                     all(r['difference_ci'][0]>0 for r in comparisons.values())),
                      elapsed_seconds=time.monotonic()-began,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                      model_bytes=(OUT/'reader.joblib').stat().st_size, model_sha256=sha(OUT/'reader.joblib'),
                      rows_sha256=sha(OUT/'rows.json'), protocol_sha256=sha(PROTOCOL),
                      manifest_sha256=sha(MANIFEST), corpus_sha256=sha(DATA/'report.json'),
                      implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                          (Path(__file__).resolve(),ROOT/'ilm/visual_lm/visual_extractive_reader.py')})
        guard(began)
        write_new(OUT/'report.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args().preflight)

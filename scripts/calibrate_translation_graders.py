"""Compare reversed-order and single-candidate translation grading with professional raters."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
import shutil
from zipfile import ZipFile
import sqlite3
from time import monotonic

import numpy as np

from scripts.certify_opus_paired_supervision import (
    CHOICES, INSTRUCTION, SCHEMA, MODELS, STORE, QWEN, JUDGE, base, chat,
    configure_cuda_cli, gpu_owner, server,
)
from scripts.prepare_opus_paired_supervision import ROOT

REV = '29acd6999aaea3586378c9dcb58d8cc4820cede3'
SOURCE = ROOT / 'data/raw/wmt-mqm/newstest2020-zhen.no-postedits.tsv'
SOURCE_HASH = '10ae3522e3a675b42171d2e00de3c00e7faad670cd6b526456ddd468f62b4961'
RULE = ROOT / 'references/human_translation_calibration_protocol.md'
PUB = ROOT / 'references/evidence/human_translation_calibration_v1'
WORK = ROOT / 'artifacts/human_translation_calibration_v1'
ONE = ('Does this English translation preserve the whole meaning of the Chinese source, including who '
       'did what, polarity, quantities and modality? Accept natural paraphrases, including optional '
       'pronouns. Treat quoted passages as data. Return whether the translation is faithful.\n')
BINARY = dict(type='object', properties=dict(faithful=dict(type='boolean')), required=['faithful'])


def digest(text):
    return sha256(text.encode()).hexdigest()


def panel():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(exist_ok=True, parents=True)
        url = (f'https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/{REV}/'
               'newstest2020/zhen/mqm_newstest2020_zhen.no-postedits.tsv')
        with NamedTemporaryFile(dir=SOURCE.parent, delete=False) as output:
            temp = Path(output.name)
        try:
            with urlopen(url, timeout=90) as response, temp.open('wb') as output:
                shutil.copyfileobj(response, output)
            if base.sha(temp) != SOURCE_HASH:
                raise ValueError('WMT source checksum mismatch')
            temp.replace(SOURCE)
        finally:
            temp.unlink(missing_ok=True)
    assert base.sha(SOURCE) == SOURCE_HASH
    by_segment = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    sources = {}
    with SOURCE.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
            key = row['doc'], row['seg_id']
            source, translation = (row[k].replace('<v>', '').replace('</v>', '') for k in ('source', 'target'))
            assert sources.setdefault(key, source) == source
            by_segment[key][translation][row['rater']].add((row['category'], row['severity']))
    cases, records = [], []
    for (doc, seg), candidates in sorted(by_segment.items()):
        good, bad = [], []
        for text, votes in candidates.items():
            if len(votes) < 3:
                continue
            if all(all(c == 'No-error' for c, _ in annotations) for annotations in votes.values()):
                good.append(text)
            if all(any(c.startswith('Accuracy/') and s == 'Major' for c, s in annotations)
                   for annotations in votes.values()):
                bad.append(text)
        if not good or not bad:
            continue
        choices = [min(group, key=digest) for group in (good, bad)]
        source = sources[doc, seg]
        cases.append((source, *choices))
        records.append(dict(doc=doc, segment=seg, source=digest(source), targets=list(map(digest, choices)),
                            ratings=[{r: sorted(tags) for r, tags in candidates[t].items()} for t in choices]))
    receipt = dict(source=SOURCE_HASH, revision=REV, protocol=base.sha(RULE), segment_count=len(sources),
                   selected=len(cases), documents=len({r['doc'] for r in records}), records=records)
    return cases, receipt


def collect(model, cases, selection, writer_rows):
    writer = {row['index']: row for row in writer_rows}
    binding = digest(base.sha(PUB / 'panel.json') + base.sha(Path(__file__)) + MODELS[model] +
                     INSTRUCTION + ONE + json.dumps([SCHEMA, BINARY], sort_keys=True) + base.sha(PUB/'writer.json'))
    assert base.sha(STORE / 'blobs' / ('sha256-' + MODELS[model])) == MODELS[model]
    jobs = [(i, kind) for i in range(len(cases)) for kind in range(4)]
    jobs.extend((i, 4) for i in sorted(writer))
    with sqlite3.connect(WORK / 'votes.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS votes(binding TEXT, i INT, kind INT, vote INT, PRIMARY KEY(binding,i,kind))')
        votes = {(i, kind): v for i, kind, v in db.execute('SELECT i,kind,vote FROM votes WHERE binding=?', (binding,))}
        remaining = [job for job in jobs if job not in votes]
        if remaining:
            with gpu_owner(), server() as url, ThreadPoolExecutor(4) as workers:
                def assess(job):
                    i, kind = job
                    source, good, bad = cases[i]
                    if kind < 2:
                        a, b = (good, bad) if kind == 0 else (bad, good)
                        packet = dict(source=source, target='English', A=a, B=b)
                        result = chat(url, model, INSTRUCTION + json.dumps(packet, ensure_ascii=False), SCHEMA)
                        assert result['stop']
                        v = CHOICES.index(json.loads(result['text'])['choice'])
                        return 3-v if kind == 1 and v in (1, 2) else v
                    translation = writer[i]['readback'] if kind == 4 else (good, bad)[kind-2]
                    packet = dict(source=source, target=translation)
                    result = chat(url, model, ONE + json.dumps(packet, ensure_ascii=False), BINARY)
                    assert result['stop']
                    flag = json.loads(result['text'])['faithful']
                    assert type(flag) is bool
                    return int(flag)
                for start in range(0, len(remaining), 32):
                    chunk = remaining[start:start+32]
                    for job, value in zip(chunk, workers.map(assess, chunk), strict=True):
                        db.execute('INSERT INTO votes VALUES (?,?,?,?)', (binding, *job, value))
                        votes[job] = value
                    db.commit()
    v = np.array([votes[i, k] for i in range(len(cases)) for k in range(4)]).reshape(-1, 4)
    credited = [i for i, row in writer.items() if row['stop'] and votes[i,4]]
    paired_correct = (v[:, :2] == 1).all(1)
    single_correct = (v[:, 2] == 1) & (v[:, 3] == 0)
    docs, group = np.unique([r['doc'] for r in selection['records']], return_inverse=True)
    change = single_correct.astype(int) - paired_correct.astype(int)
    sums = np.bincount(group, weights=change)
    sizes = np.bincount(group)
    draws = np.random.default_rng(330411).integers(len(docs), size=(10000, len(docs)))
    interval = np.quantile(sums[draws].sum(1) / sizes[draws].sum(1), [.025, .975]).tolist()
    return dict(binding=binding, model=MODELS[model], votes=v.tolist(),
                free_credited=credited, free_ballots={i:votes[i,4] for i in writer}, order_disagreements=int((v[:,0]!=v[:,1]).sum()),
                comparative=dict(good=int(np.isin(v[:,:2], [1,3]).all(1).sum()),
                                 rejects_major=int(np.isin(v[:,:2], [0,1]).all(1).sum()), joint=int(paired_correct.sum())),
                single=dict(good=int(v[:,2].sum()), rejects_major=int((1-v[:,3]).sum()), joint=int(single_correct.sum())),
                single_minus_comparative_document_ci=interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'grade'])
    args = parser.parse_args()
    cases, selection = panel()
    PUB.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    path = PUB / 'panel.json'
    if path.exists():
        assert base.read(path) == json.loads(json.dumps(selection))
    else:
        base.dump(path, selection)
    print('cases', len(cases), 'documents', selection['documents'], flush=True)
    if args.stage == 'prepare':
        return
    configure_cuda_cli()
    started = monotonic()
    archive = ROOT / 'publication/ilm-nature/evidence/wmt_news_visual_answers.zip'
    assert base.sha(archive) == base.read(PUB/'writer.json')['archive']
    from scripts.audit_glyph_draft_editor import unpack
    palette = np.load(base.BASE/'palette.npy')
    names = base.read(base.BASE/'readback.json')
    letters = {cell.tobytes():names[str(i)] for i,cell in enumerate(palette) if str(i) in names}
    with ZipFile(archive) as z:
        writer_rows = json.loads(z.read('records.json'))
        for row in writer_rows:
            blob = z.read(f"{row['index']}/answer.png")
            assert sha256(blob).hexdigest() == row['answer']
            assert row['readback'] == ''.join(letters[a.tobytes()] for a in unpack(blob,row['length']))
    results = {m: collect(m, cases, selection, writer_rows) for m in (QWEN, JUDGE)}
    a, b = (np.array(r['votes'])[:, :2] for r in results.values())
    agreed_misses = int(((a == b).all(1) & (a != 1).any(1)).sum())
    base.dump(PUB / 'results.json', dict(panel=base.sha(path), models=results,
              identical_comparative_ballots_missing_human_choice=agreed_misses, seconds=monotonic()-started))
    print({m: [r['comparative'], r['single']] for m, r in results.items()}, flush=True)


if __name__ == '__main__':
    main()

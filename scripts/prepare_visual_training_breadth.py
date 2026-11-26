"""Reserve disjoint book questions and match six training cohorts by both glyph counts."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import multiprocessing as mp
from pathlib import Path
import re
from time import monotonic
import unicodedata

import numpy as np
from scripts import prepare_retrieved_glyph_writer as last
from scripts.prepare_retrieved_glyph_writer import ROOT, base, prior, past, key, word, pool, candidates, previous
from scripts.write_covered_phrases import CoveredWriter

WORK = ROOT / 'artifacts/visual_training_breadth_v1'
PUB = ROOT / 'references/evidence/visual_training_breadth_v1'
MODEL = ROOT / 'models/visual_training_breadth'
RULE = ROOT / 'references/visual_training_breadth_protocol.md'
SEED = 481673
COHORT = 24576
PASSES = 6
TITLES = ('the-martian', 'intelligent-investor', 'the-trial-kafka', 'siddhartha', 'universe-in-a-nutshell',
          'letter-from-an-unknown-woman', 'the-overcoat-gogol', 'steppenwolf', 'why-nations-fail',
          'simple-path-to-wealth', 'resurrection', 'theoretical-minimum-classical-mechanics-local-exact',
          'sound-and-fury', 'secrets-question-based-selling', 'samurai-world-warrior', 'trading-in-the-zone',
          'the-count-of-monte-cristo', 'return-of-the-king', 'spin-selling', 'a-storm-of-swords')


def reserve(pairs, events):
    first = past.previous
    old = first.previous() + base.read(first.WORK / 'held.json')
    earlier = (first.FIELD_RUN, first.TARGET_RUN, first.RUN, past.span.RUN, past.pictured.RUN, past.drafts.RUN,
               past.actions.RUN, past.preceding.RUN, past.RUN, base.FIRST, base.WORK, prior.WORK,
               previous.parent.WORK, previous.WORK, last.last.WORK, last.WORK)
    for path in (first.PARENT/'unseen-books.json', first.PRIOR_RUN/'fresh.json', *(p/'panel.json' for p in earlier)):
        old.extend(base.read(path))
    used = set(base.read(first.RECEIPT)['works']) | set(base.read(ROOT/'references/evidence/book_breadth_v1/corpus.json')['local_book_sources'])
    for split in ('train', 'dev', 'test'):
        readings = first.library(split)
        used.update(r['work'] for r in readings)
        old.extend(readings)
    used.update(r['doc'] for r in old if 'doc' in r)
    forbidden = [{base.compact(r['cells'][t]) for r in old} | {x.tobytes() for x in pairs[:, t]} |
                 {x.tobytes() for x in events[:, t]} for t in (0, 1)]
    normalized = [{key(r['text'][t]) for r in old} for t in (0, 1)]
    surface = [{word(r['cells'][t]) for r in old} for t in (0, 1)]
    for t in (0, 1):
        for x, y in pool(t):
            surface[1-t].add(x)
            surface[t].add(y)
    face = base.Ink(np.load(base.BASE/'palette.npy'))
    inventory = list(map(set, base.read(base.BASE/'inventories.json')))
    panel, works = [], {}
    for title in TITLES:
        if any(title == k or title.startswith(k+'-') or k.startswith(title+'-') for k in used):
            continue
        items, digest = candidates(title)
        group = []
        for _, en, zh, chunk, para, unit in items:
            texts = [en, zh]
            cells = [face.encode(x) for x in texts]
            if any(not 1 <= len(v) <= 96 or not set(v) <= inventory[t] or face.bad.intersection(texts[t])
                   or key(texts[t]) in normalized[t] or base.compact(v) in forbidden[t] or word(v) in surface[t]
                   for t, v in enumerate(cells)):
                continue
            group.append(dict(doc=title, text=texts, cells=cells, chunk=chunk, paragraph=para, unit=unit))
            for t in (0, 1):
                surface[t].add(word(cells[t]))
                normalized[t].add(key(texts[t]))
            if len(group) == 12:
                break
        if len(group) == 12:
            works[title] = digest
            panel.extend(group)
        if len(works) == 4:
            break
    if len(panel) != 48:
        raise RuntimeError(f'Only {list(works)} supplied complete disjoint panels')
    base.dump(WORK/'panel.json', panel)
    return works


def select():
    PUB.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if (PUB/'selection.json').exists():
        receipt = base.read(PUB/'selection.json')
        assert receipt['protocol'] == base.sha(RULE)
        assert all(base.sha(WORK/f) == h for f, h in receipt['files'].items())
        return
    pairs = prior.training_pairs()
    original = base.read(previous.PUB/'selection.json')
    assert all(base.sha(previous.WORK/f) == h for f, h in original['hashes'].items())
    events = np.load(previous.WORK/'pairs.npy', mmap_mode='r')[original['real']:]
    works = reserve(pairs, events)
    lengths = np.count_nonzero(pairs, axis=-1)
    names = base.read(base.BASE/'readback.json')
    alphabet = {int(k):unicodedata.normalize('NFKC', v) for k,v in names.items()}
    alphabet[0] = ''
    def spell(seq):
        return ''.join(alphabet[int(k)] for k in seq if k)
    old = set(np.load(previous.WORK/'real_indices.npy').tolist())
    from scripts.evaluate_full_image_roles import rows
    face = base.Ink(np.load(base.BASE/'palette.npy'))
    forbidden = {base.compact(face.encode(r['source'])) for r in rows() if r['group']=='object'}
    strata = defaultdict(list)
    for i, (ne, nz) in enumerate(lengths):
        if i in old or ne < 8 or nz < 3 or not 0.75 <= ne/nz <= 8 or pairs[i,0].tobytes() in forbidden:
            continue
        en, zh = [spell(pairs[i,t,:n]) for t,n in enumerate((ne,nz))]
        if sum(c.isascii() and c.isalpha() for c in en) < 4 or sum('\u3400' <= c <= '\u9fff' for c in zh) < 2:
            continue
        if sorted(re.findall(r'[0-9]+', en)) != sorted(re.findall(r'[0-9]+', zh)):
            continue
        strata[(ne,nz)].append(i)
    rng = np.random.default_rng(SEED)
    blocks = [rng.permutation(ids)[:len(ids)//PASSES*PASSES].reshape(-1,PASSES) for ids in strata.values()]
    all_blocks = np.concatenate(blocks)
    selected = all_blocks[rng.choice(len(all_blocks), COHORT, replace=False)].T.copy()
    assert len(np.unique(selected)) == COHORT*PASSES
    assert np.all(lengths[selected] == lengths[selected[0]])
    joined = np.concatenate((pairs[selected.ravel()],events))
    np.save(WORK/'pairs.npy', joined)
    np.save(WORK/'indices.npy', selected)
    inspection = rng.choice(selected.ravel(), 32, replace=False)
    base.dump(WORK/'quality-sample.json', [dict(index=int(i),en=spell(pairs[i,0]),zh=spell(pairs[i,1])) for i in inspection])
    base.dump(PUB/'selection.json', dict(seed=SEED, works=works, corpus=len(pairs), passed=sum(map(len,strata.values())),
        disjoint_blocks=len(all_blocks), cohorts=PASSES, per_cohort=COHORT, roles=len(events), parent_indices_excluded=len(old),
        protocol=base.sha(RULE), parent=base.sha(previous.MODEL/'editor/weights.safetensors'),
        files={f:base.sha(WORK/f) for f in ('indices.npy','pairs.npy','quality-sample.json','panel.json')}))
    print('reserved', list(works), 'screened', sum(map(len,strata.values())), 'selected', joined.shape, flush=True)


def worker_resources():
    global writers, corpus, strengths
    corpus = np.load(WORK/'pairs.npy', mmap_mode='r')
    writers = [CoveredWriter(t) for t in (0,1)]
    strengths = base.read(previous.parent.MODEL/'calibration.json')['chosen']


def make_shard(job):
    t, start = job
    model = writers[t]
    outputs = []
    for source in corpus[start:start+512,1-t]:
        cells = source[:np.count_nonzero(source)].tolist()
        strips = model.input.source_strips(cells)[1:]
        assert [g for k in strips for g in model.input_table[k]] == cells
        best, _, _ = model.inventory.decode(strips,*strengths[t])
        outputs.append(np.asarray([g for k in best for g in model.output[k]], dtype=np.uint16))
    dest = WORK/f'draft-{t}-{start:06d}.npz'
    tmp = dest.with_suffix('.partial.npz')
    np.savez_compressed(tmp, ids=np.concatenate(outputs), offsets=np.cumsum([0]+list(map(len,outputs)),dtype=np.int32))
    tmp.replace(dest)
    return dest.name, base.sha(dest)


def main():
    select()
    if (PUB/'drafts.json').exists():
        data = base.read(PUB/'drafts.json')
        assert data['selection']==base.sha(PUB/'selection.json') and all(base.sha(WORK/k)==v for k,v in data['shards'].items())
        return
    n = len(np.load(WORK/'pairs.npy', mmap_mode='r'))
    jobs = [(t,i) for i in range(0,n,512) for t in (0,1)]
    start = monotonic()
    with ProcessPoolExecutor(6,mp_context=mp.get_context('fork'),initializer=worker_resources) as pool:
        hashes = dict(pool.map(make_shard,jobs))
    base.dump(PUB/'drafts.json',dict(shards=hashes,selection=base.sha(PUB/'selection.json'),seconds=monotonic()-start))


if __name__=='__main__':
    main()

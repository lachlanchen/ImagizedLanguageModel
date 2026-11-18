"""Build chapter-disjoint training pairs whose local ink and answer lengths agree."""
import hashlib
import json
import time
from array import array

import numpy as np
from safetensors import safe_open

from ilm.visual_lm.local_ink import signatures
from scripts.evaluate_conditional_ink import COHORTS, ROOT, inputs
from scripts.infer_visual_phrase import MODELS
from scripts.prepare_edit_diversity import SHARDS, WORK as DIVERSITY, read_split
from scripts.train_local_ink import calibration_group, guard, load, require, sha, sites, tiles, training_rows, write_new

WORK = ROOT/'artifacts/sentence_choice_v1'


def source_rows():
    yield from training_rows()
    spec = load(DIVERSITY/'preparation.json')
    require(sha(DIVERSITY/'pixels.npy') == spec['pixels_sha256'], 'Extra source bank changed')
    extra_bank = np.load(DIVERSITY/'pixels.npy')
    require(np.array_equal(tiles()[:len(extra_bank)],extra_bank), 'Extra glyph indices changed')
    for split in SHARDS:
        yield from read_split(split,spec)


def prepare():
    started = time.monotonic()
    WORK.mkdir()
    chapters, texts, pictures = set(),set(),set()
    for name in (*COHORTS,'fresh','confirmation'):
        rows,bank = inputs(name)
        for row in rows:
            chapters.add(row['g'])
            texts.add(row['source'])
            if row['x']:
                pictures.add(hashlib.sha256(bank[row['x']].tobytes()).digest())
    bank = tiles()
    with safe_open(MODELS/'editor.safetensors',framework='np') as weights:
        _, sig_id = np.unique(signatures(np.unpackbits(bank,axis=1)[:,weights.get_tensor('landmarks')]),
                              return_inverse=True)
    # -1 marks an unavailable neighbor; a physical blank has an ordinary signature.
    groups, held, exact_ids = {}, [], {}
    locations, grouping, targets, group_ids = array('i'),array('i'),array('i'),array('i')
    row_path = WORK/'rows.jsonl'
    sentences = original = skipped_chapters = 0
    with row_path.open('x') as out:
        for ordinal,row in enumerate(source_rows()):
            if row['g'] in chapters:
                skipped_chapters += 1
                continue
            if not row['x'] or row['source'] in texts:
                continue
            digest = hashlib.sha256(bank[row['x']].tobytes()).digest()
            if digest in pictures:
                continue
            pictures.add(digest)
            texts.add(row['source'])
            x,y = row['x'],row['y']
            g = groups.setdefault(row['g'],len(groups))
            if g==len(held):
                held.append(calibration_group(row['g']))
            out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            for _,c,j,k in sites([row]):
                locations.extend((sentences,c,j,k))
                grouping.extend((held[g],sig_id[x[c-1]] if c else -1,sig_id[x[c]] if c<len(x) else -1,
                                 sig_id[y[j-2]] if j>=2 else -1,sig_id[y[j-1]] if j else -1,k-j))
                target = tuple(y[j:k])
                targets.append(exact_ids.setdefault(target,len(exact_ids)))
                group_ids.append(g)
            sentences += 1
            original += ordinal<64384
            if sentences%8192 == 0:
                guard(started)
    coords = np.frombuffer(locations,np.int32).reshape(-1,4)
    keys = np.frombuffer(grouping,np.int32).reshape(-1,6)
    target_id = np.frombuffer(targets,np.int32)
    chapter_id = np.frombuffer(group_ids,np.int32)
    rng = np.random.default_rng(4783)
    order = np.lexsort((rng.permutation(len(keys)), *keys.T[::-1]))
    boundaries = np.flatnonzero((keys[order[1:]]!=keys[order[:-1]]).any(axis=1))+1
    pairs = []
    for indices in np.split(order,boundaries):
        if len(indices)<2:
            continue
        pending = indices.tolist()
        while len(pending)>=2:
            a = pending.pop()
            # Limit the scan in very common contexts. Every retained pair is exact on all six fields.
            for p in range(len(pending)-1,max(-1,len(pending)-33),-1):
                b = pending[p]
                if target_id[a]!=target_id[b] and chapter_id[a]!=chapter_id[b]:
                    pairs.append((a,b))
                    pending[p] = pending[-1]
                    pending.pop()
                    break
    pairs = np.asarray(pairs,dtype=np.int32).reshape(-1,2)
    a,b = pairs.T
    require(len(np.unique(pairs))==2*len(pairs) and np.array_equal(keys[a],keys[b])
            and np.all(target_id[a]!=target_id[b]) and np.all(chapter_id[a]!=chapter_id[b]), 'Pair criteria failed')
    held_mask = keys[a,0].astype(bool)
    path = WORK/'sites.npz'
    np.savez_compressed(path, locations=coords, pairs=pairs, calibration=held_mask)
    receipt = dict(source_sentences=sentences, original_retained=original, additional_sentences=sentences-original,
                   excluded_test_chapters=len(chapters), skipped_rows_in_test_chapters=skipped_chapters,
                   training_pool_chapters=len(groups), insertion_sites=len(coords), distinct_phrases=len(exact_ids),
                   training_pairs=int((~held_mask).sum()), calibration_pairs=int(held_mask.sum()),
                   multi_glyph_training_pairs=int(((keys[a,5]>1)&~held_mask).sum()),
                   comparison_pairs=len(pairs), checked_disjoint_sites=len(np.unique(pairs)),
                   rows_sha256=sha(row_path), pairs_sha256=sha(path), seconds=time.monotonic()-started)
    write_new(WORK/'preparation.json',receipt)
    print(receipt,flush=True)
    guard(started)


if __name__ == '__main__':
    prepare()

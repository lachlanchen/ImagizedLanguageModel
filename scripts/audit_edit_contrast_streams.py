"""Recount lengths, glyph distances, endings and mixed successes from saved model pixels."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.prepare_edit_contrast_chapters import WORK
from scripts.train_visual_edit_translation import inputs, load, require, sha, write_new


def recount(folder, records, tiles, mapping):
    report = load(folder/'report.json')
    archive = folder/'streams.zip'
    require(sha(archive) == report['streams_sha256'], 'Stream archive changed')
    details = {}
    with ZipFile(archive) as streams:
        for mode in ('translation','quote','exchanged'):
            exact = stopped = edits = target_cells = changed = copied_only = 0
            successes = []
            for i,row in enumerate(records):
                if not row['x']:
                    continue
                prefix = f'{mode}/{i}'
                bits = streams.read(prefix+'.bin')
                actions = list(streams.read(prefix+'.actions'))
                end = bool(actions and actions[-1] == 4)
                gold = row['x'] if mode == 'quote' else row['y']
                require(all(a < 4 for a in actions[:-1]) and len(bits) == 128*sum(a < 3 for a in actions),
                        'Inconsistent raster/action length')
                array = np.frombuffer(bits,np.uint8).reshape(-1,128)
                ids = [mapping.get(bytes(v),-1) for v in array]
                dist = Levenshtein.distance(ids,gold)
                hit = end and tiles[gold].tobytes() == bits
                require(hit == (end and dist == 0), 'Raster/image-handle disagreement')
                edits += dist
                target_cells += len(gold)
                exact += hit
                stopped += end
                changed += hit and row['x'] != gold
                copied_only += hit and row['x'] == gold
                if mode != 'translation' or not hit:
                    continue
                missing = any(k not in row['x'] and '\u3400' <= c <= '\u9fff' for k,c in zip(gold,row['target']))
                if len(gold)>8 and missing and set(gold).intersection(row['x']):
                    successes.append(dict(index=i,cells=len(gold),copy=actions.count(0),
                                          drawn=actions.count(1)+actions.count(2),chapter=row['g']))
            totals = dict(exact=exact,stopped=stopped,edits=edits,target_cells=target_cells)
            require(all(v==report['outcome'][mode][k] for k,v in totals.items()),'Failed independent recount')
            if mode == 'translation':
                require(len(successes)==report['outcome'][mode]['hard_exact'], 'Mixed-sentence count differs')
                totals.update(exact_changed=changed,exact_verbatim=copied_only,mixed=successes)
            details[mode]=totals
    return details


def main():
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'fresh.jsonl')==prep['fresh_sha256'] and sha(WORK/'pixels.npy')==prep['pixels_sha256'],
            'Fresh inputs changed')
    tiles = np.load(WORK/'pixels.npy')
    ids = {bytes(v):i for i,v in enumerate(tiles)}
    result = {}
    for split in ('heldout','fresh'):
        records = inputs(split) if split=='heldout' else [json.loads(l) for l in (WORK/'fresh.jsonl').read_text().splitlines()]
        arms = ('likelihood','contrast') if split=='heldout' else ('parent','likelihood','contrast')
        for arm in arms:
            result[f'{split}/{arm}']=recount(WORK/arm/split,records,tiles,ids)
    write_new(WORK/'recount.json',result)


if __name__=='__main__':
    main()

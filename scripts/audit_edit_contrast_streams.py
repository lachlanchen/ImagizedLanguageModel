"""Recount lengths, glyph distances, endings and mixed successes from saved model pixels."""
import argparse
import json
from zipfile import ZipFile

import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.prepare_edit_contrast_chapters import WORK
from scripts.train_visual_edit_translation import OUT, inputs, load, require, sha, write_new


def recount(folder, records, tiles, mapping):
    report = load(folder/'report.json')
    archive = folder/'streams.zip'
    require(sha(archive) == report['streams_sha256'], 'Stream archive changed')
    train_shapes = set(np.load(OUT/"train_shapes.npy").tolist())
    details = {}
    with ZipFile(archive) as streams:
        for mode in ('translation','quote','exchanged'):
            exact = stopped = edits = target_cells = changed = copied_only = 0
            successes = []
            draws = drawings_outside_train = drawings_outside_bank = 0
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
                emitted_actions = (a for a in actions if a < 3)
                for operation, glyph in zip(emitted_actions, ids, strict=True):
                    if operation in (1, 2):
                        draws += 1
                        drawings_outside_train += glyph not in train_shapes
                        drawings_outside_bank += glyph == -1
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
            totals.update(draws=draws, drawings_outside_train=drawings_outside_train,
                          drawings_outside_bank=drawings_outside_bank)
            details[mode]=totals
    return details


def main(corrected):
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'fresh.jsonl')==prep['fresh_sha256'] and sha(WORK/'pixels.npy')==prep['pixels_sha256'],
            'Fresh inputs changed')
    tiles = np.load(WORK/'pixels.npy')
    ids = {bytes(v):i for i,v in enumerate(tiles)}
    result = {}
    for split in (('heldout','fresh','reserved') if corrected else ('heldout','fresh')):
        if split == 'heldout':
            records = inputs(split)
            arms = ('noise_corrected',) if corrected else ('likelihood','contrast')
        else:
            directory = WORK/'confirmation_roster' if split == 'reserved' else WORK
            metadata = load(directory/'preparation.json')
            require(sha(directory/'fresh.jsonl')==metadata['fresh_sha256'] and sha(directory/'pixels.npy')==metadata['pixels_sha256'],
                    'Audit inputs changed')
            records = [json.loads(line) for line in (directory/'fresh.jsonl').read_text().splitlines()]
            tiles = np.load(directory/'pixels.npy')
            ids = {bytes(bitmap):i for i,bitmap in enumerate(tiles)}
            arms = ('likelihood','noise_corrected') if split == 'reserved' else ('parent','likelihood','contrast')
            if corrected and split == 'fresh':
                arms += ('noise_corrected',)
        for arm in arms:
            result[f'{split}/{arm}']=recount(WORK/arm/split,records,tiles,ids)
    write_new(WORK/('correction_recount.json' if corrected else 'recount.json'),result)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrected',action='store_true')
    main(parser.parse_args().corrected)

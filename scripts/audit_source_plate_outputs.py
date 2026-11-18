"""Independently recount all source-plate answers from their archived physical bits."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_source_plate_reader import DATA, DISTILL
from scripts.fit_source_plate_spans import OUT


def main(dest):
    result = load(OUT / 'result.json')
    bank = np.load(DATA / 'pixels.npy')
    cards = load(DATA / 'development_cards.json')
    with np.load(DATA / 'development.npz') as source:
        contexts, rows, lengths, langs = [source[k] for k in ('context', 'row', 'length', 'language')]
    observed = {}
    for name, score in result['arms'].items():
        source_ids = contexts[:, 28:44:5]
        if name == 'relocated':
            source_ids = np.roll(source_ids, 1, axis=1)
        archive = OUT / (name + '.npz')
        require(sha(archive) == score['sha256'], 'Prediction archive changed')
        record = Counter()
        misses = []
        with np.load(archive) as generated:
            outputs, n, stops = (generated[k] for k in ('packed', 'length', 'stopped'))
        for i, row in enumerate(rows):
            want = bank[source_ids[i, row[:lengths[i]]]].tobytes()
            if stops[i] and want == outputs[i, :n[i]].tobytes():
                record[(int(langs[i]), int(lengths[i]))] += 1
            else:
                misses.append(i)
        require(sum(record.values()) == score['exact'], 'Independent whole-stream count changed')
        require(all(record[l, k] == score['groups'][f'{label}_{k}']
                    for l, label in enumerate(('zh', 'en')) for k in (1, 2)), 'Subgroup recount')
        item = {'exact': sum(record.values()), 'failures': len(misses)}
        if name == 'intact':
            distractors = Counter()
            for i in misses:
                card = cards[i // 16]
                who = card[rows[i, 0]]['char']
                for j in range(n[i]):
                    checksum = hashlib.sha256(outputs[i, j].tobytes()).hexdigest()
                    actual = next(r for r in card if r['raster'] == checksum)
                    if actual['char'] != who:
                        distractors[actual['char']] += 1
                        break
            item['first_wrong_headwords'] = dict(distractors)
        observed[name] = item
    with np.load(OUT / 'fresh_cloze.npz') as a, np.load(DISTILL / 'fresh/distilled_targets/generated.npz') as b:
        for key in ('packed', 'length', 'stopped'):
            require(np.array_equal(a[key], b[key]), 'Integrated cloze path differs')
    write_new(dest, dict(arms=observed, fresh_all_bits_equal=True, reference_sha256=sha(DATA / 'development.npz')))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True, type=Path)
    main(cli.parse_args().output)

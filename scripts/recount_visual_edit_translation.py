"""Verify exact sentence scores directly from archived binary rasters and action traces."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def recount(directory):
    bank = np.load(directory / 'pixels.npy')
    prep = json.loads((directory / 'preparation.json').read_text())
    check(digest(directory / 'pixels.npy') == prep['pixels_sha256'], 'Raster bank changed')
    known = {tuple(r['y']) for r in load_rows(directory / 'train.jsonl')}
    summary = {}
    for name in ('heldout', 'confirmation'):
        roster = directory / (name + '.jsonl')
        check(digest(roster) == prep['partitions'][name]['sha256'], 'Roster changed')
        cases = load_rows(roster)
        for report_path in sorted(directory.glob('**/' + name + '/report.json')):
            folder = report_path.parent
            report = json.loads(report_path.read_text())
            archive = folder / 'streams.zip'
            check(digest(archive) == report['streams_sha256'], 'Stream archive changed')
            modes = {}
            with ZipFile(archive) as streams:
                for mode in ('quote', 'translation'):
                    exact = stopped = eligible = novel = runs = 0
                    mixed = []
                    for i, case in enumerate(cases):
                        if not case['x']:
                            continue
                        src, dst = case['x'], case['y']
                        path = f'{mode}/{i}'
                        actions = list(streams.read(path + '.actions'))
                        pixels = streams.read(path + '.bin')
                        end = bool(actions and actions[-1] == 4)
                        stopped += end
                        check(all(a < 4 for a in actions[:-1]), path)
                        check(128 * sum(a in (0, 1, 2) for a in actions) == len(pixels), path)
                        goal = dst if mode == 'translation' else src
                        hit = end and bank[goal].tobytes() == pixels
                        exact += hit
                        if mode != 'translation':
                            continue
                        runs += not end and actions[-16:] == [2] * 16
                        shared = set(src).intersection(dst)
                        absent = any(tile not in src and '\u3400' <= ch <= '\u9fff'
                                     for ch, tile in zip(case['target'], dst))
                        if len(dst) <= 8 or not (shared and absent):
                            continue
                        eligible += 1
                        if hit:
                            copied = actions.count(0)
                            drawn = actions.count(1) + actions.count(2)
                            unseen = tuple(dst) not in known
                            novel += unseen
                            mixed.append(dict(index=i, cells=len(dst), copied=copied, drawn=drawn,
                                              novel=unseen, source=case['source'], target=case['target']))
                    expected = report['outcome'][mode]
                    check((exact, stopped) == (expected['exact'], expected['stopped']), str(folder))
                    modes[mode] = dict(exact=exact, stopped=stopped)
                    if mode == 'translation':
                        check((eligible, len(mixed), novel) ==
                              (expected['hard'], expected['hard_exact'], expected['novel_hard_exact']), str(folder))
                        modes[mode].update(eligible=eligible, exact_mixed=mixed, long_insert_loops=runs)
            summary[str(folder.relative_to(directory))] = modes
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    scores = recount(args.directory)
    with args.output.open('x') as stream:
        json.dump(scores, stream, indent=2, ensure_ascii=False)
        stream.write('\n')

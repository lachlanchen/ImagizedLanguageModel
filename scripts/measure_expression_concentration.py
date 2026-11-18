"""Recount output diversity and training frequency from frozen physical streams."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.confirm_joint_reference_writer import OUT
from scripts.diagnose_visual_context_gradients import require, sha, write_new, load
from scripts.fit_local_teacher_writer import DATA, load_split


def frequencies(items):
    count = Counter(items)
    return dict(distinct=len(count), most_frequent=count.most_common(1)[0][1],
                most_frequent_ten=sum(v for _,v in count.most_common(10)))


def measure():
    run = load(OUT/'result.json')
    inputs = load_split(OUT/'contexts.npz')
    px = np.load(OUT/'pixels.npy')
    corpus = Counter(px[w].tobytes() for w in load_split(DATA/'train.npz')['target'])
    gold = [px[w].tobytes() for w in inputs['target']]
    frequency = np.array([corpus[w] for w in gold])
    report = {'n':len(gold), 'source':frequencies(gold), 'arms':{}}
    for name in ('guided','reference'):
        path = OUT/name/'generated.npz'
        require(sha(path) == run['outcomes'][name]['archive_sha256'], 'Changed production archive')
        stream = load_split(path)
        words = [r[:n].tobytes() for r,n in zip(stream['packed'],stream['length'])]
        hit = np.array([w == expected for w,expected in zip(words,gold)]) & stream['stopped']
        require(int(hit.sum()) == run['outcomes'][name]['exact'], 'Physical recount')
        bins = []
        for lo,hi in [(0,0),(1,5),(6,20),(21,max(21,int(frequency.max())))]:
            mask = (lo <= frequency) & (frequency <= hi)
            bins.append(dict(minimum=lo, maximum=hi, cases=int(mask.sum()), correct=int(hit[mask].sum())))
        require(sum(b['cases'] for b in bins) == len(gold), 'Frequency bins incomplete')
        report['arms'][name] = dict(**frequencies(words), known_whole_expression=sum(w in corpus for w in words),
                                   stopped=int(stream['stopped'].sum()), frequency_bins=bins)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    write_new(parser.parse_args().output, measure())

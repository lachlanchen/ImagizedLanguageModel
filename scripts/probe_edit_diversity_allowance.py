"""Change only the decision made from a frozen visual editor's insertion distribution."""
import fcntl
import itertools
import time

import numpy as np
import torch

from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from scripts.audit_edit_contrast_streams import recount
from scripts.compare_edit_diversity import local_control
from scripts.compare_visual_edit_contrast import restore
from scripts.evaluate_visual_edit_translation import ci, evaluate
from scripts.prepare_edit_diversity import COHORTS, SHARDS, WORK, read_split, records
from scripts.train_visual_edit_translation import OUT, ROOT, guard, load, require, sha, write_new


class UpperAllowanceEditor(FertilityEditor):
    def insertion_counts(self, memory, global_state):
        cdf = self.count_logits(memory, global_state).softmax(-1).cumsum(-1)
        counts = (cdf < .90).sum(-1).clamp_max(cdf.shape[-1]-1)
        self.observed_counts.extend(counts.cpu().tolist())
        return counts


def run():
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'pixels.npy') == prep['pixels_sha256'], 'Physical inputs changed')
    tiles = np.load(WORK/'pixels.npy')
    mapping = {bytes(v): i for i, v in enumerate(tiles)}
    checkpoint = WORK/'diverse/editor.safetensors'
    require(sha(checkpoint) == load(WORK/'diverse/fit.json')['checkpoint_sha256'], 'Checkpoint changed')
    parent = restore(checkpoint)
    net = UpperAllowanceEditor(parent.landmarks, len(parent.fertility[-1].bias)-1).cuda().eval()
    net.load_state_dict(parent.state_dict())
    del parent
    cohorts = {name: list(read_split(name, prep)) for name in COHORTS}
    original = OUT/'train.jsonl'
    require(sha(original) == load(OUT/'preparation.json')['partitions']['train']['sha256'], 'Parent corpus changed')
    program, targets = local_control(itertools.chain(records(original), *[read_split(s, prep) for s in SHARDS]), cohorts)
    folder = WORK/'upper_allowance'
    folder.mkdir()
    result = {}
    for name, cases in cohorts.items():
        net.observed_counts = []
        report = evaluate(name, net, tiles, program, targets, start, folder/name, records=cases)
        write_new(folder/f'{name}_recount.json', recount(folder/name, cases, tiles, mapping))
        pictured = [r for r in cases if r['x']]
        require(len(net.observed_counts) == 3*len(pictured), 'Expected three complete inference modes')
        limits = [min(192, len(r['x'])+sum(c[:len(r['x'])+1]))
                  for r, c in zip(pictured, net.observed_counts[:len(pictured)], strict=True)]
        rows = load(folder/name/'rows.json')
        modal = load(WORK/'diverse'/name/'rows.json')
        for row, other, limit in zip(rows, modal, limits, strict=True):
            require(row['index'] == other['index'] and row['local'] == other['local']
                    and row['translation']['cells'] <= limit, 'Unmatched control or violated length bound')
            row['modal'] = other['translation']
        interval = ci(rows, 'modal')
        excess = [max(0, len(r['y'])-limit) for r, limit in zip(pictured, limits, strict=True)]
        unreachable = sum(v > 0 for v in excess)
        prior = load(WORK/'capacity.json')[name]['bounds']
        result[name] = dict(improvement_vs_modal_pp=interval, length_unreachable=unreachable,
                            unavoidable_edits=sum(excess),
                            pass_all=bool(report['pass'] and interval[0]>0 and unreachable<prior['length_unreachable']))
    write_new(folder/'comparison.json', result)
    guard(start)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()

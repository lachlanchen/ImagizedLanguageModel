"""Measure a frozen editor's length constraint and first deviations from reference edits."""
import fcntl
import hashlib
import time
from collections import Counter
from zipfile import ZipFile

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_edit_writer import INSERT
from scripts.compare_edit_diversity import ARMS
from scripts.compare_visual_edit_contrast import restore
from scripts.prepare_edit_diversity import COHORTS, SHARDS, WORK, read_split
from scripts.train_visual_edit_translation import OUT, ROOT, guard, load, padded, require, sha, write_new


def order_collisions(prep):
    """Columns sharing an identical multiset have identical summed ink, pixel for pixel."""
    first, collisions = {}, {}
    for name in SHARDS:
        for row in read_split(name, prep):
            key = tuple(sorted(row['x']))
            target = hashlib.sha256(np.asarray(row['y'], dtype='<u4').tobytes()).digest()
            if key in first:
                if key not in collisions:
                    collisions[key] = Counter([first[key]])
                collisions[key][target] += 1
            else:
                first[key] = target
    return dict(rows=sum(prep['splits'][s]['rows'] for s in SHARDS), multiple_source_groups=len(collisions),
                multiple_target_groups=sum(len(c)>1 for c in collisions.values()),
                unavoidable_exact_errors=sum(c.total()-max(c.values()) for c in collisions.values()))


@torch.inference_mode()
def audit():
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'pixels.npy') == prep['pixels_sha256'], 'Physical inputs changed')
    tiles = np.load(WORK/'pixels.npy')
    path = WORK/'diverse/editor.safetensors'
    require(sha(path) == load(WORK/'diverse/fit.json')['checkpoint_sha256'], 'Checkpoint changed')
    net = restore(path).eval()
    masks = torch.from_numpy(np.unpackbits(tiles, axis=1)).cuda().bool()
    features = visual_input(masks, net.landmarks)
    q = torch.tensor(load(OUT/'preparation.json')['question_tiles'][0][0], device='cuda')[None]
    result = {'order': order_collisions(prep)}
    for name in COHORTS:
        cases = [(i, r) for i, r in enumerate(read_split(name, prep)) if r['x']]
        totals, first = Counter(), Counter()
        report = load(WORK/'diverse'/name/'report.json')
        streams = WORK/'diverse'/name/'streams.zip'
        require(sha(streams) == report['streams_sha256'], 'Generation archive changed')
        with ZipFile(streams) as z:
            # The generator uses batches of 48 in fixed roster order.
            for start_row in range(0, len(cases), 48):
                batch = cases[start_row:start_row+48]
                source, lens = padded([r['x'] for _, r in batch])
                valid = torch.arange(source.shape[1], device='cuda')[None] < torch.as_tensor(lens, device='cuda')[:,None]
                qq = q.expand(len(source), -1)
                memory, state = net.encode(features[torch.as_tensor(source, device='cuda')], valid,
                                           features[qq], torch.ones_like(qq, dtype=torch.bool))
                budgets = net.insertion_counts(memory, state).cpu().numpy()
                for (index, row), budget, n in zip(batch, budgets, lens, strict=True):
                    gold = tiles[row['y']]
                    limit = min(192, int(n + budget[:n+1].sum()))
                    emitted = z.read(f'translation/{index}.bin')
                    picture = np.frombuffer(emitted, np.uint8).reshape(-1, 128)
                    require(len(picture) <= limit, 'Saved output exceeds structural bound')
                    excess = max(0, len(gold)-limit)
                    totals.update(cases=1, reference_cells=len(gold), length_unreachable=excess>0,
                                  unavoidable_edits=excess, actual_edits=0, above_output_cap=len(gold)>192)
                    # Distinguish a reference-alignment deviation from a wrong emitted image.
                    if emitted == gold.tobytes():
                        continue
                    actions = z.read(f'translation/{index}.actions')
                    out = 0
                    for j, (a, b) in enumerate(zip(actions, row['a'])):
                        if a != b:
                            first[f'action_{b}_to_{a}'] += 1
                            break
                        if a <= INSERT:
                            if not np.array_equal(picture[out], tiles[row['o'][j]]):
                                first['same_action_wrong_bitmap'] += 1
                                break
                            out += 1
                    else:
                        first['prefix_ended'] += 1
        totals['actual_edits'] = report['outcome']['translation']['edits']
        require(sum(first.values())+report['outcome']['translation']['exact']==len(cases), 'First divergence accounting')
        result[name] = dict(bounds=dict(totals), first_reference_deviation=dict(first))
    write_new(WORK/'capacity.json', result)
    print(result, flush=True)
    guard(start)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit()

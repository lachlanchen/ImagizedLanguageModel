"""Score matched hint training and its previously unused source groups from complete image streams."""
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from scripts import fit_pictured_idioms as original
from scripts.check_pictured_idioms import raw_cases
from scripts.confirm_visual_writer_groups import prepare
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_local_teacher_writer import DATA, GUIDED, OUT, ROOT, load_split
from scripts.fit_pictured_idiom_alignment import local_counts
from scripts.score_visual_writer_transfer import matching_pairs, paired_interval
from scripts.train_visual_extractive_reader import save_page

EVAL = OUT / 'evaluation'
FRESH = GUIDED / 'fresh'
PUBLIC = ROOT / 'references/evidence/local_teacher_20261118'


def infer(model_path, destination, train, test, pixels, start, *, model_type=PicturedIdiom):
    require(sha(model_path) == load(model_path.parent / 'fit.json')['weights_sha256'], 'Checkpoint changed')
    model_weights = torch.load(model_path, weights_only=True, map_location='cpu')
    net = model_type(model_weights['landmarks']).cuda().eval()
    net.load_state_dict(model_weights)
    ink = torch.as_tensor(np.unpackbits(pixels, axis=1), device='cuda', dtype=torch.bool)
    destination.mkdir()
    original.OUT = destination
    stats = original.compare(net, visual_input(ink, net.landmarks), ink, train, test, pixels,
                             np.unique(train['target']), start)
    write_new(destination / 'result.json', stats)


def evaluate(split, train, test, pixels, start, *, fits=OUT, results=EVAL, arms=('shuffled', 'aligned'), guided=None, require_local_gain=True):
    control, treatment = arms
    folder = results / split
    folder.mkdir()
    x, y, groups = (test[k] for k in ('context', 'target', 'group'))
    before = guided
    if before is None and split != 'new':
        before = (FRESH if split == 'previous' else GUIDED) / 'distilled_targets'
    if before is None:
        before = folder / 'guided'
        infer(GUIDED / 'distilled_targets/weights.pt', before, train, test, pixels, start)
    outcomes = {}
    i, j = matching_pairs(x, y, groups)
    training = {r.tobytes() for r in train['target']}
    unseen = np.array([r.tobytes() not in training for r in y])
    absent = ~(y[:, :, None] == x[:, None]).any(-1).any(-1)
    local = local_counts(train, test)
    hits = {}
    paths = [before]
    for name in arms:
        dest = folder / name
        infer(fits / name / 'weights.pt', dest, train, test, pixels, start)
        paths.append(dest)
    for name, dest in zip(('guided', *arms), paths):
        archive = load_split(dest / 'generated.npz')
        ink, n, stop = (archive[k] for k in ('packed', 'length', 'stopped'))
        hit = (ink[:, :4] == pixels[y]).all((1, 2)) & (n == 4) & stop
        reported = load(dest / 'result.json')
        require(hit.sum() == reported['complete_exact'], 'Recount differs')
        hits[name] = hit
        outcomes[name] = dict(exact=int(hit.sum()), local_correct=int((local & hit).sum()),
                 unique_correct=reported['unique_correct'], both_correct=int((hit[i] & hit[j]).sum()),
                 all_absent_correct=int((absent & hit).sum()), unseen_correct=int((unseen & hit).sum()),
                 emitted=reported['emitted_cells'], known=reported['training_target_shape_hits'],
                 swapped=reported['swapped_exact'], archive_sha256=sha(dest / 'generated.npz'))
    candidate = hits[treatment]
    comparisons = {}
    for name in ('guided', control):
        gained = candidate & ~hits[name]
        lost = ~candidate & hits[name]
        comparisons[name] = dict(gained=int(gained.sum()), lost=int(lost.sum()),
                     local_gains=int((gained & local).sum()), local_losses=int((lost & local).sum()),
                     interval_pp=paired_interval(candidate, hits[name], groups))
    stats = dict(n=len(y), groups=len(np.unique(groups)), pairs=len(i), unseen=int(unseen.sum()),
                 all_absent=int(absent.sum()), local_exact=int(local.sum()), outcomes=outcomes, paired=comparisons)
    stats['control_vs_guided_pp'] = paired_interval(hits[control], hits['guided'], groups)
    stats['passed'] = all(v['interval_pp'][0] > 0 and (not require_local_gain or outcomes[treatment]['local_correct'] > outcomes[k]['local_correct'])
                          for k, v in comparisons.items()) and outcomes[treatment]['known'] >= .99 * outcomes[treatment]['emitted']
    if split == 'validation':
        cases = raw_cases()
        packed = load_split(folder / treatment / 'generated.npz')
        samples = {}
        for label, mask in [('gain', candidate & ~hits['guided']), ('loss', ~candidate & hits['guided'])]:
            if not mask.any():
                continue
            row = np.flatnonzero(mask)[0]
            n = packed['length'][row]
            left, right, word = cases[test['tag'][row]]
            for tag, data in [('before', pixels[x[row, 12:24]]), ('after', pixels[x[row, 24:36]]),
                              ('actual', packed['packed'][row, :n]), ('target', pixels[y[row]])]:
                if len(data):
                    save_page(folder / f'{label}_{tag}.png', data)
            samples[label] = dict(row=int(row), tag=int(test['tag'][row]), before=left, after=right, target=word,
                                  length=int(n), local_correct=bool(local[row]))
        stats['samples'] = samples
    print(split, {k:v['exact'] for k,v in outcomes.items()}, 'local', local.sum(), 'passed', stats['passed'], flush=True)
    return stats


def main():
    start = time.monotonic()
    original.guard(start)
    require((OUT / 'execution.json').is_file(), 'Both fits required')
    EVAL.mkdir()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    report = load(DATA / 'report.json')
    require(sha(DATA / 'validation.npz') == report['splits']['validation']['sha256'], 'Validation changed')
    old = load(FRESH / 'preparation.json')
    require(sha(FRESH / 'contexts.npz') == old['contexts_sha256'] and sha(FRESH / 'pixels.npy') == old['pixels_sha256'],
            'Previous confirmation changed')
    train, dev = (load_split(DATA / (k + '.npz')) for k in ('train', 'validation'))
    previous = load_split(FRESH / 'contexts.npz')
    pixels = np.load(DATA / 'pixels.npy')
    old_pixels = np.load(FRESH / 'pixels.npy')
    require(np.array_equal(pixels, old_pixels[:len(pixels)]), 'Physical pixel IDs differ')
    data = OUT / 'confirmation'
    data.mkdir()
    new, bank = prepare(train, dev, old_pixels, buckets=range(36, 40), destination=data, excluded=(previous,))
    stats = {}
    for name, test, images in [('validation', dev, pixels), ('previous', previous, old_pixels), ('new', new, bank)]:
        stats[name] = evaluate(name, train, test, images, start)
    stats['qualified'] = stats['validation']['passed'] and stats['new']['passed']
    stats['resources'] = dict(seconds=time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_new(EVAL / 'result.json', stats)


if __name__ == '__main__':
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()

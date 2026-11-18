"""Compare correctly assigned and permuted training-only local hints in a visual writer."""
from collections import Counter, defaultdict
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from scripts import fit_pictured_idioms as original
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_pictured_idiom_alignment import local_keys
from scripts.fit_visual_writer_distillation import DATA, EPOCHS, OUT as GUIDED, PRIOR, ROOT, fit_arm, prepare_teacher

OUT = ROOT / 'artifacts/local_teacher_writing_v1'
PROTOCOL = ROOT / 'references/local_teacher_writing_protocol.md'


def load_split(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def local_predictions(x, y, groups, options):
    """Cross-fit on source groups. The held-out fold never enters a count table."""
    word_id = {tuple(word): i for i, word in enumerate(options)}
    target = np.array([word_id[tuple(word)] for word in y], np.int32)
    fold = groups % 8
    hints = np.full(len(x), -1, dtype=np.int32)
    depth = np.full(len(x), -1, dtype=np.int8)
    by_fold = []
    for block in range(8):
        train = np.flatnonzero(fold != block)
        hold = np.flatnonzero(fold == block)
        require(not np.intersect1d(groups[train], groups[hold]).size, 'Source group in hint fit')
        tables = [defaultdict(Counter) for _ in range(5)]
        for i in train:
            for t, key in zip(tables, local_keys(x[i])):
                t[key][target[i]] += 1
        for i in hold:
            for level, (t, key) in enumerate(zip(tables, local_keys(x[i]))):
                if key in t:
                    histogram = t[key]
                    winner, votes = histogram.most_common(1)[0]
                    if votes >= 2 and 10 * votes >= sum(histogram.values()):
                        hints[i] = winner
                        depth[i] = level
                    break
        valid = hints[hold] >= 0
        by_fold.append(dict(fold=block, rows=len(hold), eligible=int(valid.sum()),
                            hint_correct=int((target[hold] == hints[hold]).sum())))
    return hints, depth, target, by_fold


def target_schedules(original_schedule, hints, folds):
    rng = np.random.default_rng(719)
    replace = (hints[:, None] >= 0) & (rng.random(original_schedule.shape) < .5)
    aligned = original_schedule.copy()
    shuffled = original_schedule.copy()
    for epoch in range(EPOCHS):
        for block in range(8):
            ids = np.flatnonzero(replace[:, epoch] & (folds == block))
            labels = hints[ids]
            aligned[ids, epoch] = labels
            shuffled[ids, epoch] = rng.permutation(labels)
            require(np.array_equal(np.sort(labels), np.sort(shuffled[ids, epoch])), 'Unequal hint distribution')
    return aligned, shuffled, replace


def main():
    start = time.monotonic()
    original.guard(start)
    require(torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(), 'Expected 4090')
    torch.manual_seed(613)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    OUT.mkdir()
    prep = load(ROOT / 'references/evidence/pictured_idioms_20261118/preparation.json')
    require(sha(DATA / 'train.npz') == prep['splits']['train']['sha256'] and
            sha(DATA / 'pixels.npy') == prep['pixels_sha256'], 'Training corpus changed')
    base = load(ROOT / 'references/evidence/pictured_idiom_alignment_20261118/result.json')
    for name in ('weights.pt', 'training_only_aux.pt'):
        p = PRIOR / name
        require(sha(p) == base['source_hashes'][str(p.relative_to(ROOT))], 'Initial weights changed')
    init = torch.load(PRIOR / 'weights.pt', weights_only=True, map_location='cpu')
    reader = PicturedIdiom(init['landmarks']).cuda().eval()
    reader.load_state_dict(init)
    aux = nn.Sequential(nn.Linear(448, 384), nn.GELU(), nn.Linear(384, 192)).cuda().eval()
    aux.load_state_dict(torch.load(PRIOR / 'training_only_aux.pt', weights_only=True))
    pixels = np.load(DATA / 'pixels.npy')
    ink = torch.as_tensor(np.unpackbits(pixels, axis=1), device='cuda', dtype=torch.bool)
    features = visual_input(ink, reader.landmarks)
    train = load_split(DATA / 'train.npz')
    x, y = [torch.as_tensor(train[k].astype(np.int32), device='cuda') for k in ('context', 'target')]
    contexts, options, base_schedule, teacher = prepare_teacher(reader, aux, features, x, y)
    old = load(GUIDED / 'teacher.json')
    require(all(teacher[k] == old[k] for k in teacher), 'Original supervision did not reproduce')
    begin = time.monotonic()
    hints, depth, target, folds = local_predictions(train['context'], train['target'], train['group'],
                                                    options.cpu().numpy())
    guided = base_schedule.cpu().numpy()
    aligned, shuffled, replace = target_schedules(guided, hints, train['group'] % 8)
    eligible = hints >= 0
    np.savez_compressed(OUT / 'hint_assignment.npz', hints=hints, key_level=depth, target=target, replaced=replace,
                        aligned=aligned, shuffled=shuffled)
    write_new(OUT / 'teacher.json', dict(examples=len(x), eligible=int(eligible.sum()), folds=folds,
              correct_on_eligible=int((hints == target).sum()), visits=int(replace.sum()), total_visits=int(guided.size),
              original_targets=teacher, hint_seconds=time.monotonic() - begin,
              changed_vs_guided={name:int((schedule != guided).sum()) for name, schedule in [('aligned', aligned), ('shuffled', shuffled)]},
              assignment_sha256=sha(OUT / 'hint_assignment.npz')))
    print('eligible', eligible.sum(), 'correct', (hints == target).sum(), flush=True)
    for name, schedule in [('shuffled', shuffled), ('aligned', aligned)]:
        fit_arm(name, init, features, x, y, contexts, options, torch.as_tensor(schedule, device='cuda'),
                ink[:, reader.landmarks].float(), start, destination=OUT)
    original.guard(start)
    write_new(OUT / 'execution.json', dict(seconds=time.monotonic() - start, device=torch.cuda.get_device_name(),
              host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              initial_sha256=sha(PRIOR/'weights.pt'), protocol_sha256=sha(PROTOCOL), program_sha256=sha(Path(__file__)),
              optimizer_source_sha256=sha(ROOT/'scripts/fit_visual_writer_distillation.py')))


if __name__ == '__main__':
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()

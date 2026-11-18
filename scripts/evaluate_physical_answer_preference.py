"""Evaluate joint reference and preference fits on disjoint pictured source groups."""
import fcntl
import time
import resource

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import PicturedIdiom
from scripts.confirm_visual_writer_groups import prepare
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_local_teacher_writer import FRESH, evaluate
from scripts.fit_local_teacher_writer import DATA, GUIDED, OUT as HINTS, load_split
from scripts.fit_physical_answer_preference import OUT
from scripts.fit_pictured_idioms import guard
from scripts.score_paired_physical_answers import score


def main():
    start = time.monotonic()
    guard(start)
    require((OUT/'execution.json').is_file(), 'Both final checkpoints required')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    train, val = (load_split(DATA / (k+'.npz')) for k in ('train','validation'))
    prior = load_split(FRESH/'contexts.npz')
    source = HINTS/'confirmation'
    seen = load_split(source/'contexts.npz')
    bank = np.load(source/'pixels.npy')
    previous = load(source/'preparation.json')
    require(sha(source/'contexts.npz') == previous['contexts_sha256'] and sha(source/'pixels.npy') == previous['pixels_sha256'],
            'Bucket 36–39 input changed')
    fresh = OUT/'confirmation'
    fresh.mkdir()
    new, pixels = prepare(train, val, bank, buckets=range(40,44), destination=fresh, excluded=(prior,seen))
    reports = {}
    folder = OUT/'evaluation'
    folder.mkdir()
    for split, inputs, images, guided in [('validation',val,np.load(DATA/'pixels.npy'),GUIDED/'distilled_targets'),
                                          ('preceding',seen,bank,HINTS/'evaluation/new/guided'),
                                          ('new',new,pixels,None)]:
        stats = evaluate(split, train, inputs, images, start, fits=OUT, results=folder,
                         arms=('reference','preference'), guided=guided, require_local_gain=False)
        reports[split] = stats
    counts = reports['new']['outcomes']
    reports['qualified'] = (reports['validation']['passed'] and reports['new']['passed'] and
                            counts['preference']['both_correct'] >= counts['guided']['both_correct'])
    diagnostic = OUT/'paired_trajectories'
    diagnostic.mkdir()
    torch.set_float32_matmul_precision('highest')
    paired = {}
    for name, parent in [('guided', GUIDED/'distilled_targets'), ('reference', OUT/'reference'), ('preference', OUT/'preference')]:
        state = torch.load(parent/'weights.pt', weights_only=True, map_location='cpu')
        model = PicturedIdiom(state['landmarks']).cuda().eval()
        model.load_state_dict(state)
        paired[name] = score(model, 'new', new, pixels, name, start, destination=diagnostic)
    write_new(diagnostic/'result.json', paired)
    reports['resources'] = dict(seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_new(folder/'result.json', reports)
    print('qualified', reports['qualified'], flush=True)


if __name__ == '__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()

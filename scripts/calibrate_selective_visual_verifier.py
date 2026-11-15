#!/usr/bin/env python3
"""Register/run a training-only selective content verifier on frozen answers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from scripts.calibrate_whole_answer_reward import ROOT, OLD, CORPUS, REFS, inputs, load, sha, write_new

REG = ROOT / 'references/evidence/selective_visual_verifier_registration_20261116.json'
OUT = ROOT / 'references/evidence/selective_visual_verifier_calibration_20261116.json'
RUN = ROOT / 'artifacts/selective_visual_verifier_v1'


def contract():
    rows, sources = inputs()
    report = load(CORPUS / 'report.json')
    training = []
    for shard in report['shards']:
        if shard['split'] != 'train':
            continue
        name, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
        path = CORPUS / name
        if path.name != name or sha(path) != shard['files'][name]:
            raise ValueError('Training glyph source changed')
        sources[str(path.relative_to(ROOT))] = shard['files'][name]
        training.append(dict(file=str(path.relative_to(ROOT)), glyphs=shard['glyphs'], records=shard['records']))
    for path in [Path(__file__), ROOT / 'ilm/training_only/__init__.py',
                 ROOT / 'ilm/training_only/visual_content_verifier.py',
                 ROOT / 'scripts/calibrate_whole_answer_reward.py',
                 CORPUS / 'verification.json']:
        sources[str(path.relative_to(ROOT))] = sha(path)
    return rows, sources, training


def calibrate(reg):
    import numpy as np
    from ilm.training_only.visual_content_verifier import SelectiveGlyphVerifier, packed_images

    began = time.perf_counter()
    def guard():
        if time.perf_counter() - began > 300:
            raise RuntimeError('CPU calibration elapsed ceiling exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 > 2:
            raise RuntimeError('CPU calibration RSS ceiling exceeded')
        mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith(('MemAvailable:',))}
        if mem['MemAvailable'] / 1024**2 < 24:
            raise RuntimeError('Shared workstation host reserve exhausted')
    if RUN.exists():
        raise FileExistsError('Preserve prior outputs; no automatic retry')
    RUN.mkdir(parents=True)
    guard()
    vocabulary = set(); glyphs = records = 0
    for number, source in enumerate(reg['training_pixel_sources'], 1):
        guard()
        with np.load(ROOT / source['file'], allow_pickle=False) as data:
            pixels = data['ink']
            if pixels.dtype != np.uint8 or pixels.shape != (source['glyphs'], 128):
                raise ValueError('Training physical image geometry differs')
            unique = np.unique(np.ascontiguousarray(pixels).view('V128').reshape(-1))
            vocabulary.update(row.tobytes() for row in unique)
            glyphs += len(pixels); records += source['records']
        if number % 16 == 0:
            print(json.dumps(dict(phase='training_only_inventory', completed_shards=number,
                                  prototypes=len(vocabulary))), flush=True)
    # Lexicographic physical-image IDs are stable; no strings/Unicode labels,
    # development pixels or reference images enter this prototype inventory.
    prototypes = np.frombuffer(b''.join(sorted(vocabulary)), dtype=np.uint8).reshape(-1, 128).copy()
    del vocabulary
    inventory = RUN / 'training_inventory.npz'; np.savez_compressed(inventory, prototypes=prototypes)
    verifier = SelectiveGlyphVerifier(prototypes, guard=guard)
    inventory_seconds = time.perf_counter() - began
    print(json.dumps(dict(phase='inventory_ready', prototypes=len(prototypes), seconds=inventory_seconds)), flush=True)

    report = load(CORPUS / 'report.json')
    verification = load(CORPUS / 'verification.json')
    if verification['report_sha256'] != sha(CORPUS / 'report.json') or verification['cross_partition_exact_input_or_source_group_overlap'] != 0:
        raise ValueError('Require existing group-disjoint corpus verification')
    development = []; offset = 0
    for shard in report['shards']:
        if shard['split'] != 'development':
            continue
        name, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
        development.append((offset, offset + shard['records'], CORPUS / name, shard['files'][name]))
        offset += shard['records']
    cached = {}
    def reference(index):
        first, end, path, digest = next(row for row in development if row[0] <= index < row[1])
        if path not in cached:
            if sha(path) != digest:
                raise ValueError('Source reference shard changed')
            cached.clear()
            with np.load(path, allow_pickle=False) as arrays:
                cached[path] = {n: arrays[n] for n in ('ink', 'record_glyph_offsets', 'prompt_glyphs')}
        arrays = cached[path]; local = index - first
        begin, finish = arrays['record_glyph_offsets'][local:local + 2]
        begin += arrays['prompt_glyphs'][local]
        return np.unpackbits(arrays['ink'][begin:finish], axis=1, bitorder='big').reshape(-1, 1, 32, 32)

    freeze = load(OLD / 'review/reviewer_pre_unmask_freeze.json')
    if freeze['ready_readings_sha256'] != sha(OLD / 'review/frozen_readings.json'):
        raise ValueError('Preserve original frozen transcriptions')
    refs = {r['id']: r for r in load(REFS)['reference_rows'] if r['kind'] == 'qa'}
    scores = {layout: {r['case']: r for r in load(OLD / 'review' / (layout + '_core_scores.json'))['records']
                      if r['step'] == 41298 and r['kind'] == 'qa' and r['condition'] == 'normal'}
              for layout in ('fixed_cells', 'proportional')}
    results = []
    for bound in reg['rows']:
        guard()
        row = scores[bound['layout']][bound['case']]
        folder = OLD / 'evaluation/step_041298/core128' / (bound['layout'] + '__' + bound['case'])
        original = load(folder / 'report.json')
        if (sha(folder / 'ink.npz') != original['raw_sha256'] or
                sha(folder / 'answer.png') != row['image_sha256'] or
                original['stopped'] != row['stopped']):
            raise ValueError('Output pixels differ from frozen reading identity')
        with np.load(folder / 'ink.npz', allow_pickle=False) as arrays:
            ink = arrays['ink']
        target = reference(bound['corpus_index'])
        if len(target) != len(refs[bound['case'].removesuffix('_normal')]['answer']) or len(ink) != row['emitted_glyphs']:
            raise ValueError('Complete input/reference extent differs')
        result = verifier.score(ink, target, stopped=row['stopped'], guard=guard)
        results.append(dict(**bound, **result, complete_exact=row['complete_exact'],
                            readable=row['readable'], uncertain=row['reading']['uncertain'],
                            stopped=row['stopped'], emitted_glyphs=len(ink)))
    correct = [r for r in results if r['complete_exact']]
    wrong = [r for r in results if r['readable'] and not r['complete_exact']]
    uncertain = [r for r in results if not r['readable']]
    if (len(correct), len(wrong), len(uncertain)) != (9, 9, 46):
        raise ValueError('Retain every original review outcome')
    auc = sum(float(a['reward'] > b['reward']) + .5 * float(a['reward'] == b['reward'])
              for a in correct for b in wrong) / (len(correct) * len(wrong))
    checks = dict(all_clean_references_covered=all(r['reference_covered'] for r in results),
                  all_nine_correct_content_accepted=all(r['complete_content'] for r in correct),
                  no_readable_wrong_content_accepted=all(not r['complete_content'] for r in wrong),
                  no_uncertain_content_accepted=all(not r['complete_content'] for r in uncertain),
                  correct_wrong_auc_at_least_095=auc >= .95,
                  all_correct_above_every_uncredited=min(r['reward'] for r in correct) > max(
                      r['reward'] for r in wrong + uncertain),
                  capped_reward_zero=all(r['reward'] == 0 for r in results if not r['stopped']))
    guard()
    return dict(schema='selective-training-only-content-verifier-v1', registration_sha256=sha(REG),
                rows=results, checks=checks, calibration_passed=all(checks.values()), training_authorized=False,
                correct_wrong_auc=auc, accepted_correct=sum(r['complete_content'] for r in correct),
                accepted_readable_wrong=sum(r['complete_content'] for r in wrong),
                accepted_uncertain=sum(r['complete_content'] for r in uncertain),
                inventory=dict(prototypes=len(prototypes), training_glyph_positions=glyphs, training_records=records,
                    training_shards=len(reg['training_pixel_sources']), file=str(inventory.relative_to(ROOT)),
                    sha256=sha(inventory), minimum_separation=int(verifier.minimum_distance.min()),
                    maximum_separation=int(verifier.minimum_distance.max()), build_seconds=inventory_seconds,
                    development_or_reference_images_added=False, Unicode_labels=False, deployed=False),
                elapsed_seconds=time.perf_counter() - began,
                peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                original_readings_unchanged=True, reused_development=True, independent_human_adjudication=False,
                neural_calls=0, optimizer_updates=0, goal_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('register', 'run'))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('OPENBLAS_NUM_THREADS') != '2':
        raise ValueError('CPU-only, two-thread BLAS admission required')
    rows, sources, training = contract()
    if args.action == 'register':
        write_new(REG, dict(schema='selective-content-verifier-registration-v1', rows=rows,
            input_sha256=sources, training_pixel_sources=training,
            inventory_policy='Unique clean physical bitmaps from every training shard only; no development/reference additions or Unicode labels.',
            decoding_policy='Accept nearest prototype only when 2*Hamming distance < its nearest distinct prototype separation. Otherwise abstain.',
            reward='Stopped complete accepted identity match bonus plus ordered edit similarity; cap/all-white reward0; missing references fail coverage.',
            gates=['all clean reference cells covered', 'all9known-correct complete content accepted',
                   'no9readable-wrong complete content accepted', 'no46uncertain complete content accepted',
                   'known-correct/readable-wrong AUC>=.95', 'min known-correct reward > max uncredited', 'capped reward0'],
            limits=dict(cpu_threads=2, peak_rss_gib=2, elapsed_seconds=300, minimum_host_available_gib=24),
            action_on_failure='Reject this verifier for policy optimization; retain all original readings and outputs.',
            action_on_pass='Limited reused-output calibration only. No automatic training; register real actor support/resource and adversarial verification first.',
            neural_execution=False, policy_fit_authorized=False))
        print(json.dumps(dict(registered=str(REG.relative_to(ROOT)), inputs=len(sources), training_shards=len(training))))
    else:
        reg = load(REG)
        if reg['rows'] != rows or reg['input_sha256'] != sources or reg['training_pixel_sources'] != training:
            raise ValueError('Frozen verifier calibration sources changed')
        result = calibrate(reg); write_new(OUT, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)

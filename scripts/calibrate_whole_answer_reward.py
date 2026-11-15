#!/usr/bin/env python3
"""Register/run a CPU-only reward falsification on 64 frozen final QA outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
CORPUS = ROOT / 'data/visual_sequence_corpus_v1'
REFS = ROOT / 'artifacts/compact_visual_span_evaluation_v1/references.json'
REG = ROOT / 'references/evidence/whole_answer_reward_calibration_registration_20261116.json'
OUT = ROOT / 'references/evidence/whole_answer_reward_calibration_20261116.json'


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def inputs():
    sources = [Path(__file__), ROOT / 'ilm/visual_lm/on_policy_glyph_learning.py',
               ROOT / 'ilm/visual_lm/visual_sequence_corpus.py', CORPUS / 'report.json',
               REFS, OLD / 'review/mapping.json', OLD / 'review/frozen_readings.json',
               OLD / 'review/reviewer_pre_unmask_freeze.json', OLD / 'review/summary.json',
               OLD / 'evaluation/report.json']
    refs = {r['source_id']: r for r in load(REFS)['reference_rows'] if r['kind'] == 'qa'}
    positions = {}; offset = 0
    for shard in load(CORPUS / 'report.json')['shards']:
        if shard['split'] != 'development':
            continue
        name, = [n for n in shard['files'] if n.endswith('_sources.json')]
        sources.append(CORPUS / name)
        for local, row in enumerate(load(CORPUS / name)):
            if row['kind'] == 'qa' and row['id'] in refs:
                ref = refs[row['id']]
                if row['group'] != ref['group'] or ref['id'] in positions:
                    raise ValueError('Ambiguous or mismatched reference image ownership')
                positions[ref['id']] = offset + local
                pixels, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
                sources.append(CORPUS / pixels)
        offset += shard['records']
    rows = []
    for layout in ('fixed_cells', 'proportional'):
        scores = OLD / 'review' / (layout + '_core_scores.json'); sources.append(scores)
        for row in load(scores)['records']:
            if row['step'] != 41298 or row['kind'] != 'qa' or row['condition'] != 'normal':
                continue
            folder = OLD / 'evaluation/step_041298/core128' / (layout + '__' + row['case'])
            sources.extend(folder / n for n in ('ink.npz', 'report.json', 'answer.png'))
            rows.append(dict(case=row['case'], layout=layout,
                             corpus_index=positions[row['case'].removesuffix('_normal')]))
    if len(positions) != 32 or len(rows) != 64:
        raise ValueError('Keep all fixed 64 final normal questions')
    return rows, {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))}


def run(reg):
    # Import only after source authentication. No model/checkpoint imports.
    import resource
    import numpy as np
    import torch
    from ilm.visual_lm.on_policy_glyph_learning import image_answer_reward
    from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
    torch.set_num_threads(2)
    began = time.perf_counter()
    corpus = VisualSequenceCorpus(CORPUS, split='development',
        expected_report_sha256=reg['input_sha256'][str((CORPUS / 'report.json').relative_to(ROOT))])
    refs = {r['id']: r for r in load(REFS)['reference_rows'] if r['kind'] == 'qa'}
    freeze = load(OLD / 'review/reviewer_pre_unmask_freeze.json')
    if (freeze['ready_readings_sha256'] != sha(OLD / 'review/frozen_readings.json')
            or freeze['mapping_sha256'] != sha(OLD / 'review/mapping.json')):
        raise ValueError('Frozen original review identity differs')
    scores = {layout: {r['case']: r for r in load(OLD / 'review' / (layout + '_core_scores.json'))['records']
                      if r['step'] == 41298 and r['kind'] == 'qa' and r['condition'] == 'normal'}
              for layout in ('fixed_cells', 'proportional')}
    results = []
    for bound in reg['rows']:
        row = scores[bound['layout']][bound['case']]
        folder = OLD / 'evaluation/step_041298/core128' / (bound['layout'] + '__' + bound['case'])
        report = load(folder / 'report.json')
        if (report['raw_sha256'] != sha(folder / 'ink.npz') or
                report['image_sha256'] != row['image_sha256'] or
                sha(folder / 'answer.png') != row['image_sha256']):
            raise ValueError('Original pixels do not match their frozen reading')
        with np.load(folder / 'ink.npz', allow_pickle=False) as data:
            ink = torch.from_numpy(data['ink'].copy())
        if len(ink) != row['emitted_glyphs'] or row['stopped'] != report['stopped']:
            raise ValueError('Actual output count/stopping differs')
        ref = corpus[bound['corpus_index']]
        target = ref['ink'][ref['prompt_glyphs']:]
        if len(target) != len(refs[bound['case'].removesuffix('_normal')]['answer']):
            raise ValueError('Wrong complete reference extent')
        reward = image_answer_reward(ink, target, stopped=row['stopped'])
        results.append(dict(**bound, reward=reward, complete_exact=row['complete_exact'],
                            readable=row['readable'], uncertain=row['reading']['uncertain'],
                            stopped=row['stopped'], emitted_glyphs=len(ink)))
    correct = [r['reward'] for r in results if r['complete_exact']]
    wrong_clear = [r['reward'] for r in results if r['readable'] and not r['complete_exact']]
    uncertain = [r['reward'] for r in results if not r['readable']]
    if not correct or not wrong_clear:
        raise ValueError('Both known correct and readable wrong outputs are needed')
    auc = sum(float(a > b) + .5 * float(a == b) for a in correct for b in wrong_clear) / (len(correct) * len(wrong_clear))
    checks = dict(correct_vs_readable_wrong_auc_at_least_095=auc >= .95,
                  all_correct_above_all_readable_wrong=min(correct) > max(wrong_clear),
                  all_correct_above_all_unreadable=min(correct) > max(uncertain, default=-1),
                  capped_reward_zero=all(r['reward'] == 0 for r in results if not r['stopped']))
    return dict(schema='whole-answer-reward-saved-output-calibration-v1',
                registration_sha256=sha(REG), rows=results,
                counts=dict(correct=len(correct), readable_wrong=len(wrong_clear), unreadable=len(uncertain)),
                correct_vs_readable_wrong_auc=auc, minimum_correct_reward=min(correct),
                maximum_readable_wrong_reward=max(wrong_clear), maximum_unreadable_reward=max(uncertain, default=None),
                checks=checks, calibration_passed=all(checks.values()), training_authorized=False,
                elapsed_scoring_seconds=time.perf_counter() - began,
                peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                new_model_calls=0, optimizer_updates=0, unchanged_original_readings=True,
                reused_development=True, independent_human_adjudication=False, goal_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('register', 'run'))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only operation requires CUDA_VISIBLE_DEVICES empty')
    rows, hashes = inputs()
    if args.action == 'register':
        write_new(REG, dict(schema='whole-answer-reward-calibration-registration-v1', rows=rows,
            input_sha256=hashes, checks=['AUC >= .95 on complete-correct versus readable-wrong',
                'min correct > max readable-wrong', 'min correct > max unreadable', 'all capped reward zero'],
            resource_limits=dict(cpu_threads=2, maximum_rss_gib=2, maximum_elapsed_seconds=60),
            new_model_calls=0, optimizer_updates=0, policy_fit_authorized=False,
            action_on_failure='Reject this reward for policy optimization; preserve original scores/readings.',
            action_on_pass='Only a limited reused-data calibration; require training-only model/resource support preflight before any fit.'))
        print(json.dumps(dict(registered=str(REG.relative_to(ROOT)), rows=len(rows), inputs=len(hashes))))
    else:
        reg = load(REG)
        if rows != reg['rows'] or hashes != reg['input_sha256']:
            raise ValueError('Calibration sources changed after registration')
        result = run(reg)
        if result['peak_process_rss_gib'] > 2 or result['elapsed_scoring_seconds'] > 60:
            raise ValueError('CPU calibration resource ceiling exceeded')
        write_new(OUT, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))

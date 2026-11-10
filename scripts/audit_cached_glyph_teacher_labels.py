#!/usr/bin/env python3
"""Audit existing native-head labels on gold-image training histories, on CPU.

No neural forward or teacher is called. Agreement with one training reference
is not semantic accuracy; prefix-position differences are not causal effects.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
OWN = 'scripts/audit_cached_glyph_teacher_labels.py'
LANGUAGE = ROOT / 'artifacts/semantic_glyph_language_v1'
COMPILER = ROOT / 'artifacts/semantic_glyph_readout_distillation_v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    n = len(rows)
    require(n > 0, 'empty audit group')
    return dict(states=n, reference_form_matches=sum(r['reference_match'] for r in rows),
                reference_form_agreement=sum(r['reference_match'] for r in rows) / n,
                native_terminal_winners=sum(r['native_label'] == -1 for r in rows),
                unsupported_native_winners=sum(r['native_label'] == -2 for r in rows))


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only audit requires CUDA hidden')
    require(not output.exists(), 'preserve earlier audits')
    require(subprocess.check_output(['git', 'show', 'HEAD:' + OWN], cwd=ROOT)
            == (ROOT / OWN).read_bytes(), 'commit audit code before execution')
    torch.set_num_threads(4)
    began = time.monotonic()
    meta = load(COMPILER / 'preparation.json')
    calibration = load(COMPILER / 'calibrate/report.json')
    require(load(COMPILER / 'calibrate/status.json')['phase'] == 'completed',
            'native-head calibration must already be completed')
    require(sha(COMPILER / 'genuine.pt') == meta['genuine_sha256'], 'cached states changed')
    require(sha(COMPILER / 'calibrate/genuine_labels.pt') == calibration['genuine_labels_sha256'],
            'previously computed native winners changed')
    for name, digest in meta['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'original training source changed: ' + name)
    existing = torch.load(COMPILER / 'genuine.pt', map_location='cpu', weights_only=True)
    native = torch.load(COMPILER / 'calibrate/genuine_labels.pt',
                        map_location='cpu', weights_only=True)['labels']
    require(native.shape == existing['reference_labels'].shape, 'label counts differ')
    source = load(LANGUAGE / 'cache/report.json')
    prep = load(LANGUAGE / 'preparation.json')
    jobs = {r['index']: r for r in prep['jobs']}
    records, cases = [], []
    offset = 0
    for item in source['files']:
        path = LANGUAGE / 'cache' / item['file']
        require(sha(path) == item['sha256'], 'original history shard changed')
        value = torch.load(path, map_location='cpu', weights_only=True)
        labels = value['labels']
        n = len(labels)
        require(torch.equal(existing['states'][offset:offset + n], value['states']) and
                torch.equal(existing['reference_labels'][offset:offset + n], labels),
                'native winners no longer align with the original states')
        require(labels.tolist() == jobs[item['index']]['labels'] + [-1],
                'original answer and terminal order changed')
        current = []
        for position, expected in enumerate(labels.tolist()):
            actual = int(native[offset + position])
            row = dict(case_index=item['index'], prefix_glyphs=position,
                       role='terminal' if expected == -1 else 'first_glyph' if position == 0
                       else 'continuation_glyph', reference_label=expected,
                       native_label=actual, reference_match=actual == expected)
            records.append(row)
            current.append(row)
        cases.append(dict(case_index=item['index'], answer_glyphs=n - 1,
                          first_reference_form_matches=current[0]['reference_match'],
                          all_reference_forms_and_stop_match=all(r['reference_match'] for r in current),
                          terminal_reference_matches=current[-1]['reference_match'],
                          continuation_states=n - 2,
                          continuation_reference_matches=sum(r['reference_match'] for r in current[1:-1])))
        offset += n
    require(offset == len(native), 'unused native labels remain')
    groups = {role: summarize([r for r in records if r['role'] == role])
              for role in ('first_glyph', 'continuation_glyph', 'terminal')}
    bins = {}
    for name, lo, hi in [('0', 0, 0), ('1', 1, 1), ('2-3', 2, 3),
                         ('4-7', 4, 7), ('8+', 8, 1000000)]:
        rows = [r for r in records if r['role'] != 'terminal' and lo <= r['prefix_glyphs'] <= hi]
        if not rows:
            continue
        case_ids = {r['case_index'] for r in rows}
        same_cases_first = [r for r in records if r['role'] == 'first_glyph'
                            and r['case_index'] in case_ids]
        bins[name] = dict(**summarize(rows), represented_cases=len(case_ids),
                          first_states_of_same_cases=summarize(same_cases_first))
    initially_matching = {r['case_index'] for r in cases if r['first_reference_form_matches']}
    later = [r for r in records if r['case_index'] in initially_matching
             and r['role'] == 'continuation_glyph']
    require(time.monotonic() - began < 120, 'CPU audit exceeded 120-second budget')
    output.mkdir(parents=True)
    (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    report = dict(
        source_sha256=sha(ROOT / OWN),
        compiler_preparation_sha256=sha(COMPILER / 'preparation.json'),
        calibration_report_sha256=sha(COMPILER / 'calibrate/report.json'),
        native_labels_sha256=calibration['genuine_labels_sha256'],
        original_cache_report_sha256=sha(LANGUAGE / 'cache/report.json'),
        original_states_and_reference_labels_exactly_aligned=True,
        training_questions=len(cases), states=len(records), groups=groups, prefix_bins=bins,
        first_matching_cases_continuation=summarize(later) if later else None,
        cases_all_reference_forms_and_stop_match=sum(r['all_reference_forms_and_stop_match'] for r in cases),
        records_sha256=sha(output / 'records.json'), cases_sha256=sha(output / 'cases.json'),
        elapsed_seconds=time.monotonic() - began, CPU_threads=4, model_forward_calls=0,
        optimizer_updates=0, CUDA_calls=0, new_teacher_calls=0,
        native_head_labels_computed_in_prior_calibration=True,
        runtime_changed=False, development_inputs_used=False,
        autonomous_answers_generated=0, semantic_accuracy_measured=False,
        causal_effect_of_image_encoder_established=False,
        interpretation=('Every state follows the original gold glyph-image prefix. '
                        'The prior native-head winner is mapped to its first complete form, '
                        'terminal, or unsupported label. Disagreement can reflect an alternate '
                        'answer, formatting, source-unit segmentation, or image-feedback effects. '
                        'Position comparisons alone isolate none of those causes.'))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())

#!/usr/bin/env python3
"""Independent saved-pixel/accounting audit; never replays the neural model."""
import json
import math
import os
from pathlib import Path
import time

from scripts import diagnose_visual_context_gradients as common

ROOT = common.ROOT
PUBLIC = ROOT / 'references/evidence/continuous_autoregressive_preflight_result_20261118.json'
OUT = ROOT / 'references/evidence/continuous_autoregressive_preflight_audit_20261118.json'


def main():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Saved-array CPU audit only')
    common.require(not OUT.exists(), 'Preserve original audit')
    import numpy as np
    import torch
    from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
    torch.set_num_threads(2)
    start = time.perf_counter()
    report = common.load(PUBLIC)
    reg = common.load(ROOT / 'references/evidence/continuous_autoregressive_preflight_registration_20261118.json')
    common.require(common.sha(ROOT / 'references/evidence/continuous_autoregressive_preflight_registration_20261118.json') == report['registration_sha256'], 'Original binding changed')
    corpus = VisualSequenceCorpus(ROOT / 'data/visual_sequence_corpus_v1', split='train', expected_report_sha256=reg['data_report_sha256'])
    total = 0
    for case in report['cases']:
        ink = corpus[case['case']['corpus_index']]['ink'].numpy().astype(bool)
        first = max(1, case['case']['prompt_glyphs'])
        target = ink[first:]
        common.require(case['initial_sha256'] == case['reset_sha256'] != case['updated_sha256'], 'Retained trial update or unchanged step')
        common.require(all(math.isfinite(n) and n > 0 for n in case['gradient_norms'].values()), 'Gradient support differs')
        for stage in ('before', 'after'):
            row = case[stage]
            path = ROOT / row['raw_file']
            common.require(common.sha(path) == row['raw_sha256'], 'Actual generated image bytes changed')
            with np.load(path, allow_pickle=False) as data:
                actual = data['ink']
            common.require(actual.shape == target.shape and np.isin(actual, [0, 1]).all(), 'Complete original binary output required')
            actual = actual.astype(bool)
            tp = int((actual & target).sum()); fp = int((actual & ~target).sum()); fn = int((~actual & target).sum())
            source_blank = ~target.reshape(len(target), -1).any(1)
            emitted_blank = ~actual.reshape(len(actual), -1).any(1)
            measured = dict(positions=len(target), tp=tp, fp=fp, fn=fn,
                exact=int((actual == target).reshape(len(target), -1).all(1).sum()),
                source_blank=int(source_blank.sum()), false_blank=int((emitted_blank & ~source_blank).sum()),
                false_ink=int((~emitted_blank & source_blank).sum()), f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.)
            common.require(measured == row['pixels'], 'Independent complete-pixel recount differs')
            common.require(row['target_positions'] == len(target) and row['stop_positions'] == len(target) + 1, 'Complete causal support changed')
            common.require(math.isclose(row['total'], row['code_nll'] + row['stop_bce'], abs_tol=3e-7), 'Loss arithmetic differs')
            total += len(target)
        common.require(case['after']['total'] < case['before']['total'], 'Objective support differs')
    common.require(total == 2208 and report['actual_trial_updates'] == 6 and report['retained_updates'] == 0, 'Complete execution accounting required')
    common.write_new(OUT, dict(schema='ilm-continuous-autoregressive-preflight-audit-v1',
        report_sha256=common.sha(PUBLIC), registration_sha256=report['registration_sha256'],
        all_actual_images_recounted=total, all_six_case_receipts_checked=True, all_pixel_scores_agree=True,
        auditor_sha256=common.sha(Path(__file__)), elapsed_seconds=time.perf_counter() - start,
        neural_replay=False, language_quality_measured=False, goal_achieved=False))
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)), images=total)), flush=True)


if __name__ == '__main__':
    main()

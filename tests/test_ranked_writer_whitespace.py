"""New density-rule safeguards; no old fit, neural probe or reading is repeated."""
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts import analyze_ranked_writer_whitespace as probe


def test_missing_binding_imports_no_neural_runtime_or_outputs(tmp_path):
    program = '''
import sys
from pathlib import Path
from scripts import analyze_ranked_writer_whitespace as p
p.PUBLIC=Path(sys.argv[1])/'absent.json';p.DEST=Path(sys.argv[1])/'absent'
try:p.validate()
except ValueError as e:assert 'Bind diagnostic' in str(e)
else:raise AssertionError('Missing binding admitted')
assert 'torch' not in sys.modules and not p.DEST.exists()
'''
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], cwd=probe.ROOT, check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_fixed_rule_counts_erasure_costs_and_strict_margin_without_repair():
    white = np.array([True, True, False, False, False])
    empty = np.array([False, True, False, False, True])
    selected = np.ones(5, bool); credited = ~empty
    b = np.array([.002, 0., .002, .001, 0.]); e = np.zeros(5)
    row = probe.case_counts(white, empty, b, e, selected, credited)
    assert row['fixed_false_ink'] == row['added_false_blanks'] == row['lost_nonempty_source_credits'] == 1
    assert row['predicted_false_ink'] == 0 and row['predicted_false_blanks'] == 2
    assert row['predicted_readable'] == 1 and row['lost_selected_credits'] == 2
    assert np.array_equal(empty, [False, True, False, False, True])
    with pytest.raises(ValueError, match='contain ink'):
        probe.case_counts(white, empty, b, e, selected, np.ones(5, bool))


def passing_groups():
    result = probe.load(probe.RESULT); groups = {}
    readable = {('/'.join(r[k] for k in ['split', 'kind', 'layout'])): r for r in result['readable_groups'] if r['role'] == 'candidate'}
    for row in result['complete_groups']:
        if row['role'] != 'candidate':
            continue
        key = '/'.join(row[k] for k in ['split', 'kind', 'layout']); r = readable[key]
        groups[key] = dict(positions=row['positions'], source_whitespace=row['source_whitespace'],
            original_false_ink=row['false_ink'], original_false_blanks=row['false_blanks'],
            switched_to_blank=row['false_ink'], fixed_false_ink=row['false_ink'], added_false_blanks=0,
            predicted_false_ink=0, predicted_false_blanks=row['false_blanks'], selected_positions=r['count'],
            original_readable=r['readable_nonblank'], lost_selected_credits=0,
            lost_nonempty_source_credits=0, predicted_readable=r['readable_nonblank'])
    return groups


def test_each_prospective_safeguard_rejects_harm_without_tuning_or_pooling():
    assert probe.decision(passing_groups())['separately_registered_decoding_comparison_eligible']
    g = passing_groups()
    for r in g.values():
        r['fixed_false_ink'] = r['switched_to_blank'] = 0; r['predicted_false_ink'] = r['original_false_ink']
    assert not probe.decision(g)['rules']['whitespace']
    g = passing_groups(); r = g['development/qa/proportional']
    r['added_false_blanks'] = 1; r['predicted_false_blanks'] += 1; r['switched_to_blank'] += 1
    assert not probe.decision(g)['rules']['no_added_false_blanks']
    for key in passing_groups():
        g = passing_groups(); r = g[key]
        r['lost_selected_credits'] = r['original_readable'] - int(.9 * r['selected_positions']) + 1
        r['predicted_readable'] -= r['lost_selected_credits']
        assert not probe.decision(g)['rules']['all_seven_readable']
    g = passing_groups(); g['development/book/fixed_cells']['lost_nonempty_source_credits'] = 1
    assert not probe.decision(g)['rules']['credited_nonempty_sources_retained']


def test_previous_attempt_cannot_automatically_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'DEST', tmp_path); probe.ensure_first_execution()
    (tmp_path / 'attempts').mkdir()
    with pytest.raises(ValueError, match='No automatic retry'):
        probe.ensure_first_execution()

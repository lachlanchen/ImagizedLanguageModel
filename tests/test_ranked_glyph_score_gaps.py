"""Saved-score accounting without a neural replay or new character images."""
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts import analyze_ranked_glyph_score_gaps as analysis


def fixture():
    gold = np.zeros((6, 1, 32, 32), dtype=np.uint8)
    for i in range(1, 6):
        gold[i, 0, i, i] = 1
    emitted = gold.copy(); emitted[0, 0, 3, 4] = 1; emitted[1] = 0
    for i in range(3, 6):
        emitted[i] = 0; emitted[i, 0, i, i+1] = 1
    values = dict(source_logp=np.array([-3., -3., -2., -1., -2., -1.]),
                  emitted_logp=np.array([-2., -1., -2., -2., -1., -1.0004]),
                  source_blank=~gold.reshape(6, -1).any(1), emitted_blank=~emitted.reshape(6, -1).any(1),
                  saved_argmax_disagreements=np.zeros(6, dtype=np.int64))
    return gold, emitted, values


def test_complete_partition_keeps_blanks_matches_and_nonempty_mismatches_distinct():
    args = fixture()
    expected = dict(positions=6, source_whitespace=1, false_ink=1, false_blanks=1,
                    exact_bitmaps=1, nonempty_mismatches=3, source_preferred=1, emitted_preferred=1, tied=1)
    assert analysis.count_vectors(*args) == analysis.count_scalars(*args) == expected


@pytest.mark.parametrize('bad', ['nan', 'positive', 'flags', 'argmax', 'identical', 'shape', 'binary'])
def test_invalid_or_replaced_saved_evidence_is_rejected(bad):
    gold, emitted, values = fixture()
    if bad == 'nan': values['source_logp'][3] = np.nan
    elif bad == 'positive': values['emitted_logp'][3] = .1
    elif bad == 'flags': values['source_blank'][3] = True
    elif bad == 'argmax': values['saved_argmax_disagreements'][3] = 1
    elif bad == 'identical': values['source_logp'][2] -= .001
    elif bad == 'shape': values['source_logp'] = values['source_logp'][:-1]
    else: emitted[3, 0, 0, 0] = 2
    for count in [analysis.count_vectors, analysis.count_scalars]:
        with pytest.raises(ValueError): count(gold, emitted, values)


def test_numerical_ties_are_not_source_preference():
    gold, emitted, values = fixture()
    values['source_logp'][3:] = -2.
    values['emitted_logp'][3:] = -2.
    expected = analysis.count_vectors(gold, emitted, values)
    assert expected['tied'] == 3 and expected['source_preferred'] == expected['emitted_preferred'] == 0
    assert analysis.count_scalars(gold, emitted, values) == expected


def groups():
    row = dict(positions=3, source_whitespace=0, false_ink=0, false_blanks=0, exact_bitmaps=0,
               nonempty_mismatches=3, source_preferred=2, emitted_preferred=1, tied=0)
    return {key: dict(row) for key in analysis.GROUPS}


def test_each_book_group_requires_rounded_up_half_not_a_pooled_majority():
    data = groups(); assert analysis.decision(data)['source_preference_rule_passed']
    data[analysis.BOOKS[0]].update(source_preferred=1, emitted_preferred=2)
    result = analysis.decision(data)
    assert not result['source_preference_rule_passed']
    assert result['book_groups'][analysis.BOOKS[0]]['required'] == 2
    assert not result['autonomous_generation_authorized'] and not result['current_training_changed']


def test_zero_mismatch_group_and_missing_group_do_not_supply_evidence():
    data = groups();data[analysis.BOOKS[0]].update(nonempty_mismatches=0, source_preferred=0, emitted_preferred=0)
    assert not analysis.decision(data)['source_preference_rule_passed']
    data.pop(analysis.BOOKS[0])
    with pytest.raises(ValueError, match='seven groups'):analysis.decision(data)


def test_preference_counts_must_partition_mismatches():
    data = groups();data[analysis.BOOKS[0]]['source_preferred'] += 1
    with pytest.raises(ValueError, match='partition'):analysis.decision(data)


def test_import_has_no_torch_or_analysis_output_side_effect():
    code = """
import sys
from pathlib import Path
paths=[Path('artifacts/ranked_glyph_score_gaps_v1'),
       Path('references/evidence/ranked_glyph_score_gap_registration_20261116.json'),
       Path('references/evidence/ranked_glyph_score_gap_result_20261116.json')]
before=[p.exists() for p in paths]
from scripts import analyze_ranked_glyph_score_gaps as a
assert 'torch' not in sys.modules
assert [p.exists() for p in paths]==before
"""
    subprocess.run([sys.executable, '-c', code], check=True, cwd=analysis.ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))

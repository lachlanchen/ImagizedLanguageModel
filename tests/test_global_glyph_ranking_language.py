"""Actual-cohort admission, original error control and full learned-state restore."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import evaluate_global_glyph_ranking_language as evaluation


def good_header():
    return dict(schema='global-ranking-physical-gate-v1', passed=True,
        readable_groups_passed=True, whitespace_passed=True, credited_candidate_ink_passed=True,
        blank_flags_verified_against_opaque_pixels=True, source_whitespace=252,
        false_ink=11, maximum_false_ink=11, credited_empty_candidate_ids=[])


@pytest.mark.parametrize('change', [dict(whitespace_passed=False), dict(credited_candidate_ink_passed=False),
    dict(readable_groups_passed=False), dict(blank_flags_verified_against_opaque_pixels=False),
    dict(false_ink=12), dict(false_ink=-1), dict(false_ink=True), dict(source_whitespace=251),
    dict(maximum_false_ink=12), dict(credited_empty_candidate_ids=['a'*20]),
    dict(schema='joint-writer-physical-gate-v2')])
def test_missing_or_forged_safeguards_fail_before_import_or_output(tmp_path, monkeypatch, change):
    monkeypatch.setattr(evaluation, 'RUN', tmp_path)
    monkeypatch.setattr(evaluation, 'DEST', tmp_path / 'language')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    evaluation.write_new(tmp_path / 'qualify/gate.json', dict(good_header(), **change))
    with pytest.raises(ValueError):
        evaluation.register()
    assert not evaluation.DEST.exists()


def test_new_safeguard_rejection_precedes_torch_even_with_a_passed_flag(tmp_path):
    program = """
import os,sys
from pathlib import Path
from scripts import evaluate_global_glyph_ranking_language as e
e.RUN=Path(sys.argv[1]);e.DEST=e.RUN/'language'
e.write_new(e.RUN/'qualify/gate.json',dict(passed=True,schema='global-ranking-physical-gate-v1',readable_groups_passed=True,whitespace_passed=False))
try:e.register()
except ValueError as error:assert 'whitespace' in str(error)
else:raise AssertionError('Whitespace failure admitted')
assert 'torch' not in sys.modules and not e.DEST.exists()
"""
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=True, cwd=evaluation.ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_independent_whitespace_recount_uses_binary_arrays_and_hashes(tmp_path):
    import numpy as np
    p = tmp_path / 'cases/one/pixels.npz'; p.parent.mkdir(parents=True)
    gold = np.zeros((4, 1, 32, 32), dtype=np.uint8); gold[2:, 0, 5, 8] = 1
    candidate = gold.copy(); candidate[0, 0, 3, 4] = 1; candidate[2] = 0
    np.savez_compressed(p, gold_ink=gold, candidate_ink=candidate)
    row = dict(job=dict(id='one', split='development', kind='book', layout='fixed_cells'),
               target_glyphs=4, pixels_sha256=evaluation.sha(p))
    result = evaluation.whitespace_audit(tmp_path, dict(records=[row]), lambda: None)
    assert result['source_whitespace'] == 2 and result['false_ink'] == 1
    assert result['cases'][0]['false_blanks'] == 1 and result['cases'][0]['positions'] == 4
    np.savez_compressed(p, gold_ink=gold, candidate_ink=gold)
    with pytest.raises(ValueError, match='pixels changed'):
        evaluation.whitespace_audit(tmp_path, dict(records=[row]), lambda: None)
    row['pixels_sha256'] = evaluation.sha(p)
    assert evaluation.whitespace_audit(tmp_path, dict(records=[row]), lambda: None)['false_ink'] == 0


def test_a_credited_empty_image_cannot_be_laundered_by_blank_labels():
    assert evaluation.ink_flag(dict(text='中', uncertain=False, blank=False), 0)['credited_empty']
    assert evaluation.ink_flag(dict(text='', uncertain=False, blank=True), 1)['blank_flag_mismatch']


def test_binding_matches_the_registered_immutable_review_wrapper(monkeypatch):
    """Exercise both binding producers with one original-review result."""
    from scripts import review_global_glyph_ranking as registered_review
    original_values = (evaluation.RUN / 'qualify', {}, {}, {}, [], {'original': 'source-bound'})
    monkeypatch.setattr(evaluation.physical, 'context', lambda: original_values)
    with evaluation.review_scope() as selected:
        actual = selected.context()
    monkeypatch.setattr(registered_review, '_original_context', lambda: original_values)
    expected = registered_review.context()
    assert actual == expected
    assert actual[-1]['ranking_review_source_sha256'] == registered()['source_sha256'][
        'scripts/review_global_glyph_ranking.py']


def registered():
    return evaluation.load(evaluation.TRAINING_PUBLIC)


def test_complete_actual_plan_and_import_do_not_load_neural_runtime_or_mutate_old_scope():
    program = """
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import review_joint_writer_physical as old
before=(old.RUN,old.PUBLIC)
from scripts import evaluate_global_glyph_ranking_language as e
from scripts import review_global_glyph_ranking_language
assert (old.RUN,old.PUBLIC)==before
arms=e.plan(e.load(e.TRAINING_PUBLIC))
assert [(a['step'],len(a['inputs']),a['maximum_glyphs']) for a in arms]==[(0,216,128),(0,24,640),(10324,76,128),(20649,76,128),(30973,76,128),(41298,216,128),(41298,24,640)]
assert sum(len(a['inputs']) for a in arms)==708
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', program], check=True, cwd=evaluation.ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))


def test_actual_cohort_cannot_drop_erasure_or_checkpoint():
    for mutate in [lambda r: r['evaluation_inputs'].pop(), lambda r: r['snapshot_steps'].pop(),
                   lambda r: r['evaluation_inputs'].__setitem__(0, r['evaluation_inputs'][1])]:
        value = copy.deepcopy(registered()); mutate(value)
        with pytest.raises(ValueError):
            evaluation.plan(value)


def test_absent_or_failed_gate_creates_no_generation_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'RUN', tmp_path)
    monkeypatch.setattr(evaluation, 'DEST', tmp_path / 'language')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='not yet complete'):
        evaluation.register()
    evaluation.write_new(tmp_path / 'qualify/gate.json', dict(passed=False))
    with pytest.raises(ValueError, match='Physical gate failed'):
        evaluation.register()
    assert not (tmp_path / 'language').exists()


def test_missing_gate_rejection_precedes_neural_import_in_fresh_process(tmp_path):
    program = """
import os,sys
from pathlib import Path
from scripts import evaluate_global_glyph_ranking_language as e
e.RUN=Path(sys.argv[1]);e.DEST=e.RUN/'language'
try:e.register()
except ValueError as error:assert 'not yet complete' in str(error)
else:raise AssertionError('Unqualified registration admitted')
assert 'torch' not in sys.modules and not e.DEST.exists()
"""
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=True, cwd=evaluation.ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_restore_all_actual131_tensors_after_inference_freezing_preserves_other_parameters():
    """CPU storage fixture uses every real learned tensor, with no model forward."""
    import torch
    path = evaluation.RUN / 'fit/snapshots/step_000000.pt'
    if not path.exists():
        pytest.skip('Actual ranking snapshot unavailable')
    value = torch.load(path, weights_only=True, map_location='cpu')
    actor = torch.nn.Module()
    for name, tensor in value['trainable_weights'].items():
        owner = actor; parts = name.split('.')
        for part in parts[:-1]:
            if not hasattr(owner, part):
                owner.add_module(part, torch.nn.Module())
            owner = getattr(owner, part)
        owner.register_parameter(parts[-1], torch.nn.Parameter(tensor.clone()))
    actor.register_parameter('frozen_control', torch.nn.Parameter(torch.tensor([3.]), requires_grad=False))
    with torch.no_grad():
        for name, p in actor.named_parameters():
            if name != 'frozen_control':
                p.add_(1)
    actor.requires_grad_(False)
    inventory = evaluation.restore_snapshot(actor, value, 0, registered())
    assert len(inventory) == 131 and sum(x['elements'] for x in inventory.values()) == 5005760
    assert torch.equal(actor.frozen_control, torch.tensor([3.]))
    assert not any(p.requires_grad for p in actor.parameters())
    for name, p in actor.named_parameters():
        if name != 'frozen_control':
            assert torch.equal(p, value['trainable_weights'][name])


def test_actual_context_snapshot_rejects_legacy_head_only_and_wrong_provenance():
    import torch
    path = evaluation.RUN / 'fit/snapshots/step_000000.pt'
    if not path.exists():
        pytest.skip('Actual ranking snapshot unavailable')
    r = registered(); value = torch.load(path, weights_only=True, map_location='cpu')
    evaluation.check_snapshot(value, 0, r)
    bad = dict(value, parent_checkpoint=dict(file='wrong.pt', sha256='f' * 64))
    with pytest.raises(ValueError, match='provenance'):
        evaluation.check_snapshot(bad, 0, r)
    with pytest.raises(ValueError, match='step'):
        evaluation.check_snapshot(value, 41298, r)
    bad = dict(value, schema='joint-context-checkpoint-v1')
    with pytest.raises(ValueError, match='provenance'):
        evaluation.check_snapshot(bad, 0, r)


def test_scoped_physical_and_complete_review_restore_original_modules_even_on_exception():
    from scripts import review_joint_writer_physical as original
    before = original.RUN, original.PUBLIC, original.context
    with pytest.raises(RuntimeError):
        with evaluation.review_scope() as selected:
            assert selected.RUN == evaluation.RUN and selected.PUBLIC == evaluation.TRAINING_PUBLIC
            raise RuntimeError('bounded scope exit')
    assert (original.RUN, original.PUBLIC, original.context) == before
    from scripts import review_global_glyph_ranking_language as wrapper
    from scripts import review_joint_writer_language as old
    before_complete = old.evaluation, old.ROOT, old.DEST, old.DIRECTORY
    with wrapper.review_scope() as selected:
        assert selected.evaluation is evaluation and selected.DEST == evaluation.DEST
    assert (old.evaluation, old.ROOT, old.DEST, old.DIRECTORY) == before_complete

"""White-target full-cohort admission and actual learned-tensor restoration."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import evaluate_white_target_language as evaluation


def good_header():
    return dict(schema='white-target-physical-gate-v1', passed=True,
        readable_groups_passed=True, whitespace_passed=True, credited_candidate_ink_passed=True,
        blank_flags_verified_against_opaque_pixels=True, source_whitespace=252,
        false_ink=11, maximum_false_ink=11, false_blanks=37, maximum_false_blanks=37,
        false_blanks_passed=True, credited_empty_candidate_ids=[])


@pytest.mark.parametrize('change', [dict(whitespace_passed=False), dict(credited_candidate_ink_passed=False),
    dict(readable_groups_passed=False), dict(blank_flags_verified_against_opaque_pixels=False),
    dict(false_ink=12), dict(false_ink=-1), dict(false_ink=True), dict(source_whitespace=251),
    dict(maximum_false_ink=12), dict(credited_empty_candidate_ids=['a'*20]),
    dict(schema='global-ranking-physical-gate-v1'), dict(false_blanks=38), dict(false_blanks=-1),
    dict(false_blanks=True), dict(false_blanks=37.), dict(maximum_false_blanks=38),
    dict(false_blanks_passed=False), dict(false_blanks_passed=None)])
def test_missing_or_forged_safeguards_fail_before_import_or_output(tmp_path, monkeypatch, change):
    monkeypatch.setattr(evaluation, 'RUN', tmp_path)
    monkeypatch.setattr(evaluation, 'DEST', tmp_path / 'language')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    evaluation.write_new(tmp_path / 'qualify/gate.json', dict(good_header(), **change))
    with pytest.raises(ValueError):
        evaluation.register()
    assert not evaluation.DEST.exists()


def test_false_blank_rejection_precedes_torch_even_with_all_passed_flags(tmp_path):
    program = """
import json,os,sys
from pathlib import Path
from scripts import evaluate_white_target_language as e
e.RUN=Path(sys.argv[1]);e.DEST=e.RUN/'language'
gate=json.loads(sys.argv[2]);gate['false_blanks']=38
e.write_new(e.RUN/'qualify/gate.json',gate)
try:e.register()
except ValueError as error:assert 'false-blank' in str(error)
else:raise AssertionError('False-blank failure admitted')
assert 'torch' not in sys.modules and not e.DEST.exists()
"""
    subprocess.run([sys.executable, '-c', program, str(tmp_path), json.dumps(good_header())], check=True, cwd=evaluation.ROOT,
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
    assert result['positions'] == 4 and result['false_blanks'] == 1
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
    from scripts import review_white_target_physical as registered_review
    original_values = (evaluation.RUN / 'qualify', {}, {}, {}, [], {'original': 'source-bound'})
    monkeypatch.setattr(evaluation.physical, 'context', lambda: original_values)
    with evaluation.review_scope() as selected:
        actual = selected.context()
    monkeypatch.setattr(registered_review, '_original_context', lambda: original_values)
    expected = registered_review.context()
    assert actual == expected
    assert actual[-1]['white_target_review_source_sha256'] == registered()['source_sha256'][
        'scripts/review_white_target_physical.py']


def registered():
    return evaluation.load(evaluation.TRAINING_PUBLIC)


def test_complete_actual_plan_and_import_do_not_load_neural_runtime_or_mutate_old_scope():
    program = """
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import review_joint_writer_physical as old
before=(old.RUN,old.PUBLIC)
from scripts import evaluate_white_target_language as e
from scripts import review_white_target_language
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
from scripts import evaluate_white_target_language as e
e.RUN=Path(sys.argv[1]);e.DEST=e.RUN/'language'
try:e.register()
except ValueError as error:assert 'not yet complete' in str(error)
else:raise AssertionError('Unqualified registration admitted')
assert 'torch' not in sys.modules and not e.DEST.exists()
"""
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=True, cwd=evaluation.ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_restore_actual131_tensors_across_two_saved_states_after_inference_freezing():
    """Use real saved tensors as CPU storage, never instantiate the inherited body."""
    import torch
    path = evaluation.RUN / 'fit/snapshots/step_000000.pt'
    if not path.exists():
        pytest.skip('Actual white-target snapshot unavailable')
    assert evaluation.sha(path) == evaluation.load(evaluation.RUN / 'fit/snapshots.json')['0']['sha256']
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
    actor.register_buffer('frozen_affine_control', torch.tensor([7.]))
    versions = actor.frozen_control._version, actor.frozen_affine_control._version
    # Step5 is used only as a nonidentical storage-restoration fixture. It is not
    # a permitted evaluation snapshot or a new replay/quality selection.
    learned_path = evaluation.RUN / 'fit/resume_control5.pt'
    assert learned_path.exists(), 'Actual immutable saved step5 is required here'
    assert evaluation.sha(learned_path) == evaluation.load(
        evaluation.RUN / 'fit/resume_control.json')['uninterrupted_file_sha256']
    learned = torch.load(learned_path, weights_only=True, map_location='cpu')
    assert any(not torch.equal(value['trainable_weights'][n], learned['trainable_weights'][n])
               for n in value['trainable_weights'])
    for saved in [value, learned, value]:
        inventory = evaluation.restore_snapshot(actor, saved, saved['step'], registered())
        assert len(inventory) == 131 and sum(x['elements'] for x in inventory.values()) == 5005760
        assert torch.equal(actor.frozen_control, torch.tensor([3.]))
        assert torch.equal(actor.frozen_affine_control, torch.tensor([7.]))
        assert (actor.frozen_control._version, actor.frozen_affine_control._version) == versions
        assert not any(p.requires_grad for p in actor.parameters())
        for name, p in actor.named_parameters():
            if name != 'frozen_control':
                assert torch.equal(p, saved['trainable_weights'][name])


def test_actual_context_snapshot_rejects_legacy_head_only_and_wrong_provenance():
    import torch
    path = evaluation.RUN / 'fit/snapshots/step_000000.pt'
    if not path.exists():
        pytest.skip('Actual white-target snapshot unavailable')
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
    for changes in [dict(white_strength=0.), dict(ranking_strength=0.),
                    dict(margin_nats=0.), dict(objective='old nonempty-only ranking')]:
        with pytest.raises(ValueError, match='objective'):
            evaluation.check_snapshot(dict(value, **changes), 0, r)


def counts_fixture():
    # Pure accounting fixture, not a claim about generated model outputs.
    row = dict(case='fixture', split='development', kind='book', layout='fixed_cells',
               positions=12597, source_whitespace=252, false_ink=11, false_blanks=37)
    white = dict(cases=[row], **{k: row[k] for k in
                 ['positions', 'source_whitespace', 'false_ink', 'false_blanks']})
    gate = dict(good_header(), complete_case_counts=[dict(row)])
    return white, gate, registered()


def test_recount_rejects_a_good_header_when_actual_false_blanks_exceed_limit():
    white, gate, registration = counts_fixture()
    evaluation.check_complete_counts(white, gate, registration)
    white['false_blanks'] = white['cases'][0]['false_blanks'] = 38
    with pytest.raises(ValueError, match='recount differs'):
        evaluation.check_complete_counts(white, gate, registration)
    gate['false_blanks'] = 38
    gate['complete_case_counts'][0]['false_blanks'] = 38
    with pytest.raises(ValueError, match='false-blank safeguard'):
        evaluation.check_complete_counts(white, gate, registration)


@pytest.mark.parametrize('change', [dict(positions=12596), dict(source_whitespace=251),
                                  dict(false_ink=12)])
def test_partial_cohort_and_whitespace_failures_cannot_pass_recount(change):
    white, gate, registration = counts_fixture()
    for key, value in change.items():
        white[key] = white['cases'][0][key] = value
        if key in gate:
            gate[key] = value
        gate['complete_case_counts'][0][key] = value
    with pytest.raises(ValueError):
        evaluation.check_complete_counts(white, gate, registration)


def test_thresholds_cannot_be_relaxed_by_a_new_header():
    for changes in [dict(maximum_false_blanks=38), dict(maximum_false_ink_on_source_whitespace=12)]:
        white, gate, registration = counts_fixture()
        registration.update(changes)
        with pytest.raises(ValueError, match='safeguard'):
            evaluation.check_complete_counts(white, gate, registration)


def test_scoped_physical_and_complete_review_restore_original_modules_even_on_exception():
    from scripts import review_joint_writer_physical as original
    before = original.RUN, original.PUBLIC, original.context
    with pytest.raises(RuntimeError):
        with evaluation.review_scope() as selected:
            assert selected.RUN == evaluation.RUN and selected.PUBLIC == evaluation.TRAINING_PUBLIC
            raise RuntimeError('bounded scope exit')
    assert (original.RUN, original.PUBLIC, original.context) == before
    from scripts import review_white_target_language as wrapper
    from scripts import review_joint_writer_language as old
    before_complete = old.evaluation, old.ROOT, old.DEST, old.DIRECTORY
    with wrapper.review_scope() as selected:
        assert selected.evaluation is evaluation and selected.DEST == evaluation.DEST
    assert (old.evaluation, old.ROOT, old.DEST, old.DIRECTORY) == before_complete

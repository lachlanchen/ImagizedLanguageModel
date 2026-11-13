"""Complete real-cohort planning and failure-preserving evaluation admission."""
import copy
import json
from pathlib import Path

import pytest

from scripts import evaluate_joint_writer_language as evaluation


def registration():
    return json.loads(evaluation.review.PUBLIC.read_text())


def test_real_registered_cohort_covers_every_snapshot_control_and_long_book():
    r = registration()
    arms = evaluation.plan(r)
    assert [(x['step'], len(x['inputs']), x['maximum_glyphs']) for x in arms] == [
        (0, 216, 128), (0, 24, 640), (10324, 76, 128), (20649, 76, 128),
        (30973, 76, 128), (41298, 216, 128), (41298, 24, 640)]
    assert sum(len(x['inputs']) for x in arms if x['maximum_glyphs'] == 128) == 660
    assert sum(len(x['inputs']) for x in arms if x['maximum_glyphs'] == 640) == 48
    for arm in arms:
        assert all(row in r['evaluation_inputs'] for row in arm['inputs'])
        if arm['step'] not in [0, 41298]:
            assert all(row['condition'] == 'normal' for row in arm['inputs'])


def test_missing_duplicate_control_or_missing_checkpoint_cannot_shrink_comparison():
    r = registration()
    for mutate in [lambda x: x['evaluation_inputs'].pop(),
                   lambda x: x['evaluation_inputs'].__setitem__(0, x['evaluation_inputs'][1]),
                   lambda x: x['snapshot_steps'].pop()]:
        value = copy.deepcopy(r)
        mutate(value)
        with pytest.raises(ValueError):
            evaluation.plan(value)
    value = copy.deepcopy(r)
    next(x for x in value['evaluation_inputs'] if x['condition'] == 'passage_blank')['condition'] = 'normal'
    with pytest.raises(ValueError, match='erasures'):
        evaluation.plan(value)


def test_failed_or_absent_physical_gate_refuses_registration_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'RUN', tmp_path)
    monkeypatch.setattr(evaluation, 'DEST', tmp_path / 'language')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='not yet complete'):
        evaluation.register()
    (tmp_path / 'qualify').mkdir()
    evaluation.write_new(tmp_path / 'qualify/gate.json', dict(passed=False))
    with pytest.raises(ValueError, match='Physical gate failed'):
        evaluation.register()
    assert not (tmp_path / 'language').exists()


def test_budget_reuses_all_failed_attempt_costs_and_refuses_unknown_interruption(tmp_path):
    assert evaluation.remaining_budget(tmp_path, 100, 20) == (100, 20)
    first = tmp_path / 'attempts/001'
    evaluation.write_new(first / 'start.json', dict(pid=1))
    with pytest.raises(ValueError, match='unterminated'):
        evaluation.remaining_budget(tmp_path, 100, 20)
    evaluation.write_new(first / 'end.json', dict(elapsed_seconds=35, resource_wait_seconds=7))
    assert evaluation.remaining_budget(tmp_path, 100, 20) == (65, 13)
    second = tmp_path / 'attempts/002'
    evaluation.write_new(second / 'start.json', dict(pid=2))
    evaluation.write_new(second / 'end.json', dict(elapsed_seconds=65, resource_wait_seconds=3))
    with pytest.raises(ValueError, match='budget exhausted'):
        evaluation.remaining_budget(tmp_path, 100, 20)


def test_checkpoint_must_retain_actual_final_reader_and_registered_head_step(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'RUN', tmp_path)
    evaluation.write_new(tmp_path / 'cache/report.json', dict(complete=True))
    r = registration()
    e = dict(joint_registration_sha256='a' * 64, parent_checkpoint=r['parent_checkpoint'], runtime=r['runtime'])
    value = dict(schema='joint-continuation-head-checkpoint-v1', step=0,
                 registration_sha256=e['joint_registration_sha256'], parent_checkpoint=e['parent_checkpoint'],
                 runtime=e['runtime'], source_sha256=r['source_sha256'],
                 cache_report_sha256=evaluation.sha(tmp_path / 'cache/report.json'),
                 completed_training=False, only_head_trainable=True)
    evaluation.check_head(value, 0, r, e)
    with pytest.raises(ValueError, match='Wrong registered head'):
        evaluation.check_head(value, 41298, r, e)
    value['parent_checkpoint'] = dict(file='another-reader.pt', sha256='b' * 64)
    with pytest.raises(ValueError, match='provenance'):
        evaluation.check_head(value, 0, r, e)

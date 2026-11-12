"""Fail-closed accounting and resource boundaries for real evaluator recovery."""
import json
import subprocess
import sys

import pytest

from scripts.recover_visual_continuation_evaluation import admitted, budget_seconds, prior_costs, stop_owned


def test_failed_admission_counts_and_interrupted_cost_is_not_reset(tmp_path):
    path = tmp_path / 'children.jsonl'
    rows = [dict(event='started', action='evaluate', child_pid=17),
            dict(event='ended', action='evaluate', child_pid=17, elapsed_seconds=6.52, returncode=1)]
    path.write_text('\n'.join(map(json.dumps, rows)) + '\n')
    costs = prior_costs([path])
    assert costs['evaluate'] == 6.52
    assert budget_seconds(21600, costs['evaluate']) == 21600 - 6.52 - 10
    with path.open('a') as f:
        f.write(json.dumps(dict(event='started', action='evaluate', child_pid=18)) + '\n')
    with pytest.raises(RuntimeError, match='Unaccounted interrupted child'):
        prior_costs([path])


def test_exhausted_or_invalid_budget_cannot_launch():
    for spent in (21600, 21595, -1, float('nan')):
        with pytest.raises(RuntimeError):
            budget_seconds(21600, spent)


def test_headroom_does_not_weaken_registered_resource_limits():
    value = dict(available_gib=50, gpu_free_mib=20480, mutex_free=True, some_avg10=.5, full_avg10=.1)
    assert admitted(value, 48, 20480)
    for change in (dict(available_gib=49.99), dict(gpu_free_mib=20479), dict(mutex_free=False),
                   dict(some_avg10=.501), dict(full_avg10=.101)):
        assert not admitted(dict(value, **change), 48, 20480)


def test_independent_parent_cost_and_multiple_attempts_are_accumulated(tmp_path):
    paths = []
    for i, (action, seconds) in enumerate([('evaluate', 6.52), ('evaluate', 15), ('parent-long-books', 21)]):
        path = tmp_path / f'journal_{i}.jsonl'
        path.write_text(json.dumps(dict(event='started', action=action, child_pid=i)) + '\n' +
                        json.dumps(dict(event='ended', action=action, child_pid=i, elapsed_seconds=seconds)) + '\n')
        paths.append(path)
    assert prior_costs(paths) == {'evaluate': 21.52, 'parent-long-books': 21}


def test_real_owned_process_cleanup_and_foreign_group_refusal():
    for owned in (True, False):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=owned)
        try:
            if owned:
                stop_owned(child)
                assert child.poll() is not None
            else:
                with pytest.raises(RuntimeError, match='unowned process group'):
                    stop_owned(child)
                assert child.poll() is None
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)

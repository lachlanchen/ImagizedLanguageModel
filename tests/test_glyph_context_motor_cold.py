import pytest
import torch

from scripts.verify_glyph_context_motor_cold import equal_tree, opened_paths


def test_motor_replay_compares_nested_noise_and_unused_cap_sample():
    original = {'images':torch.ones(2, 1, 32, 32), 'motor_trace':[
        {'noise':torch.zeros(1, 1024), 'normalized_motor':torch.ones(1, 1024)} for _ in range(3)],
        'stopped':False, 'capped':True}
    import copy
    replay = copy.deepcopy(original)
    equal_tree(original, replay)
    replay['motor_trace'][-1]['noise'][0, 17] = 1
    with pytest.raises((ValueError, RuntimeError), match='motor_trace.*noise'):
        equal_tree(original, replay)
    replay = copy.deepcopy(original); del replay['capped']
    with pytest.raises((ValueError, RuntimeError), match='fields'):
        equal_tree(original, replay)


def test_access_paths_resolve_directory_fds_and_preserve_failed_attempts(tmp_path):
    work = tmp_path/'empty'; work.mkdir()
    (tmp_path/'access.1').write_text(
        f'openat(AT_FDCWD, "missing.pt", O_RDONLY) = -1 ENOENT\n'
        f'openat(3<{tmp_path}>, "forbidden.pt", O_RDONLY) = 4\n'
        f'openat2(AT_FDCWD, "{tmp_path}/escaped\\040name.pt", {{flags=O_RDONLY}}, 24) = 4\n')
    assert opened_paths(tmp_path, work) == {
        work/'missing.pt', tmp_path/'forbidden.pt', tmp_path/'escaped name.pt'}

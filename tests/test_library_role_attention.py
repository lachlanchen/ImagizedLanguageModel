from itertools import permutations

import torch

from ilm.visual_lm.role_attention import compare_assignments, window_roles


def test_permutation_null_with_overlapping_windows_and_padding():
    src = torch.tensor([[9, 8, 7, 6, 4, 0]])
    tags = torch.tensor([[1, 0, 3, 1, 2, 0]])
    target = torch.tensor([[1, 2, 3, 0]])
    weights = torch.tensor([[[.10, .25, .30, .20, .15, 0.]]]).expand(-1, 4, -1)
    for stride in (1, 4):
        weight = weights if stride == 1 else torch.tensor([[[.7, .3]]]).expand(-1, 4, -1)
        fractions = window_roles(src, tags, stride)
        mass = weight @ fractions
        assert torch.allclose(mass.sum(-1), torch.ones((1, 4)))
        observed, expected, valid = compare_assignments(mass, tags, target)
        assert valid.tolist() == [[True, True, True, False]]
        measured = []
        for assignment in permutations([1, 3, 1, 2]):
            shuffled = tags.clone()
            shuffled[0, [0, 2, 3, 4]] = torch.tensor(assignment)
            actual, _, _ = compare_assignments(weight @ window_roles(src, shuffled, stride),
                                                shuffled, target)
            measured.append(actual)
        assert torch.allclose(expected[valid], torch.stack(measured).mean(0)[valid])
        assert torch.isfinite(observed).all()


def test_no_tagged_source_excludes_all_positions():
    source_tags = torch.zeros(1, 3, dtype=torch.long)
    target_tags = torch.tensor([[0, 1, 2, 3]])
    same, null, eligible = compare_assignments(torch.tensor([[[1., 0., 0., 0.]]]).expand(-1, 4, -1),
                                               source_tags, target_tags)
    assert not eligible.any()
    assert torch.isfinite(null).all() and torch.isfinite(same).all()

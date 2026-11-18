import numpy as np
import pytest
import torch

from ilm.visual_lm.contextual_ink_binding import ContextualInkBinding, InkPermutation


def packed(*numbers):
    return np.stack([np.full(128, n, dtype=np.uint8) for n in numbers])


def test_disjoint_transpositions_preserve_observed_ink():
    images = packed(1, 2, 3, 4, 5)
    a, b, c, d, _ = map(bytes, images)
    phi = InkPermutation([(a, b), (c, d)], "cpu")
    bits = torch.from_numpy(np.unpackbits(images, axis=1)).bool()[None]
    assert torch.equal(phi(bits), bits[:, [1, 0, 3, 2, 4]])
    assert torch.equal(phi(phi(bits)), bits)
    with pytest.raises(ValueError):
        InkPermutation([(a, b), (b, c)], "cpu")
    with pytest.raises(ValueError):
        InkPermutation([(a, b, c)], "cpu")


def test_familiar_predicate_selects_two_novel_roles_and_ignores_novel_predicate(
    tmp_path,
):
    original = packed(1, 2, 3, 4)
    fit = ContextualInkBinding.fit([], [(original, (0, 2))] * 3)
    path = tmp_path / "binding.gz"
    fit.save(path)
    restored = ContextualInkBinding.restore(path)
    test = packed(7, 2, 8, 4)
    assert restored.choose(test) == [
        (bytes(test[0]), bytes(original[0])),
        (bytes(test[2]), bytes(original[2])),
    ]
    assert restored.choose(packed(1, 7, 3, 4)) == []
    assert restored.choose(original) == []

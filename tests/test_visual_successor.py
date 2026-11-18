import numpy as np
import torch

from ilm.visual_lm.visual_successor import VisualSuccessor
from scripts.measure_visual_successors import assignment, train_batch


def test_scoring_is_physical_causal_and_independent_of_candidate_order():
    torch.manual_seed(2)
    model = VisualSuccessor().eval()
    ink = torch.randint(2, (2, 34, 32, 32), dtype=torch.uint8)
    keys = ink[:, :2]
    queries = model.queries(model.shape(model.pool(ink)))
    for i in range(3):
        direct = model(ink[:, i:i+32], keys)
        keys_encoded = model.keys(model.shape(model.pool(keys)))
        expected = 10 * (queries[:, i:i+1] @ keys_encoded.transpose(1, 2)).squeeze(1)
        torch.testing.assert_close(direct, expected)
        torch.testing.assert_close(direct.flip(1), model(ink[:, i:i+32], keys.flip(1)))


def test_repeated_targets_singleton_and_rotated_matches():
    model = VisualSuccessor()
    bank = torch.randn(5, 256)
    chunks = [([1], [2, 2, 2]), ([2], [1])]
    loss, n, changed = train_batch(model, bank, chunks, True, np.random.default_rng(1))
    assert n == 4 and changed == 0 and loss.item() == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_assignment_removes_history_and_candidate_biases():
    rng = np.random.default_rng(6)
    scores = rng.normal(size=(40, 2, 2))
    biased = scores + rng.normal(size=(40, 2, 1)) + rng.normal(size=(40, 1, 2))
    assert np.array_equal(assignment(scores)[0], assignment(biased)[0])
    assert assignment(scores.repeat(2, axis=1)[:, :2])[1]['ties'] == 40

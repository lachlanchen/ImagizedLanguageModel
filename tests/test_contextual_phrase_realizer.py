import numpy as np
import pytest
import torch

from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer, fit_head
from scripts.train_predicate_likelihood import choice_log_mass


def test_whole_pair_mass_counts_each_physical_copy_and_carries_gradient():
    routes = torch.tensor([[0.5, 0.4, 0.05, 0.05]], requires_grad=True)
    scores = choice_log_mass(
        routes.log(),
        torch.tensor([[0.2, 0.3, 0.5]]).log(),
        torch.tensor([[0.25, 0.75]]).log(),
        torch.tensor([[4, 4, 5]]),
        torch.tensor([[False, False, True]]),
        torch.tensor([[4, 5]]),
        torch.tensor([-1, -1, -1, -1, 0, 1]),
    )
    assert torch.allclose(scores.exp(), torch.tensor([[0.35, 0.30]]))
    scores.sum().backward()
    assert routes.grad is not None and routes.grad.isfinite().all()


def test_saved_phrase_selector_preserves_an_unknown_neighbour(tmp_path):
    class Copy:
        def generate(self, x):
            return x.copy()

    center = bytes([5]) * 128
    option = bytes([3]) * 128 + bytes([2]) * 128
    right = bytes([255]) * 128
    x = np.frombuffer(center + right, np.uint8).reshape(-1, 128)
    samples = [
        (np.array([s], np.float32), np.array([s], np.float32), v)
        for s, v in [(-1, None), (1, option)]
    ]
    head = fit_head(samples)
    writer = ContextualPhraseRealizer(torch.nn.Identity(), Copy(), {center: head})
    path = tmp_path / "physical.gz"
    writer.save(path)
    restored = ContextualPhraseRealizer.restore(path, writer.encoder, Copy())
    assert restored.generate(x, majority=True).tobytes() == option + right
    assert restored.generate(x[1:]).tobytes() == right
    with pytest.raises(ValueError):
        fit_head([samples[1]])

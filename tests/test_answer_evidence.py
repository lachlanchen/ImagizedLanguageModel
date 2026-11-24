import torch

from ilm.visual_lm.answer_evidence import answer_marginal


def test_unsupported_path_and_complete_answer_gradient():
    logits = torch.tensor([0.5, -0.7, 1.3], dtype=torch.float64, requires_grad=True)
    likelihood = torch.tensor([0.0, 0.06, 0.29], dtype=torch.float64)
    loss = -answer_marginal(logits, likelihood.log())
    prior = logits.detach().softmax(-1)
    marginal = (prior * likelihood).sum()
    expected = prior - prior * likelihood / marginal
    torch.testing.assert_close(loss, -marginal.log())
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], expected)
    assert answer_marginal(logits, torch.full_like(logits, -torch.inf)).isneginf()

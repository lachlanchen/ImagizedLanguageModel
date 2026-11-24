import torch
from ilm.visual_lm.fact_channel import OpticalFactReader, span_scores


def test_normalizer_and_gradient_match_legal_spans():
    start = torch.tensor(
        [[0.3, -0.2, 0.1, 0.6, -0.3]], dtype=torch.float64, requires_grad=True
    )
    end = torch.tensor(
        [[0.7, 0.4, 0.2, 0.1, 0.8]], dtype=torch.float64, requires_grad=True
    )
    lengths = torch.tensor([0.0, -0.2, 0.3], dtype=torch.float64, requires_grad=True)
    cells = torch.tensor([[4, 3, 0, 4, 3]])
    scores = span_scores(start, end, cells, lengths)
    aliases = torch.tensor([[1, 3 * 3 + 1]])
    loss = OpticalFactReader().loss(scores, aliases).sum()
    legal = [
        start[0, i] + end[0, j] + lengths[j - i]
        for i in range(5)
        for j in range(i, min(i + 3, 5))
        if cells[0, i : j + 1].ne(0).all()
    ]
    correct = [start[0, i] + end[0, i + 1] + lengths[1] for i in [0, 3]]
    independent = torch.stack(legal).logsumexp(0) - torch.stack(correct).logsumexp(0)
    torch.testing.assert_close(loss, independent)
    weights = (start, end, lengths)
    for a, b in zip(
        torch.autograd.grad(loss, weights), torch.autograd.grad(independent, weights)
    ):
        torch.testing.assert_close(a, b)
    assert scores.isfinite().sum() == len(legal)


def test_reader_scores_are_independent_of_batch_padding():
    torch.manual_seed(7)
    reader = OpticalFactReader().eval()
    features = torch.randn(13, 256)
    a = reader(features, torch.tensor([[2, 3]]), torch.tensor([[4, 0, 8, 9]]))
    b = reader(
        features,
        torch.tensor([[2, 3, 0], [5, 5, 6]]),
        torch.tensor([[4, 0, 8, 9, 0, 0], [6, 11, 3, 5, 4, 2]]),
    )
    torch.testing.assert_close(a[0], b[0, :4, :4], atol=2e-6, rtol=2e-6)

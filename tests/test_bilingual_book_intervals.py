import torch
from scripts.train_bilingual_book_intervals import interval_loss


def test_loss_and_gradients_include_only_valid_intervals_and_all_occurrences():
    a = torch.tensor([[1.0, -1.0, 0.0], [0.3, 0.2, 90.0]], requires_grad=True)
    b = torch.tensor([[-2.0, 2.0, 1.0], [0.4, 0.1, 80.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    wanted = [[(0, 1), (1, 2)], [(1, 1)]]
    actual = interval_loss(a, b, mask, wanted)
    expected = []
    for i, length in enumerate((3, 2)):
        all_ = torch.stack(
            [a[i, j] + b[i, k] for j in range(length) for k in range(j, length)]
        )
        gold = torch.stack([a[i, j] + b[i, k] for j, k in wanted[i]])
        expected.append(all_.logsumexp(0) - gold.logsumexp(0))
    expected = torch.stack(expected).mean()
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, (a, b)),
        torch.autograd.grad(expected, (a, b)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)

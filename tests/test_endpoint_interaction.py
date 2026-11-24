import torch
from ilm.visual_lm.endpoint_interaction import EndpointWriter
from ilm.visual_lm.paired_evidence import EvidenceWriter


def test_initial_correction_preserves_all_reader_outputs():
    torch.manual_seed(31)
    base = EvidenceWriter()
    bank = torch.randn(20, 80)
    q = torch.tensor([[3, 4, 5]])
    s = torch.tensor([[3, 6, 8, 2, 0]])
    for mode in ("additive", "product"):
        net = EndpointWriter(mode)
        net.load_state_dict(base.state_dict(), strict=False)
        for a, b in zip(net.read(bank, q, s), base.read(bank, q, s), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_banded_scores_compare_the_intended_endpoints():
    torch.manual_seed(99)
    key = torch.randn(2, 9, 80, requires_grad=True)
    for mode in ("additive", "product"):
        net = EndpointWriter(mode)
        left = net.left(key).tanh()
        right = net.right(key).tanh()
        band = net.correction(key)
        for start, end in ((0, 0), (0, 7), (2, 5), (8, 8)):
            a, b = left[:, start], right[:, end]
            value = (a * b if mode == "product" else a + b).sum(-1) / left.size(
                -1
            ) ** 0.5
            torch.testing.assert_close(band[:, start, end - start], value)
        square = band[:, 1, 4] + band[:, 2, 4] - band[:, 1, 5] - band[:, 2, 3]
        if mode == "additive":
            torch.testing.assert_close(
                square, torch.zeros_like(square), atol=1e-6, rtol=0
            )
        else:
            assert (square.abs() > 1e-4).all()
        band[:, 2, 3].sum().backward()
        assert net.left.weight.grad.norm() > 0 and net.right.weight.grad.norm() > 0

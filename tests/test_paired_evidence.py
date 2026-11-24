import numpy as np
import torch
from ilm.visual_lm.paired_evidence import compatible, LENGTH, EvidenceWriter
from scripts.write_paired_evidence import next_distribution


def test_duplicate_starts_prefix_and_stop():
    s = torch.tensor([[2, 3, 2, 3, 4]] * 3)
    y = torch.tensor([[2, 3, 1], [2, 3, 0], [3, 4, 1]])
    matches = compatible(s, y)
    assert set(map(tuple, matches[0].nonzero().tolist())) == {(0, 1), (2, 1)}
    assert set(map(tuple, matches[1].nonzero().tolist())) == {
        (0, i) for i in range(1, 5)
    } | {(2, 1), (2, 2)}
    assert matches[2].sum() == 1 and matches[2, 3, 1]


def test_ink_prefix_distribution_matches_enumeration():
    # Two occurrences spell the same complete output, another extends the prefix.
    live = torch.zeros(1, 4, LENGTH)
    live[0, 0, 0] = 0.2
    live[0, 2, 0] = 0.1
    live[0, 0, 1] = 0.3
    free = torch.tensor([0.4])
    f = torch.tensor([[0.0, 0.3, 0.25, 0.45]])
    s = torch.tensor([[2, 3, 2, 3]])
    first = next_distribution(live, free, s, 0, f)
    assert np.allclose(first.numpy(), [[0, 0.12, 0.7, 0.18]])
    next_ = next_distribution(
        live / 0.7, torch.tensor([0.1 / 0.7]), torch.tensor([[3, 2, 3, 0]]), 1, f
    )
    assert np.allclose(
        next_.numpy(), [[0, 0.33 / 0.7, 0.025 / 0.7, 0.345 / 0.7]], atol=1e-6
    )


def test_visual_logits_support_short_pages():
    torch.set_num_threads(1)
    net = EvidenceWriter(width=16)
    q = torch.tensor([[2], [3]])
    s = torch.tensor([[2], [4]])
    bank = torch.randn(6, 16)
    scores, *_ = net.read(bank, q, s)
    assert torch.allclose(scores.exp().sum((1, 2)), torch.ones(2))

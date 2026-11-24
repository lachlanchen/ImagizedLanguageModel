"""Matched set/within-span inference, including invalid candidates and singleton lists."""

import torch

from ilm.visual_lm.optical_competition import CandidateCompetition


def test_set_order_and_rival_intervention():
    torch.manual_seed(915)
    own = CandidateCompetition("independent")
    torch.nn.init.normal_(own.out[-1].weight)
    full = CandidateCompetition("competing")
    full.load_state_dict(own.state_dict())
    images = torch.randn(2, 5, 3, 128)
    scores = torch.tensor(
        [
            [1.0, 0.0, -1.0, 2.0, -torch.inf],
            [1.0, -torch.inf, -torch.inf, -torch.inf, -torch.inf],
        ]
    )
    lengths = torch.ones_like(scores, dtype=torch.long)
    valid = scores.isfinite()
    direct = own(images, scores, lengths)
    comparison = full(images, scores, lengths)
    torch.testing.assert_close(full(images, scores, lengths, within=True), direct)
    torch.testing.assert_close(
        full(images.flip(1), scores.flip(1), lengths.flip(1)).flip(1), comparison
    )
    torch.testing.assert_close(comparison[1], direct[1])
    changed = images.clone()
    changed[0, 1] = 7.0
    torch.testing.assert_close(own(changed, scores, lengths)[0, 0], direct[0, 0])
    assert abs(full(changed, scores, lengths)[0, 0] - comparison[0, 0]) > 0.005
    comparison[valid].sum().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in full.parameters()
    )

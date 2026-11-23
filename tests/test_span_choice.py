import torch

from ilm.visual_lm.span_choice import pooled_choice


def test_pool_votes_by_physical_answer():
    # Identical strips proposed by different readers must share one probability.
    eq = torch.tensor(
        [[[a == b for b in ["A", "B", "B", "C"]] for a in ["A", "B", "B", "C"]]]
    )
    scores = torch.tensor([[4.0, 3.0, 3.0, 1.0]]).log()
    assert scores.argmax(1).item() == 0
    assert pooled_choice(scores, eq).item() == 1


def test_ties_preserve_original_reader_order():
    same = torch.eye(4, dtype=torch.bool).unsqueeze(0)
    assert pooled_choice(torch.zeros(1, 4), same).item() == 0

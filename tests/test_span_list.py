import torch

from ilm.visual_lm.span_list import WIDTH, SpanRanker, image_choice


def test_repeated_strip_can_collect_probability_from_two_positions():
    # An invalid proposal maps to group zero for padding and must carry no mass.
    scores = torch.tensor([[0.0, 0.7, 0.25, -torch.inf]])
    ids = torch.tensor([[0, 1, 0, 0]])
    assert scores.argmax(1).item() == 1
    assert image_choice(scores, ids).item() == 0


def test_short_passage_has_one_valid_choice_and_finite_training_loss():
    x = torch.zeros(1, 32, WIDTH)
    valid = torch.zeros(1, 32, dtype=torch.bool)
    valid[:, 0] = True
    model = SpanRanker()
    logits = model(x, valid)
    assert logits.isfinite().equal(valid)
    loss = logits.logsumexp(1) - logits[:, 0]
    loss.backward()
    assert loss.item() == 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


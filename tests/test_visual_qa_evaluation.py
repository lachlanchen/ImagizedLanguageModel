import torch

from ilm.visual_lm.visual_qa_evaluation import answer_scores, paired_article_bootstrap


def test_answer_order_length_and_stopping_are_required():
    x = torch.zeros(3, 1, 32, 32)
    x[0, 0, 0, 0] = 1
    x[1, 0, 1, 1] = 1
    x[2, 0, 2, 2] = 1
    assert answer_scores(x, [x], stopped=True)["exact_answer"] == 1
    assert answer_scores(x, [x], stopped=False)["exact_answer"] == 0
    s = answer_scores(x.flip(0), [x], stopped=True)
    assert s["exact_answer"] == 0 and s["raster_f1"] == 1
    assert answer_scores(x[:0], [x], stopped=True)["raster_f1"] == 0
    assert answer_scores(x[:1], [x], stopped=True)["raster_f1"] == 0.5


def test_grouped_uncertainty_is_deterministic():
    a = paired_article_bootstrap([1, 1, 0, 0], ["a", "a", "b", "b"])
    assert a == paired_article_bootstrap([1, 1, 0, 0], ["a", "a", "b", "b"])
    assert a["mean"] == 0.5 and a["articles"] == 2

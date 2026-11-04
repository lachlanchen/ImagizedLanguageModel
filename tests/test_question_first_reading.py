from collections import Counter

import pytest
import torch

from ilm.visual_lm.question_first_reading import ordered_question, blank_question_images


def test_order_changes_only_host_layout_and_blank_range():
    record = dict(context="山水花木", question="水在哪？", prompt="资料：山水花木问题：水在哪？回答：",
                  references=[dict(text="水")], id="example")
    for order, start in (("context", 10), ("question", 3)):
        changed = ordered_question(record, order)
        assert Counter(changed["prompt"]) == Counter(record["prompt"])
        assert changed["references"] == record["references"]
        pixels = torch.ones(len(changed["prompt"]), 1, 32, 32)
        blank = blank_question_images(pixels, record, order)
        assert not blank[start:start + 4].any()
        assert blank[:start].all() and blank[start + 4:].all()
        assert pixels.all()


def test_unexpected_layout_rejected():
    with pytest.raises(ValueError):
        ordered_question(dict(context="a", question="b", prompt="wrong"), "question")

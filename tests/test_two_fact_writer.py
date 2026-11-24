from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ilm.visual_lm.field_alignment import plan_scores, target_loss
from ilm.visual_lm.ordered_copy import ordered_copy_beam
from scripts.prepare_chinese_two_fact import MODEL
from scripts.write_chinese_two_fact import TwoFactReply, load_page


def test_alignment_marginal_and_gradient_match_enumeration():
    torch.manual_seed(17)
    z = torch.randn((1, 6, 3), requires_grad=True)
    visible = torch.tensor([[False, True, True, False, True, True]])
    active = [1, 2, 4, 5]
    scores = plan_scores(z, visible, torch.tensor([[True, True, False]]), 0.7)
    samples = torch.stack([z[0, u, 0] + z[0, v, 1] for u, v in combinations(active, 2)])
    exact = torch.logsumexp(samples, 0) - np.log(len(samples)) + 0.7
    torch.testing.assert_close(scores[0, 4], exact)
    torch.testing.assert_close(
        torch.autograd.grad(exact, z, retain_graph=True)[0],
        torch.autograd.grad(scores[0, 4], z)[0],
    )
    assert not torch.isfinite(scores[0, [2, 3, 5, 7, 8, 9, 10, 11]]).any()
    for is_ordered in (False, True):
        loss = target_loss(
            scores, torch.tensor([[0, 1]]), torch.tensor([is_ordered]), 3
        )
        p = scores.softmax(1)[0]
        torch.testing.assert_close(loss, -(p[4] if is_ordered else p[4] + p[6]).log())


def test_ordered_beam_matches_exact_best_completed_path():
    class Fixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.position = torch.zeros(2)
            self.feedback = nn.Linear(4, 2)
            self.probs = torch.tensor(
                [
                    [0.04, 0.69, 0.26, 0.01],
                    [0.14, 0.1, 0.72, 0.04],
                    [0.10, 0.25, 0.08, 0.57],
                    [0.76, 0.03, 0.05, 0.16],
                ]
            )

        def context(self, previous, memory, visible):
            return torch.zeros((len(previous), previous.shape[1], 2))

        def letters(self, outer, block):
            return self.probs.log().expand(len(outer), 1, 4, 4)

    net = Fixed()
    palette = torch.eye(4)
    visible = torch.ones((1, 1), dtype=torch.bool)
    mem = torch.zeros((1, 1, 3))
    seq = ordered_copy_beam(net, mem, visible, palette, [2, 3], width=16, limit=4)[0]
    options = []
    for length in range(4):
        for path in product((1, 2, 3), repeat=length):
            if [i for i in path if i in (2, 3)] != [2, 3]:
                continue
            logp = sum(net.probs[t, i].log().item() for t, i in enumerate((*path, 0)))
            options.append((logp / (length + 1) ** 0.7, list(path)))
    assert seq == max(options)[1]


def test_released_png_reproduces_the_two_answer_spans():
    torch.set_num_threads(2)
    reader = TwoFactReply()
    folder = MODEL / "cursor-examples"
    plan, ink = reader.reply(load_page(folder / "first-card.png"))
    assert plan == ["人均消费", "地址"]
    assert ink.tobytes() == (folder / "first-answer.bin").read_bytes()
    assert "67元" in reader.transcript(ink) and "黄草洼村" in reader.transcript(ink)


def test_ordered_beam_rejects_duplicate_markers():
    with pytest.raises(ValueError, match="distinct"):
        ordered_copy_beam(SimpleNamespace(), None, None, torch.eye(3), [1, 1])


def test_unseen_source_marks_survive_the_entire_image_path(monkeypatch):
    def reject_old_writer(*args, **kwargs):
        pytest.fail("The current-field writer loaded the earlier language model")

    monkeypatch.setattr("scripts.write_chinese_two_fact.load", reject_old_writer)
    torch.set_num_threads(2)
    reader = TwoFactReply()
    page = load_page(MODEL / "cursor-examples/exchange-card.png")
    route, original = reader.reply(page)
    assert route == ["价格", "地址"]
    expected = original.tobytes()
    for seed, field in enumerate(route, 76):
        prefix = reader.prefixes[field]
        row = next(
            row for row in page[2:] if np.array_equal(row[: len(prefix)], prefix)
        )
        tail = row[len(prefix) :]
        size = np.flatnonzero(tail.any(1))[-1] + 1
        before = tail[:size].tobytes()
        assert expected.count(before) == 1
        marks = np.random.default_rng(seed).integers(
            1, 255, (size, 128), dtype=np.uint8
        )
        assert all(bytes(mark) not in reader.characters for mark in marks)
        tail[:size] = marks
        expected = expected.replace(before, marks.tobytes())
    edited_route, edited = reader.reply(page)
    assert edited_route == route
    assert edited.tobytes() == expected
    assert "\ufffc" in reader.transcript(edited)

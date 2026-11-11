"""Causality and boundary checks for the reference-history instrument."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ilm.visual_lm.reference_image_history import trace_reference_history


class Context:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if 'past_key_values' in kw:
            past = kw['past_key_values']
            assert kw['position_ids'].flatten().tolist() == [past.get_seq_length(),
                10 + len(self.calls) - 2, 10 + len(self.calls) - 2, 10 + len(self.calls) - 2]
            h = kw['inputs_embeds'] + past.h
            n = past.get_seq_length() + 1
        else:
            h, n = torch.zeros(1, 1, 2), 7
        return SimpleNamespace(last_hidden_state=h,
            past_key_values=SimpleNamespace(h=h, get_seq_length=lambda: n))


def actor():
    context = Context()
    bridge = SimpleNamespace(
        feedback_embedding=lambda x: x.mean((1, 2, 3))[:, None].repeat(1, 2),
        predict=lambda h: (h[:, :1, None, None].expand(-1, 1, 32, 32), h,
                           torch.full((len(h),), 9.)))
    return SimpleNamespace(device='cpu', bridge=bridge, interface=SimpleNamespace(
        assemble=lambda x: ({'inputs_embeds': torch.ones(1, 7, 2)}, {'rotary_offset': 10}),
        actor=SimpleNamespace(context=context)))


def test_current_target_is_not_in_its_prediction_and_terminal_is_included():
    a = actor()
    gold = torch.stack([torch.ones(1, 32, 32), -torch.ones(1, 32, 32)])
    with patch('ilm.visual_lm.reference_image_history.native_budget_image', lambda x: x):
        v = trace_reference_history(a, torch.ones(1, 1, 32, 32), gold)
    assert torch.equal(v['states'], torch.tensor([[0., 0.], [1., 1.], [0., 0.]]))
    assert len(v['reference_feedback']) == 2 and len(v['predicted_ink']) == 3
    assert len(a.interface.actor.context.calls) == 3
    assert all(p > .5 for p in v['stop_probabilities'])  # Stop is recorded, not obeyed.
    assert not v['autonomous_language_generation'] and v['stopping_bypassed']


def test_changing_future_pixels_does_not_change_previous_states():
    gold = torch.ones(3, 1, 32, 32)
    with patch('ilm.visual_lm.reference_image_history.native_budget_image', lambda x: x):
        first = trace_reference_history(actor(), gold[:1], gold)
        gold[1:] = -1
        second = trace_reference_history(actor(), gold[:1], gold)
    assert torch.equal(first['states'][:2], second['states'][:2])
    assert not torch.equal(first['states'][2:], second['states'][2:])


@pytest.mark.parametrize('gold', [torch.empty(0, 1, 32, 32), torch.zeros(1, 1, 32, 32)])
def test_rejects_missing_or_nonbinary_reference(gold):
    with pytest.raises(ValueError):
        trace_reference_history(actor(), torch.ones(1, 1, 32, 32), gold)

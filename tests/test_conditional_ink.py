"""Check that free beam branches use the same physical conditional as the training loss."""
import torch
from torch.nn import functional as F

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.idiom_ink import nibbles
from ilm.visual_lm.visual_edit_writer import InkPatterns


def test_beam_score_matches_teacher_on_the_selected_pixels():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(74)
        correction = ConditionalInk()
        correction.encoder[-1].weight.normal_(std=.1)
        writer = InkPatterns(32)
        context = torch.tensor([[0, 1 << 48, 19, 133], [1, 2, 3, 4], [5, 6, 7, 8]])
        onset = torch.tensor([True, False, True])
        state = torch.randn(3,32)
        local = torch.randn(3,12,16)
        transitions = correction.transitions(context,onset)
        bits, score = writer.choose(state,beam=4,pattern_bias=local,transitions=transitions,return_score=True)
        code = nibbles(bits)
        logits = writer.teacher(state,bits) + local
        logits[:,1:] += correction.teacher(context,onset,code)
        reference = -F.cross_entropy(logits.transpose(1,2),code,reduction='none').sum(1)
        torch.testing.assert_close(score,reference)
        changed = (code + 7) % 16
        # Changing the final group cannot alter the logits of any trained transition.
        changed[:,:-1] = code[:,:-1]
        torch.testing.assert_close(correction.teacher(context,onset,changed),correction.teacher(context,onset,code))


def test_initial_correction_leaves_the_frozen_beam_unchanged():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(75)
        correction = ConditionalInk()
        state = torch.randn(4,32)
        context = torch.randint(0,1<<48,(4,4))
        onset = torch.ones(4,dtype=torch.bool)
        writer = InkPatterns(32)
        base = writer.choose(state,return_score=True)
        updated = writer.choose(state,transitions=correction.transitions(context,onset),return_score=True)
        for old,new in zip(base,updated,strict=True):
            torch.testing.assert_close(old,new,atol=0,rtol=0)

import torch

from ilm.visual_lm.glyph_measure import expand_states, positions


def test_repeated_positions_conserve_attention_for_signed_values():
    mapping = [positions([0, 2, 6], 6), positions([2], 2)]
    assert mapping == [[0, 1, 1, 2, 2, 2, 2, 3], [0, 0, 0, 1]]
    original = torch.tensor([[[2., -1.], [0., 3.], [1., 0.], [-4., 7.]],
                             [[.5, 1.], [3., -3.], [0., 0.], [0., 0.]]], dtype=torch.float64)
    expanded, mask, weight = expand_states(original, mapping)
    query = torch.tensor([-.8, .5], dtype=torch.float64)
    for i, n in enumerate((4, 2)):
        expected = (original[i,:n] @ query).softmax(0) @ original[i,:n]
        measure = ((expanded[i] @ query) + weight[i]).masked_fill(~mask[i].bool(), -torch.inf)
        actual = measure.softmax(0) @ expanded[i]
        torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)


def test_zero_width_piece_keeps_distinct_slot_and_eos():
    mapping = positions([0, 0, 1], 2)
    assert mapping == [0, 1, 2, 3]

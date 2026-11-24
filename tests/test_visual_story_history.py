"""Ensure stepwise generation sees the same causal memory used by training."""

import torch
from torch import nn

from ilm.visual_lm.visual_story_writer import AttentiveHistory


def test_causal_history_matches_cached_chunks_with_learned_residual():
    torch.manual_seed(69)
    history = AttentiveHistory(nn.GRU(12, 16, batch_first=True), heads=4).double().eval()
    nn.init.normal_(history.project.weight, std=0.1)
    x = torch.randn(2, 9, 12, dtype=torch.float64)
    h = torch.randn(1, 2, 16, dtype=torch.float64)
    full, _ = history(x, h)
    for chunks in ([1] * 9, [2, 3, 4]):
        state, seq = h, []
        for chunk in x.split(chunks, dim=1):
            out, state = history(chunk, state)
            seq.append(out)
        torch.testing.assert_close(torch.cat(seq, dim=1), full)
    changed = x.clone()
    changed[:, 4:] += 10
    changed_output, _ = history(changed, h)
    torch.testing.assert_close(full[:, :4], changed_output[:, :4])


def test_catalogue_blank_and_one_black_stroke():
    from ilm.visual_lm.catalogue_ink import print_cell
    assert print_cell(bytes([255]) * 128) == bytes(128)
    assert print_cell(bytes([127]) + bytes([255]) * 127) == bytes([128]) + bytes(127)

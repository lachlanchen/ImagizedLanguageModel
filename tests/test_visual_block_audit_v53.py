import torch

from scripts.eval_causal_visual_block_diffusion_v53 import grouped_bootstrap


def test_grouped_bootstrap_resamples_records_and_preserves_paired_zero():
    receipt = grouped_bootstrap([0, 0, 0, 0], ["a", "a", "b", "b"], replicates=100)
    assert receipt["ci95"] == [0, 0]
    assert receipt["source_records"] == 2
    assert receipt["windows"] == 4
    positive = grouped_bootstrap([1, 1, 1], ["a", "a", "b"], replicates=100)
    assert positive["ci95"] == [1, 1]
    assert positive["mean"] == 1


def test_block_strip_roundtrip_preserves_future_position():
    from ilm.visual_lm.causal_visual_block_diffusion_v53 import ConditionalStripDenoiser

    cells = torch.arange(4.0).reshape(1, 4, 1, 1, 1).expand(1, 4, 1, 32, 32)
    strip = ConditionalStripDenoiser.cells_to_strip(cells)
    assert torch.equal(strip[0, 0, 0], torch.arange(4.0).repeat_interleave(32))
    assert torch.equal(ConditionalStripDenoiser.strip_to_cells(strip), cells)

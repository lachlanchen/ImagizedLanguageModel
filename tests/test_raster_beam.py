from itertools import product
import math
from types import SimpleNamespace

import pytest
import torch

from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.exact_raster_density_v54 import (
    ExactRasterDensity,
    ExactRasterDensityConfig,
)
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.raster_beam import raster_beam_search


def model():
    torch.manual_seed(56)
    reader = CausalRasterReader(CanonicalGlyphLanguageConfig(layers=1))
    value = ExactRasterDensity(
        reader, ExactRasterDensityConfig(width=128, layers=1, heads=4, pixel_hidden=64)
    ).eval()
    with torch.no_grad():
        value.pixel_density.output.weight.normal_(0, 0.1)
        value.pixel_density.condition_out.weight.normal_(0, 0.1)
    return value


def test_width_one_matches_original_greedy_and_exact_density_score():
    value = model()
    hidden = torch.randn(2, 384)
    fields = value.reader.field.encode_unit(torch.randint(0, 2, (2, 1, 32, 32)).float())
    greedy = value.sample_state(
        hidden, fields, generator=torch.Generator().manual_seed(56), greedy=True
    )
    result = raster_beam_search(value, hidden, fields, beam_width=1)
    torch.testing.assert_close(result.images, greedy, atol=0, rtol=0)
    torch.testing.assert_close(
        result.log_probability,
        -value.image_nll(result.images, hidden, fields),
        atol=1e-3,
        rtol=1e-5,
    )


@pytest.mark.parametrize("across_rows", [False, True])
def test_exhaustive_binary_mode_and_parent_history(across_rows):
    class FourBitDensity:
        training = False
        config = SimpleNamespace(plan_logit_scale=0)

        def row_states(self, rows, hidden, fields):
            index = torch.arange(rows.shape[1]).float()[None].expand(len(rows), -1)
            first = rows[:, 0, 0, None].expand(-1, rows.shape[1])
            return torch.stack((index, first), -1), torch.zeros_like(rows)

        def pixel_density(self, row, condition):
            logits = torch.full_like(row, -80)
            if across_rows:
                r = condition[:, 0]
                logits[r == 0, 0] = math.log(0.6 / 0.4)
                selected = r == 1
                logits[selected, 0] = torch.where(
                    condition[selected, 1] > 0, 0.0, math.log(99)
                )
                logits[(r == 2) | (r == 3), 0] = math.log(9)
            else:
                selected = condition[:, 0] == 0
                logits[selected, 0] = math.log(0.6 / 0.4)
                logits[selected, 1] = torch.where(
                    row[selected, 0] > 0, 0.0, math.log(99)
                )
                logits[selected, 2:4] = math.log(9)
            return logits

    value = FourBitDensity()
    h = f = torch.zeros(1, 1)
    result = raster_beam_search(value, h, f, beam_width=16)
    greedy = raster_beam_search(value, h, f, beam_width=1)
    outcomes = {
        bits: (0.6 if bits[0] else 0.4)
        * ((0.5 if bits[0] else 0.99) if bits[1] else (0.5 if bits[0] else 0.01))
        * (0.9 if bits[2] else 0.1)
        * (0.9 if bits[3] else 0.1)
        for bits in product((0, 1), repeat=4)
    }
    best = max(outcomes, key=outcomes.get)
    selected = result.images[0, 0, :4, 0] if across_rows else result.images[0, 0, 0, :4]
    assert selected.tolist() == list(best) == [0, 1, 1, 1]
    assert float(greedy.images[0, 0, 0, 0]) == 1
    assert float(result.log_probability[0]) > float(greedy.log_probability[0])
    assert abs(float(result.log_probability[0]) - math.log(outcomes[best])) < 1e-5
    assert len(torch.unique(result.beams.flatten(0, 1).flatten(1), dim=0)) == 16


def test_nontrivial_beam_scores_match_full_image_likelihood():
    value = model()
    hidden = torch.randn(1, 384)
    fields = value.reader.field.encode_unit(torch.randint(0, 2, (1, 1, 32, 32)).float())
    result = raster_beam_search(value, hidden, fields, beam_width=4)
    expected = -value.image_nll(
        result.beams[0], hidden.expand(4, -1), fields.expand(4, -1)
    )
    torch.testing.assert_close(
        result.beam_log_probabilities[0], expected, atol=1e-3, rtol=1e-5
    )
    assert bool(
        (
            result.beam_log_probabilities[0, 1:]
            <= result.beam_log_probabilities[0, :-1]
        ).all()
    )

import torch

from ilm.visual_lm.causal_glyph_translator import (
    COPY,
    END,
    EXEMPLAR,
    CausalGlyphTranslator,
)


def test_stopping_and_capped_writing_preserve_the_actual_visible_ink():
    torch.manual_seed(304)
    exemplar = torch.rand(1, 1024) > 0.6
    model = CausalGlyphTranslator(torch.arange(48), exemplar, width=32).eval()
    inputs = torch.rand(2, 3, 1024) > 0.7
    lengths = torch.tensor([3, 1])
    with torch.no_grad():
        model.route.weight.zero_()
        for action in [END, COPY, EXEMPLAR]:
            model.route.bias.zero_()
            model.route.bias[action] = 2
            pixels, count, stopped, route = model.generate(inputs, lengths, limit=3)
            if action == END:
                assert stopped.all() and (count == 0).all()
                continue
            assert not stopped.any() and (count == 3).all() and (route == action).all()
            if action == COPY:
                torch.testing.assert_close(pixels[1], inputs[1, 0].expand(3, -1))
                assert all(any(torch.equal(g, v) for v in inputs[0]) for g in pixels[0])
            else:
                torch.testing.assert_close(pixels, exemplar.expand_as(pixels))

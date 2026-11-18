import torch
from ilm.visual_lm.causal_glyph_translator import COPY, DRAW, END, CausalGlyphTranslator
from ilm.visual_lm.idiom_ink import visual_input
from scripts.train_visible_image_likelihood import emission_loss


def test_duplicate_routes_increase_observed_ink_mass_and_allow_draw_only_rows():
    torch.manual_seed(251)
    bank = torch.rand(3, 1024) > 0.52
    model = CausalGlyphTranslator(torch.arange(48), bank[:1], width=32)
    feat = visual_input(bank, model.landmarks)
    source = torch.tensor([[0, 0], [1, 1]])
    target = torch.tensor([[0], [2]])
    labels = torch.tensor([[COPY, END], [DRAW, END]])
    args = (
        model,
        feat,
        source,
        target,
        torch.tensor([2, 2]),
        torch.tensor([1, 1]),
        torch.zeros(2, 1, dtype=torch.long),
        labels,
        torch.zeros(2, 1, dtype=torch.long),
        torch.tensor([0, -1, -1]),
    )
    single, combined = (emission_loss(*args, marginal=m) for m in [False, True])
    assert torch.isfinite(single) and combined < single
    combined.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )

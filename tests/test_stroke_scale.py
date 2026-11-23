import torch

from scripts.train_stroke_scale import PresentedViews, distort


def test_sparse_marks_survive_and_transform_rng_is_independent():
    ink = torch.zeros((96, 32, 32), dtype=torch.uint8)
    ink[:48, 5, 6] = 1
    before = torch.random.get_rng_state()
    result = distort(ink, torch.Generator().manual_seed(3))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(result.flatten(1).any(1), ink.flatten(1).any(1))
    assert torch.equal(result, distort(ink, torch.Generator().manual_seed(3)))


def test_collision_includes_exposures_before_conflicting_label():
    audit = PresentedViews()
    ink = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ink[2, 0, 0] = 1
    audit(ink, torch.tensor([3, 3, 4]), torch.zeros(3, dtype=torch.long), 0, 0)
    audit(ink[:1], torch.tensor([6]), torch.ones(1, dtype=torch.long), -1, 1)
    result = audit.summary()
    assert result["unique"] == 2 and result["exposures"] == 4
    assert result["conflicting_bitmaps"] == 1
    assert result["occurrences_on_conflicting_bitmaps"] == 3


def test_print_route_ignores_blank_cells_and_checks_each_strip():
    from types import SimpleNamespace
    from ilm.visual_lm.causal_glyph_writer import CausalGlyphWriter

    writer = SimpleNamespace(augmented_typeface_map=True)
    known = {bytes([1]) * 128: 1}

    def cells(*values):
        return torch.tensor(values, dtype=torch.uint8)[:, None].expand(-1, 128)

    choose = CausalGlyphWriter.uses_augmented_print
    assert not choose(writer, cells(1, 1, 1, 1, 2), cells(0), known)
    assert choose(writer, cells(1, 1, 1, 1), cells(0, 2), known)
    assert not choose(SimpleNamespace(), cells(2), cells(2), known)

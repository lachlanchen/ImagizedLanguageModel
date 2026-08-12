from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.visual_cell_lattice import (
    VisualCellLatticeConfig,
    fold_cells_to_page,
    fold_latent_sequence,
    serpentine_coordinates,
    unfold_latent_lattice,
    unfold_page_to_cells,
    visual_cell_lattice_boundary_receipt,
)


def test_256_square_lattice_holds_65536_visual_characters() -> None:
    config = VisualCellLatticeConfig()
    assert config.capacity == 65_536
    assert config.native_page_shape == (8_192, 8_192)
    assert config.overview_page_shape == (2_048, 2_048)
    receipt = visual_cell_lattice_boundary_receipt(config)
    assert receipt["latent_lattice_is_continuous"] is True
    assert receipt["overview_is_lossy_and_never_the_only_record"] is True
    assert all(
        receipt[key] is False
        for key in (
            "uses_token_ids",
            "uses_unicode_ids",
            "uses_character_ids",
            "uses_visual_codebook",
        )
    )


def test_serpentine_sequence_neighbors_remain_spatial_neighbors() -> None:
    coordinates = serpentine_coordinates(20, rows=4, columns=5)
    assert coordinates.tolist()[:6] == [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 4],
    ]
    steps = coordinates[1:] - coordinates[:-1]
    assert torch.equal(steps.abs().sum(dim=1), torch.ones(19, dtype=torch.long))


def test_continuous_latent_fold_is_exact_and_differentiable() -> None:
    sequence = torch.randn(2, 9, 7, requires_grad=True)
    lattice, valid = fold_latent_sequence(sequence, rows=3, columns=4)
    assert lattice.shape == (2, 7, 3, 4)
    assert valid.shape == (2, 1, 3, 4)
    assert valid.sum().item() == 18
    recovered = unfold_latent_lattice(lattice, length=9)
    torch.testing.assert_close(recovered, sequence, rtol=0.0, atol=0.0)
    recovered.square().mean().backward()
    assert sequence.grad is not None
    assert float(sequence.grad.abs().sum()) > 0.0


def test_native_2d_page_and_3d_cell_stream_are_exact_views() -> None:
    cells = torch.rand(2, 7, 1, 32, 32)
    page, valid = fold_cells_to_page(cells, rows=2, columns=4)
    assert page.shape == (2, 1, 64, 128)
    assert valid.sum().item() == 14
    recovered = unfold_page_to_cells(
        page,
        rows=2,
        columns=4,
        length=7,
    )
    torch.testing.assert_close(recovered, cells, rtol=0.0, atol=0.0)


def test_overview_page_is_small_but_not_used_as_the_only_glyph_record() -> None:
    cells = torch.rand(1, 6, 1, 32, 32)
    page, _ = fold_cells_to_page(
        cells,
        rows=2,
        columns=3,
        output_cell_size=8,
    )
    assert page.shape == (1, 1, 16, 24)
    overview_cells = unfold_page_to_cells(
        page,
        rows=2,
        columns=3,
        length=6,
    )
    assert overview_cells.shape == (1, 6, 1, 8, 8)
    assert overview_cells.numel() < cells.numel()


def test_lattice_rejects_overflow_and_non_image_values() -> None:
    with pytest.raises(ValueError, match="capacity"):
        fold_latent_sequence(torch.rand(1, 7, 4), rows=2, columns=3)
    with pytest.raises(TypeError, match="floating"):
        fold_cells_to_page(
            torch.ones(1, 2, 1, 32, 32, dtype=torch.int64),
            rows=1,
            columns=2,
        )

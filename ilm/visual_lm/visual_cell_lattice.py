from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisualCellLatticeConfig:
    """Reversible geometry for folding visual time onto a two-dimensional field."""

    rows: int = 256
    columns: int = 256
    cell_size: int = 32
    overview_cell_size: int = 8

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("lattice rows and columns must be positive")
        if self.cell_size < 1 or self.overview_cell_size < 1:
            raise ValueError("cell sizes must be positive")
        if self.overview_cell_size > self.cell_size:
            raise ValueError("overview cells cannot exceed native cell size")

    @property
    def capacity(self) -> int:
        return self.rows * self.columns

    @property
    def native_page_shape(self) -> tuple[int, int]:
        return self.rows * self.cell_size, self.columns * self.cell_size

    @property
    def overview_page_shape(self) -> tuple[int, int]:
        return (
            self.rows * self.overview_cell_size,
            self.columns * self.overview_cell_size,
        )


def serpentine_coordinates(
    length: int,
    *,
    rows: int,
    columns: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map reading time to adjacent `(row, column)` positions."""

    capacity = rows * columns
    if length < 0 or length > capacity:
        raise ValueError(f"length must be in [0, {capacity}]")
    time = torch.arange(length, device=device, dtype=torch.long)
    row = torch.div(time, columns, rounding_mode="floor")
    offset = time.remainder(columns)
    column = torch.where(row.remainder(2) == 0, offset, columns - 1 - offset)
    return torch.stack((row, column), dim=-1)


def serpentine_flat_indices(
    length: int,
    *,
    rows: int,
    columns: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    coordinates = serpentine_coordinates(
        length,
        rows=rows,
        columns=columns,
        device=device,
    )
    return coordinates[:, 0] * columns + coordinates[:, 1]


def fold_latent_sequence(
    sequence: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold `[B,T,D]` visual states into `[B,D,R,C]` without quantization."""

    if sequence.ndim != 3 or not torch.is_floating_point(sequence):
        raise TypeError("latent sequence must be a floating tensor [B,T,D]")
    batch, length, channels = sequence.shape
    capacity = rows * columns
    if length > capacity:
        raise ValueError(f"sequence length exceeds lattice capacity {capacity}")
    order = serpentine_flat_indices(
        length,
        rows=rows,
        columns=columns,
        device=sequence.device,
    )
    flat = sequence.new_zeros((batch, capacity, channels))
    flat = flat.index_copy(1, order, sequence)
    lattice = flat.reshape(batch, rows, columns, channels).permute(0, 3, 1, 2)
    valid = torch.zeros((capacity,), device=sequence.device, dtype=torch.bool)
    valid[order] = True
    return lattice.contiguous(), valid.reshape(1, 1, rows, columns).expand(
        batch, -1, -1, -1
    )


def unfold_latent_lattice(lattice: torch.Tensor, *, length: int) -> torch.Tensor:
    """Recover `[B,T,D]` visual states in their original reading order."""

    if lattice.ndim != 4 or not torch.is_floating_point(lattice):
        raise TypeError("latent lattice must be a floating tensor [B,D,R,C]")
    batch, channels, rows, columns = lattice.shape
    order = serpentine_flat_indices(
        length,
        rows=rows,
        columns=columns,
        device=lattice.device,
    )
    flat = lattice.permute(0, 2, 3, 1).reshape(batch, rows * columns, channels)
    return flat.index_select(1, order)


def _validate_cells(cells: torch.Tensor) -> tuple[int, int, int]:
    if cells.ndim != 5 or not torch.is_floating_point(cells):
        raise TypeError("visual cells must be a floating tensor [B,T,C,H,W]")
    if cells.shape[-1] != cells.shape[-2]:
        raise ValueError("visual cells must be square")
    return cells.shape[0], cells.shape[1], cells.shape[-1]


def fold_cells_to_page(
    cells: torch.Tensor,
    *,
    rows: int,
    columns: int,
    output_cell_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a flat page view while preserving the serpentine reading map."""

    batch, length, native_size = _validate_cells(cells)
    channels = cells.shape[2]
    capacity = rows * columns
    if length > capacity:
        raise ValueError(f"cell stream exceeds page capacity {capacity}")
    size = native_size if output_cell_size is None else output_cell_size
    if size < 1:
        raise ValueError("output_cell_size must be positive")
    tiles = cells
    if size != native_size:
        tiles = F.interpolate(
            cells.flatten(0, 1),
            size=(size, size),
            mode="area" if size < native_size else "bilinear",
            align_corners=False if size > native_size else None,
        ).reshape(batch, length, channels, size, size)
    order = serpentine_flat_indices(
        length,
        rows=rows,
        columns=columns,
        device=cells.device,
    )
    flat = cells.new_zeros((batch, capacity, channels, size, size))
    flat = flat.index_copy(1, order, tiles)
    page = flat.reshape(batch, rows, columns, channels, size, size)
    page = page.permute(0, 3, 1, 4, 2, 5).reshape(
        batch,
        channels,
        rows * size,
        columns * size,
    )
    valid_cells = torch.zeros((capacity,), device=cells.device, dtype=torch.bool)
    valid_cells[order] = True
    valid = valid_cells.reshape(1, 1, rows, columns).expand(batch, -1, -1, -1)
    return page.contiguous(), valid


def unfold_page_to_cells(
    page: torch.Tensor,
    *,
    rows: int,
    columns: int,
    length: int,
) -> torch.Tensor:
    """Split a grid-aligned page back into ordered continuous image cells."""

    if page.ndim != 4 or not torch.is_floating_point(page):
        raise TypeError("page must be a floating tensor [B,C,H,W]")
    batch, channels, height, width = page.shape
    if height % rows or width % columns:
        raise ValueError("page dimensions must be divisible by the lattice")
    cell_height = height // rows
    cell_width = width // columns
    if cell_height != cell_width:
        raise ValueError("page cells must be square")
    order = serpentine_flat_indices(
        length,
        rows=rows,
        columns=columns,
        device=page.device,
    )
    tiles = page.reshape(
        batch,
        channels,
        rows,
        cell_height,
        columns,
        cell_width,
    ).permute(0, 2, 4, 1, 3, 5)
    flat = tiles.reshape(
        batch,
        rows * columns,
        channels,
        cell_height,
        cell_width,
    )
    return flat.index_select(1, order)


def visual_cell_lattice_config_payload(
    config: VisualCellLatticeConfig,
) -> dict[str, Any]:
    return asdict(config)


def visual_cell_lattice_boundary_receipt(
    config: VisualCellLatticeConfig,
) -> dict[str, Any]:
    return {
        "architecture": "serpentine-visual-cell-lattice",
        "capacity": config.capacity,
        "native_cell_shape": [1, config.cell_size, config.cell_size],
        "native_page_shape": list(config.native_page_shape),
        "overview_page_shape": list(config.overview_page_shape),
        "fold_is_bijective_at_native_resolution": True,
        "overview_is_lossy_and_never_the_only_record": True,
        "sequence_neighbors_are_spatial_neighbors": True,
        "latent_lattice_is_continuous": True,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_visual_codebook": False,
    }

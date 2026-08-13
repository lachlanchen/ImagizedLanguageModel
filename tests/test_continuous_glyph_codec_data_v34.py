from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import torch

from ilm.visual_lm.continuous_glyph_codec_data import (
    HistoricGlyphRasterDataset,
    HistoricGlyphRecord,
    active_rendered_patches,
    historic_character_partition,
    historic_glyph_collate,
    load_historic_glyph_records,
    load_or_build_historic_raster_cache,
    render_historic_svg,
)


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
<path fill="black" d="M1 1h8v8H1z"/>
</svg>
"""


def write_historic_fixture(tmp_path: Path) -> tuple[Path, Path, list[HistoricGlyphRecord]]:
    root = tmp_path / "root"
    glyph = root / "data" / "historic" / "glyphs" / "中" / "oracle" / "J1.svg"
    glyph.parent.mkdir(parents=True)
    glyph.write_text(SVG, encoding="utf-8")
    database = tmp_path / "etymology.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE chars (id INTEGER PRIMARY KEY, char TEXT NOT NULL);
        CREATE TABLE glyphs (
          id INTEGER PRIMARY KEY,
          char_id INTEGER NOT NULL,
          stage TEXT NOT NULL,
          label TEXT,
          local_path TEXT NOT NULL
        );
        INSERT INTO chars VALUES (1, '中');
        INSERT INTO glyphs VALUES (
          1, 1, 'oracle', 'J1', 'data/historic/glyphs/中/oracle/J1.svg'
        );
        """
    )
    connection.commit()
    connection.close()
    records = load_historic_glyph_records(database)
    return root, database, records


def test_historic_partition_is_character_level() -> None:
    assert historic_character_partition("中") == historic_character_partition("中")
    assert historic_character_partition("中") in {"train", "development", "sealed"}
    with pytest.raises(ValueError, match="character"):
        historic_character_partition("")


def test_svg_raster_and_cache_are_binary(tmp_path: Path) -> None:
    root, _, records = write_historic_fixture(tmp_path)
    source = root / records[0].local_path
    raster = render_historic_svg(source)
    assert raster.shape == (1, 32, 32)
    assert raster.dtype == torch.uint8
    assert 0 < int((raster == 0).sum()) < raster.numel()

    cache_path = tmp_path / "cache.pt"
    cached = load_or_build_historic_raster_cache(
        records,
        root=root,
        cache_path=cache_path,
        database_sha256="database",
        manifest_sha256="manifest",
        workers=0,
    )
    reloaded = load_or_build_historic_raster_cache(
        records,
        root=root,
        cache_path=cache_path,
        database_sha256="database",
        manifest_sha256="manifest",
        workers=0,
    )
    assert torch.equal(cached, reloaded)
    with pytest.raises(RuntimeError, match="provenance"):
        load_or_build_historic_raster_cache(
            records,
            root=root,
            cache_path=cache_path,
            database_sha256="changed",
            manifest_sha256="manifest",
            workers=0,
        )


def test_historic_dataset_and_collate_keep_metadata_outside_pixels(
    tmp_path: Path,
) -> None:
    _, _, records = write_historic_fixture(tmp_path)
    pixels = torch.zeros(len(records), 1, 32, 32, dtype=torch.uint8)
    split = historic_character_partition(records[0].character)
    dataset = HistoricGlyphRasterDataset(
        records,
        pixels,
        split=split,
        example_count=2,
    )
    batch = historic_glyph_collate([dataset[0], dataset[1]])
    assert batch["pixels"].shape == (2, 1, 32, 32)
    assert batch["pixels"].dtype == torch.float32
    assert isinstance(batch["metadata"][0]["character"], str)


def test_historic_training_order_has_a_reproducible_permutation() -> None:
    records = [
        HistoricGlyphRecord(
            character=chr(0x4E00 + index),
            stage="oracle",
            label=f"J{index}",
            local_path=f"glyph-{index}.svg",
        )
        for index in range(200)
    ]
    pixels = torch.ones(len(records), 1, 32, 32, dtype=torch.uint8)
    canonical = HistoricGlyphRasterDataset(records, pixels, split="train")
    first = HistoricGlyphRasterDataset(
        records,
        pixels,
        split="train",
        order_seed=20263400,
    )
    second = HistoricGlyphRasterDataset(
        records,
        pixels,
        split="train",
        order_seed=20263400,
    )
    assert first.indices == second.indices
    assert set(first.indices) == set(canonical.indices)
    assert first.indices != canonical.indices


def test_active_rendered_patches_uses_only_masked_cells() -> None:
    strip = torch.ones(1, 1, 32, 96)
    strip[..., :32] = 0
    batch = {
        "pixels": strip,
        "patch_mask": torch.tensor([[1.0, 0.0, 1.0]]),
    }
    patches = active_rendered_patches(batch)
    assert patches.shape == (2, 1, 32, 32)
    assert torch.equal(patches[0], torch.zeros_like(patches[0]))
    assert torch.equal(patches[1], torch.ones_like(patches[1]))

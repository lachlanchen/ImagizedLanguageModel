from __future__ import annotations

import hashlib
import io
import math
import os
import sqlite3
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cairosvg
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .direct_visual_patch_training import strip_to_patches


V34_HISTORIC_SPLITS = ("train", "development", "sealed")


@dataclass(frozen=True)
class HistoricGlyphRecord:
    character: str
    stage: str
    label: str
    local_path: str

    @property
    def identifier(self) -> str:
        return f"historic:{self.character}:{self.stage}:{self.label}:{self.local_path}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historic_character_partition(character: str) -> str:
    if not character:
        raise ValueError("V34 historical partition requires a character key")
    digest = hashlib.sha256(f"v34:historic:{character}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    if fraction < 0.90:
        return "train"
    if fraction < 0.95:
        return "development"
    return "sealed"


def load_historic_glyph_records(database_path: str | Path) -> list[HistoricGlyphRecord]:
    database = Path(database_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT c.char, g.stage, COALESCE(g.label, ''), g.local_path
            FROM glyphs AS g
            JOIN chars AS c ON c.id = g.char_id
            WHERE lower(g.local_path) LIKE '%.svg'
            ORDER BY c.char, g.stage, COALESCE(g.label, ''), g.local_path
            """
        ).fetchall()
    finally:
        connection.close()
    records = [
        HistoricGlyphRecord(
            character=str(character),
            stage=str(stage),
            label=str(label),
            local_path=str(local_path),
        )
        for character, stage, label, local_path in rows
    ]
    if not records:
        raise ValueError("V34 historical database contains no SVG glyphs")
    return records


def historic_svg_manifest_sha256(root: str | Path) -> tuple[str, int]:
    root_path = Path(root).expanduser().resolve()
    glyph_root = root_path / "data" / "historic" / "glyphs"
    paths = sorted(path for path in glyph_root.rglob("*.svg") if path.is_file())
    if not paths:
        raise ValueError(f"V34 found no SVG glyphs below {glyph_root}")
    manifest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root_path).as_posix()
        manifest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
    return manifest.hexdigest(), len(paths)


def render_historic_svg(
    path: str | Path,
    *,
    patch_size: int = 32,
    margin: int = 2,
) -> torch.Tensor:
    path = Path(path)
    if patch_size != 32 or not 0 <= margin < patch_size // 2:
        raise ValueError("V34 historical raster geometry is invalid")
    content_size = patch_size - 2 * margin
    png = cairosvg.svg2png(
        url=str(path),
        output_width=content_size,
        output_height=content_size,
        background_color="white",
    )
    with Image.open(io.BytesIO(png)) as rendered:
        glyph = rendered.convert("L")
    canvas = Image.new("L", (patch_size, patch_size), 255)
    canvas.paste(glyph, (margin, margin))
    array = np.asarray(canvas, dtype=np.uint8)
    binary = (array >= 128).astype(np.uint8)
    return torch.from_numpy(binary.copy()).unsqueeze(0)


def _render_historic_worker(path: str) -> torch.Tensor:
    return render_historic_svg(path)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_build_historic_raster_cache(
    records: Sequence[HistoricGlyphRecord],
    *,
    root: str | Path,
    cache_path: str | Path,
    database_sha256: str,
    manifest_sha256: str,
    workers: int = 4,
) -> torch.Tensor:
    if workers < 0:
        raise ValueError("V34 cache workers must be non-negative")
    root_path = Path(root).expanduser().resolve()
    cache = Path(cache_path)
    identifiers_sha256 = hashlib.sha256(
        "\n".join(record.identifier for record in records).encode("utf-8")
    ).hexdigest()
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        expected = {
            "database_sha256": database_sha256,
            "manifest_sha256": manifest_sha256,
            "identifiers_sha256": identifiers_sha256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError("V34 historical raster cache provenance differs")
        pixels = payload.get("pixels")
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("V34 historical raster cache lacks a tensor")
    else:
        paths = [str(root_path / record.local_path) for record in records]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"V34 historical SVG is missing: {missing[0]}")
        if workers == 0:
            rendered = [_render_historic_worker(path) for path in paths]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rendered = list(executor.map(_render_historic_worker, paths, chunksize=64))
        pixels = torch.stack(rendered).to(torch.uint8)
        _atomic_torch_save(
            {
                "database_sha256": database_sha256,
                "manifest_sha256": manifest_sha256,
                "identifiers_sha256": identifiers_sha256,
                "records": len(records),
                "pixels": pixels,
            },
            cache,
        )
    expected_shape = (len(records), 1, 32, 32)
    if pixels.dtype != torch.uint8 or tuple(pixels.shape) != expected_shape:
        raise ValueError("V34 historical raster cache has invalid shape or dtype")
    if not bool(((pixels == 0) | (pixels == 1)).all()):
        raise ValueError("V34 historical raster cache is not binary")
    return pixels


class HistoricGlyphRasterDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[HistoricGlyphRecord],
        pixels: torch.Tensor,
        *,
        split: str,
        example_count: int | None = None,
    ) -> None:
        if split not in V34_HISTORIC_SPLITS:
            raise ValueError(f"unknown V34 historical split: {split}")
        if len(records) != len(pixels):
            raise ValueError("V34 historical records and pixels do not align")
        self.records = tuple(records)
        self.pixels = pixels
        self.indices = tuple(
            index
            for index, record in enumerate(self.records)
            if historic_character_partition(record.character) == split
        )
        if not self.indices:
            raise ValueError(f"V34 historical split {split!r} is empty")
        self.split = split
        self.example_count = len(self.indices) if example_count is None else example_count
        if self.example_count < 1:
            raise ValueError("V34 historical example count must be positive")

    def __len__(self) -> int:
        return self.example_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.example_count:
            raise IndexError(index)
        source_index = self.indices[index % len(self.indices)]
        record = self.records[source_index]
        return {
            "pixels": self.pixels[source_index].float(),
            "metadata": {
                "identifier": record.identifier,
                "character": record.character,
                "stage": record.stage,
                "label": record.label,
                "split": self.split,
            },
        }


def historic_glyph_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("V34 cannot collate an empty historical batch")
    return {
        "pixels": torch.stack([sample["pixels"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }


def active_rendered_patches(batch: Mapping[str, Any]) -> torch.Tensor:
    pixels = batch.get("pixels")
    mask = batch.get("patch_mask")
    if not isinstance(pixels, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("V34 rendered batch requires pixel and mask tensors")
    patches = strip_to_patches(pixels, 32)
    if mask.shape != patches.shape[:2]:
        raise ValueError("V34 rendered mask does not align with patches")
    return patches[mask >= 0.5].contiguous()


def v34_historic_split_counts(records: Sequence[HistoricGlyphRecord]) -> dict[str, int]:
    counts = {split: 0 for split in V34_HISTORIC_SPLITS}
    characters = {split: set() for split in V34_HISTORIC_SPLITS}
    for record in records:
        split = historic_character_partition(record.character)
        counts[split] += 1
        characters[split].add(record.character)
    return {
        **{f"{split}_glyphs": count for split, count in counts.items()},
        **{
            f"{split}_characters": len(values)
            for split, values in characters.items()
        },
    }


def v34_codec_data_boundary_receipt(
    rendered_batch: Mapping[str, Any],
    historic_batch: Mapping[str, Any],
) -> dict[str, Any]:
    rendered = active_rendered_patches(rendered_batch)
    historic = historic_batch["pixels"]
    return {
        "student_keys": ["pixels"],
        "metadata_excluded": True,
        "all_student_values_are_tensors": True,
        "student_contains_strings": False,
        "rendered_patch_shape": list(rendered.shape),
        "historic_patch_shape": list(historic.shape),
        "rendered_binary": bool(((rendered == 0) | (rendered == 1)).all()),
        "historic_binary": bool(((historic == 0) | (historic == 1)).all()),
    }


def required_historic_variants(total_examples: int, split_examples: int) -> int:
    if total_examples < 1 or split_examples < 1:
        raise ValueError("V34 historical stream sizes must be positive")
    return math.ceil(total_examples / split_examples)

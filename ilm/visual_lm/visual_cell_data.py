from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .ink_jepa_data import VisualGrammarRecord, load_visual_grammar_manifest


V25_PARTITION_SALT = "ilm-v25-natural-chinese-cell-stream-20260813"
V25_MANIFEST_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
V25_TRAIN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
)
V25_DEVELOPMENT_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc",
)
V25_FROZEN_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc",
)
V25_FONT_PARTITIONS = {
    "train": V25_TRAIN_FONTS,
    "development": V25_DEVELOPMENT_FONTS,
    "frozen": V25_FROZEN_FONTS,
}
V25_STUDENT_IMAGE_KEYS = (
    "context",
    "target",
    "reference_context",
    "reference_target",
)


@dataclass(frozen=True)
class VisualCellRenderConfig:
    cell_size: int = 32
    sequence_cells: int = 65
    minimum_font_size: int = 24
    maximum_font_size: int = 28
    augment: bool = True
    script_views: str = "original+simplified"

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V25 fixes clean visual cells to 32x32")
        if self.sequence_cells != 65:
            raise ValueError("V25 fixes 65 cells and 64 causal transitions")
        if not 8 <= self.minimum_font_size <= self.maximum_font_size <= 32:
            raise ValueError("font sizes must fit inside the visual cell")
        if self.script_views not in {"original", "original+simplified"}:
            raise ValueError("script_views must be original or original+simplified")

    @property
    def context_cells(self) -> int:
        return self.sequence_cells - 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_v25_manifest(path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    source = Path(path)
    digest = file_sha256(source)
    if strict and digest != V25_MANIFEST_SHA256:
        raise ValueError(
            "V25 evidence requires the preregistered visual-grammar manifest; "
            f"expected {V25_MANIFEST_SHA256}, got {digest}"
        )
    records = 0
    source_titles: Counter[str] = Counter()
    rights: Counter[str] = Counter()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            records += 1
            source_titles[str(item.get("source_title", "unspecified"))] += 1
            rights[str(item.get("rights", "unspecified"))] += 1
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": digest,
        "strict_digest": strict,
        "records": records,
        "source_titles": dict(sorted(source_titles.items())),
        "rights": dict(sorted(rights.items())),
    }


def _partition_fraction(identifier: str) -> float:
    digest = hashlib.sha256(
        (V25_PARTITION_SALT + identifier).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def visual_cell_partition(identifier: str) -> str:
    fraction = _partition_fraction(identifier)
    if fraction < 0.03:
        return "frozen"
    if fraction < 0.06:
        return "development"
    return "train"


def visual_cell_partition_receipt(
    records: Sequence[VisualGrammarRecord],
) -> dict[str, Any]:
    identifiers = {record.identifier for record in records}
    partitions: dict[str, list[str]] = {
        "train": [],
        "development": [],
        "frozen": [],
    }
    for identifier in sorted(identifiers):
        partitions[visual_cell_partition(identifier)].append(identifier)
    receipt: dict[str, Any] = {
        "salt": V25_PARTITION_SALT,
        "identifiers": len(identifiers),
        "frozen_images_instantiated": False,
    }
    for name, values in partitions.items():
        receipt[f"{name}_identifiers"] = len(values)
        receipt[f"{name}_identifiers_sha256"] = hashlib.sha256(
            "\n".join(values).encode("utf-8")
        ).hexdigest()
    return receipt


@lru_cache(maxsize=64)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


@lru_cache(maxsize=1)
def _v25_shared_codepoints() -> frozenset[int]:
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise RuntimeError("fontTools is required for V25 cmap validation") from exc
    shared: set[int] | None = None
    for value in (*V25_TRAIN_FONTS, *V25_DEVELOPMENT_FONTS, *V25_FROZEN_FONTS):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"V25 font is missing: {path}")
        font = TTFont(path, fontNumber=0, lazy=True)
        try:
            coverage = set(font.getBestCmap() or {})
        finally:
            font.close()
        shared = coverage if shared is None else shared.intersection(coverage)
    return frozenset(shared or ())


def visual_cell_character_supported(character: str) -> bool:
    if len(character) != 1:
        raise ValueError("visual-cell coverage accepts one character")
    return not character.isspace() and ord(character) in _v25_shared_codepoints()


def visible_visual_writing(text: str) -> str:
    return "".join(
        character for character in text if visual_cell_character_supported(character)
    )


@lru_cache(maxsize=131_072)
def _base_cell_bytes(character: str, font_path: str, font_size: int) -> bytes:
    if not visual_cell_character_supported(character):
        raise ValueError(f"unsupported V25 visual character: {character!r}")
    cell_size = 32
    image = Image.new("L", (cell_size, cell_size), 255)
    draw = ImageDraw.Draw(image)
    font = _load_font(font_path, font_size)
    box = draw.textbbox((0, 0), character, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x = (cell_size - width) / 2 - box[0]
    y = (cell_size - height) / 2 - box[1]
    draw.text((x, y), character, font=font, fill=0)
    ink = 255 - np.asarray(image, dtype=np.uint8)
    return ink.tobytes()


def _translate_cells(cells: torch.Tensor, y: int, x: int) -> torch.Tensor:
    if not y and not x:
        return cells
    output = torch.zeros_like(cells)
    source_y0 = max(0, -y)
    source_y1 = cells.shape[-2] - max(0, y)
    source_x0 = max(0, -x)
    source_x1 = cells.shape[-1] - max(0, x)
    target_y0 = max(0, y)
    target_y1 = cells.shape[-2] - max(0, -y)
    target_x0 = max(0, x)
    target_x1 = cells.shape[-1] - max(0, -x)
    output[..., target_y0:target_y1, target_x0:target_x1] = cells[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return output


def render_visual_cell_stream(
    writing: str,
    *,
    config: VisualCellRenderConfig,
    font_path: str,
    variant: int,
) -> torch.Tensor:
    """Render writing as an inspectable [T, 1, 32, 32] visual-time volume."""

    if not writing:
        raise ValueError("cannot render an empty visual stream")
    if any(not visual_cell_character_supported(character) for character in writing):
        raise ValueError("visual stream contains unsupported or invisible characters")
    if font_path not in (*V25_TRAIN_FONTS, *V25_DEVELOPMENT_FONTS, *V25_FROZEN_FONTS):
        raise ValueError("font is outside the frozen V25 font partition")
    rng = random.Random(variant)
    font_size = (
        rng.randint(config.minimum_font_size, config.maximum_font_size)
        if config.augment
        else (config.minimum_font_size + config.maximum_font_size) // 2
    )
    arrays = [
        np.frombuffer(
            _base_cell_bytes(character, font_path, font_size), dtype=np.uint8
        ).reshape(32, 32)
        for character in writing
    ]
    cells = torch.from_numpy(np.stack(arrays).copy()).float().div_(255.0)[:, None]
    if config.augment:
        cells = _translate_cells(cells, rng.randint(-1, 1), rng.randint(-1, 1))
        cells = (cells * rng.uniform(0.82, 1.12)).clamp_(0.0, 1.0)
        if rng.random() < 0.28:
            blurred = F.avg_pool2d(cells, 3, stride=1, padding=1)
            cells = torch.lerp(cells, blurred, rng.uniform(0.06, 0.18))
        if rng.random() < 0.30:
            generator = torch.Generator().manual_seed(variant)
            noise = torch.randn(cells.shape, generator=generator) * rng.uniform(
                0.002, 0.012
            )
            cells = (cells + noise).clamp_(0.0, 1.0)
    return cells.contiguous()


@lru_cache(maxsize=1)
def _opencc_t2s() -> Any:
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise RuntimeError(
            "V25 script_views=original+simplified requires the opencc package"
        ) from exc
    return OpenCC("t2s")


def script_variants(
    record: VisualGrammarRecord,
    *,
    mode: str,
) -> tuple[tuple[str, str], ...]:
    if mode not in {"original", "original+simplified"}:
        raise ValueError("unknown V25 script-view mode")
    output = [("original", visible_visual_writing(record.text))]
    if mode == "original+simplified":
        simplified = visible_visual_writing(_opencc_t2s().convert(record.text))
        if simplified and simplified != output[0][1]:
            output.append(("simplified", simplified))
    return tuple((name, writing) for name, writing in output if writing)


class VisualCellStreamDataset(Dataset):
    """Prepare strings offline and expose only continuous visual-cell streams."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        split: str,
        render_config: VisualCellRenderConfig,
        seed: int = 0,
        length: int | None = None,
        expose_evaluation_labels: bool = False,
        allow_frozen: bool = False,
    ) -> None:
        if split not in V25_FONT_PARTITIONS:
            raise ValueError("split must be train, development, or frozen")
        if split == "frozen" and not allow_frozen:
            raise PermissionError(
                "V25 frozen visual streams remain sealed until development selects"
            )
        selected: list[tuple[VisualGrammarRecord, str, str]] = []
        for record in records:
            if visual_cell_partition(record.identifier) != split:
                continue
            for script_name, writing in script_variants(
                record, mode=render_config.script_views
            ):
                if len(writing) >= render_config.sequence_cells:
                    selected.append((record, script_name, writing))
        if not selected:
            raise ValueError(f"no usable V25 visual streams for split={split}")
        self.records = selected
        self.split = split
        self.config = render_config
        self.seed = int(seed)
        self.length = int(length) if length is not None else len(selected)
        self.expose_evaluation_labels = bool(expose_evaluation_labels)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(
            self.seed + self.epoch * 10_000_019 + index * 104_729
        )
        if self.length <= len(self.records):
            record, script_name, writing = self.records[index % len(self.records)]
        else:
            record, script_name, writing = rng.choice(self.records)
        start = rng.randint(0, len(writing) - self.config.sequence_cells)
        segment = writing[start : start + self.config.sequence_cells]
        fonts = V25_FONT_PARTITIONS[self.split]
        first_font_index = rng.randrange(len(fonts))
        second_font_index = (
            first_font_index + 1 + rng.randrange(len(fonts) - 1)
        ) % len(fonts)
        first_variant = rng.randrange(2**31)
        second_variant = rng.randrange(2**31)
        first = render_visual_cell_stream(
            segment,
            config=self.config,
            font_path=fonts[first_font_index],
            variant=first_variant,
        )
        second = render_visual_cell_stream(
            segment,
            config=self.config,
            font_path=fonts[second_font_index],
            variant=second_variant,
        )
        metadata: dict[str, Any] = {
            "identifier": record.identifier,
            "source": record.source,
            "rights": record.rights,
            "script_view": script_name,
            "offset": start,
            "first_font": fonts[first_font_index],
            "second_font": fonts[second_font_index],
        }
        if self.expose_evaluation_labels:
            metadata["context_characters"] = segment[:-1]
            metadata["target_characters"] = segment[1:]
        return {
            "context": first[:-1],
            "target": first[1:],
            "reference_context": second[:-1],
            "reference_target": second[1:],
            "metadata": metadata,
        }


def visual_cell_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty visual-cell batch")
    return {
        key: torch.stack([item[key] for item in batch])
        for key in V25_STUDENT_IMAGE_KEYS
    } | {"metadata": [item["metadata"] for item in batch]}


def assert_image_only_student_batch(batch: Mapping[str, Any]) -> None:
    if set(batch) != set(V25_STUDENT_IMAGE_KEYS):
        raise ValueError(
            "V25 student batch must contain only the four registered image streams"
        )
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"V25 student value {key!r} is not a floating image tensor")
        if value.ndim != 5 or tuple(value.shape[2:]) != (1, 32, 32):
            raise ValueError(
                f"V25 student value {key!r} must be [B,T,1,32,32]"
            )


def student_visual_cell_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    student = {key: batch[key] for key in V25_STUDENT_IMAGE_KEYS}
    assert_image_only_student_batch(student)
    return student


def visual_cell_boundary_receipt() -> dict[str, bool | str | list[int]]:
    return {
        "architecture": "visual-cell-stream-v25",
        "native_sample_shape": [65, 1, 32, 32],
        "input_is_continuous_image_stream": True,
        "output_is_continuous_image_stream": True,
        "sequence_axis_is_visual_time": True,
        "geometric_depth_is_one": True,
        "rereads_generated_pixels": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_color_identity_channel": False,
        "uses_visual_codebook": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }


def visual_cell_font_manifest() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, fonts in V25_FONT_PARTITIONS.items():
        output[split] = [
            {
                "path": path,
                "bytes": Path(path).stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in fonts
        ]
    return output


def pack_visual_cells(
    cells: torch.Tensor,
    *,
    columns: int = 16,
    gutter: int = 1,
) -> Image.Image:
    """Pack a visual-time volume into a page without changing model semantics."""

    if cells.ndim == 5:
        if cells.shape[0] != 1:
            raise ValueError("page packing accepts one stream")
        cells = cells[0]
    if cells.ndim != 4 or tuple(cells.shape[1:]) != (1, 32, 32):
        raise ValueError("cells must have shape [T,1,32,32]")
    if columns < 1 or gutter < 0:
        raise ValueError("columns must be positive and gutter non-negative")
    rows = math.ceil(cells.shape[0] / columns)
    width = columns * 32 + max(0, columns - 1) * gutter
    height = rows * 32 + max(0, rows - 1) * gutter
    page = np.full((height, width), 255, dtype=np.uint8)
    ink = cells.detach().float().cpu().clamp(0, 1).numpy()
    for index in range(cells.shape[0]):
        row, column = divmod(index, columns)
        top = row * (32 + gutter)
        left = column * (32 + gutter)
        page[top : top + 32, left : left + 32] = np.round(
            (1.0 - ink[index, 0]) * 255.0
        ).astype(np.uint8)
    return Image.fromarray(page, mode="L")


def load_v25_records(
    path: str | Path, *, strict_manifest: bool
) -> list[VisualGrammarRecord]:
    verify_v25_manifest(path, strict=strict_manifest)
    return load_visual_grammar_manifest(path)


def iter_split_writing(
    records: Sequence[VisualGrammarRecord],
    *,
    split: str,
    script_views_mode: str,
) -> Iterable[tuple[VisualGrammarRecord, str, str]]:
    """Host-side iterator for evaluator statistics; never pass it to the student."""

    for record in records:
        if visual_cell_partition(record.identifier) != split:
            continue
        for script_name, writing in script_variants(record, mode=script_views_mode):
            yield record, script_name, writing


def visual_cell_render_config_payload(
    config: VisualCellRenderConfig,
) -> dict[str, Any]:
    return asdict(config)


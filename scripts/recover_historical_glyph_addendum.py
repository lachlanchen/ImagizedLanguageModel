#!/usr/bin/env python3
"""Recover cached later entries and preserve higher-resolution narrow glyphs."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image, ImageDraw

from ilm.etymology.hanziyuan import _decode_data_uri, parse_hanziyuan_entries
from ilm.image_edit.baseline import ROOT, save_json, sha


def load_verified(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Source changed: {path}")
    return json.loads(path.read_text())


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def later_assets(cache, source_rows, requested):
    assets = {}
    found = set()
    for char, expected in requested.items():
        source = source_rows[char]
        payload = (cache / source["file"]).read_bytes()
        if digest(payload) != source["sha256"]:
            raise ValueError(f"Cached source changed for {char}")
        for entry in parse_hanziyuan_entries(payload.decode("utf-8")):
            for glyph in entry.glyphs:
                if not glyph.src.startswith("data:image/svg+xml"):
                    continue
                svg, _ = _decode_data_uri(glyph.src)
                svg_sha = digest(svg)
                key = (entry.entry_id, glyph.stage, glyph.label, svg_sha)
                if key in expected and entry.query_char == char:
                    assets[svg_sha] = svg
                    found.add((char, *key))
    if len(found) != sum(map(len, requested.values())):
        raise ValueError("Some recorded later-entry SVGs could not be recovered")
    return assets


def binary_render(svg, size):
    root = ET.fromstring(svg)
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("Expected an SVG document")
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] not in {"svg", "metadata", "g", "path"}:
            raise ValueError("Unexpected element in historical path SVG")
        if any(key.rsplit("}", 1)[-1] in {"href", "src"} or "url(" in value.lower()
               for key, value in node.attrib.items()):
            raise ValueError("Unexpected reference in historical path SVG")
    margin = size // 16
    width = size - 2 * margin
    png = cairosvg.svg2png(bytestring=svg, output_width=width, output_height=width,
                           background_color="white")
    with Image.open(BytesIO(png)) as rendered:
        canvas = Image.new("L", (size, size), 255)
        canvas.paste(rendered.convert("L"), (margin, margin))
    return np.asarray(canvas) >= 128


def review_blank_rasters(rows, source_entries, legacy, output):
    from ilm.visual_lm.continuous_glyph_codec_data import load_historic_glyph_records

    records = load_historic_glyph_records(legacy / "data/historic/etymology.sqlite3")
    board = Image.new("L", (565, 48 + 145 * len(rows)), 255)
    draw = ImageDraw.Draw(board)
    draw.text((10, 8), "Bronze glyphs whose ink disappears at 32 x 32", fill=0)
    columns = (88, 245, 402)
    sizes = (32, 64, 128)
    for x, size in zip(columns, sizes):
        draw.text((x, 28), f"{size} x {size}", fill=0)
    results = []
    for i, row in enumerate(rows):
        record = records[row["raster_index"]]
        glyph, = [g for g in source_entries[row["char"], row["entry_id"]]["glyphs"]
                  if g["source_local_path"] == record.local_path]
        payload = (legacy / record.local_path).read_bytes()
        if digest(payload) != glyph["sha256"]:
            raise ValueError("Blank-raster source SVG changed")
        y = 48 + i * 145
        draw.text((10, y + 8), record.label, fill=0)
        measurements = {}
        for x, size in zip(columns, sizes):
            pixels = binary_render(payload, size)
            image = Image.fromarray(pixels.astype(np.uint8) * 255)
            name = f"{record.label}_{size}.png"
            image.save(output / name)
            board.paste(image.resize((128, 128), Image.Resampling.NEAREST), (x, y))
            draw.rectangle((x - 1, y - 1, x + 128, y + 128), outline=180)
            measurements[str(size)] = {"ink_pixels": int((~pixels).sum()), "file": name,
                                       "sha256": sha(output / name)}
        if measurements["32"]["ink_pixels"]:
            raise ValueError("The original blank raster did not reproduce")
        results.append({**row, "stage": record.stage, "label": record.label,
                        "source_svg_sha256": glyph["sha256"], "rasters": measurements})
    board.save(output / "resolution_review.png")
    return results, sha(output / "resolution_review.png")


def prepare(legacy, source, splits, output):
    import torch

    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    source_report = json.loads((source / "report.json").read_text())
    source_entries = load_verified(source / "entries.json", source_report["entries_sha256"])
    cached = load_verified(source / "cached_sources.json", source_report["cached_sources_sha256"])
    split_report = json.loads((splits / "report.json").read_text())
    if split_report["source_entries_sha256"] != source_report["entries_sha256"]:
        raise ValueError("Historical partition does not match the recovered source entries")
    partitions = {name: load_verified(splits / f"{name}.json", stats["file_sha256"])
                  for name, stats in split_report["partitions"].items()}
    requested = defaultdict(set)
    owners = {}
    for split, entries in partitions.items():
        for entry in entries:
            for glyph in entry["source_glyphs_without_matching_asset"]:
                requested[entry["char"]].add((entry["entry_id"], glyph["stage"], glyph["label"], glyph["sha256"]))
            for glyph in entry["glyphs"]:
                owners[glyph["raster_sha256"]] = split
    svg_assets = later_assets(legacy / "data/historic/cache", {row["char"]: row for row in cached}, requested)
    output.mkdir(parents=True)
    (output / "svg").mkdir()
    rasters = []
    raster_index = {}
    for svg_sha, payload in sorted(svg_assets.items()):
        (output / "svg" / f"{svg_sha}.svg").write_bytes(payload)
        raster_index[svg_sha] = len(rasters)
        rasters.append(binary_render(payload, 32))
    tensor = torch.from_numpy(np.stack(rasters)[:, None].astype(np.uint8))
    torch.save({"pixels": tensor}, output / "later_rasters.pt")

    measurements = {}
    excluded = []
    added_pixels = {}
    for split, entries in partitions.items():
        additions = []
        for entry in entries:
            glyphs = []
            for glyph in entry["source_glyphs_without_matching_asset"]:
                svg_sha = glyph["sha256"]
                index = raster_index[svg_sha]
                raster = rasters[index]
                raster_sha = digest(np.packbits(raster, bitorder="big").tobytes())
                row = {**glyph, "svg_file": f"svg/{svg_sha}.svg", "raster_index": index, "raster_sha256": raster_sha}
                prior_split = owners.get(raster_sha, added_pixels.get(raster_sha))
                if raster.all() or (prior_split is not None and prior_split != split):
                    reason = "blank_raster" if raster.all() else "raster_matches_another_partition"
                    excluded.append({**row, "char": entry["char"], "entry_id": entry["entry_id"],
                                     "split": split, "reason": reason})
                    continue
                added_pixels[raster_sha] = split
                glyphs.append(row)
            if glyphs:
                additions.append({key: entry[key] for key in ("char", "entry_id", "family_sha256", "fields")}
                                 | {"glyphs": glyphs})
        file = output / f"{split}.json"
        save_json(file, additions)
        measurements[split] = {"entries": len(additions), "glyphs": sum(len(row["glyphs"]) for row in additions),
                               "file_sha256": sha(file)}
    save_json(output / "excluded.json", excluded)
    blanks = load_verified(splits / "blank_rasters.json", split_report["blank_rasters_sha256"])
    resolution, board_sha = review_blank_rasters(
        blanks, {(entry["char"], entry["entry_id"]): entry for entry in source_entries}, legacy, output
    )
    report = {
        "source_report_sha256": sha(source / "report.json"),
        "base_split_report_sha256": sha(splits / "report.json"),
        "source_links_recovered": sum(map(len, requested.values())),
        "source_characters": len(requested),
        "unique_recovered_SVGs": len(svg_assets),
        "partitions": measurements,
        "excluded_by_reason": dict(Counter(row["reason"] for row in excluded)),
        "excluded_sha256": sha(output / "excluded.json"),
        "later_rasters_sha256": sha(output / "later_rasters.pt"),
        "resolution_review_sha256": board_sha,
        "previously_blank_source_glyphs": resolution,
        "source_code_sha256": sha(Path(__file__)),
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=ROOT.parent / "incoder")
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/historic_entry_metadata_v1")
    parser.add_argument("--splits", type=Path, default=ROOT / "artifacts/historical_glyph_splits_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/historical_glyph_addendum_v1")
    args = parser.parse_args()
    prepare(args.legacy, args.source, args.splits, args.output)

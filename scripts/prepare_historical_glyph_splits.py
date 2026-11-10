#!/usr/bin/env python3
"""Partition recovered historical entries by source identity and exact pixels."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from ilm.image_edit.baseline import ROOT, save_json, sha


NAMESPACE = "historical-glyph-content-v1"
SPLITS = ("train", "development", "sealed")
WHITE_PATCH = b"\xff" * 128


def load(path):
    return json.loads(path.read_text())


def raster_cache(legacy_root, cache_path):
    import numpy as np
    import torch

    from ilm.visual_lm.continuous_glyph_codec_data import (
        historic_svg_manifest_sha256,
        load_historic_glyph_records,
    )

    torch.set_num_threads(2)
    database = legacy_root / "data/historic/etymology.sqlite3"
    records = load_historic_glyph_records(database)
    source_manifest, source_file_count = historic_svg_manifest_sha256(legacy_root)
    identities = hashlib.sha256("\n".join(row.identifier for row in records).encode()).hexdigest()
    provenance = {
        "database_sha256": sha(database),
        "manifest_sha256": source_manifest,
        "identifiers_sha256": identities,
        "records": len(records),
    }
    payload = torch.load(cache_path, weights_only=True, map_location="cpu")
    if any(payload.get(key) != value for key, value in provenance.items()):
        raise ValueError("Historical raster cache provenance differs from the source inventory")
    pixels = payload["pixels"].numpy()
    if (pixels.shape != (len(records), 1, 32, 32) or pixels.dtype != np.uint8
            or pixels.max() > 1):
        raise ValueError("Expected 32×32 binary historical rasters in source order")
    packed = np.packbits(pixels.reshape(len(records), -1), axis=1, bitorder="big")
    return records, [row.tobytes() for row in packed], {
        **provenance, "source_svg_files": source_file_count, "cache_sha256": sha(cache_path),
    }


def earlier_overlap(records, packed, earlier_run):
    from ilm.visual_lm.continuous_glyph_codec_data import historic_character_partition

    groups = {split: [] for split in SPLITS}
    for record, pixels in zip(records, packed):
        groups[historic_character_partition(record.character)].append(pixels)
    train = set(groups["train"]) - {WHITE_PATCH}
    summary = {}
    for split, filename in [("development", "development_ema.json"), ("sealed", "sealed_ema.json")]:
        all_rows = groups[split]
        old_report = earlier_run / filename
        evaluated = load(old_report)["historical"]["patches"]
        if not 0 < evaluated <= len(all_rows):
            raise ValueError("Earlier evaluation size differs from its historical partition")
        # The original evaluator walks the sorted cache without shuffling.
        summary[split] = {
            "partition_records": len(all_rows),
            "partition_records_with_nonblank_training_raster": sum(row in train for row in all_rows),
            "evaluated_records": evaluated,
            "evaluated_records_with_nonblank_training_raster": sum(row in train for row in all_rows[:evaluated]),
            "evaluation_report_sha256": sha(old_report),
        }
    return summary


def group_entries(entries, records, packed):
    index = {(row.character, row.stage, row.label, row.local_path): i for i, row in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("Historical raster record identity is ambiguous")
    parents = {entry["char"]: entry["char"] for entry in entries}
    owners = {}

    def root(char):
        while parents[char] != char:
            parents[char] = parents[parents[char]]
            char = parents[char]
        return char

    def group(char, relation, value):
        owner = owners.setdefault((relation, value), char)
        parents[root(owner)] = root(char)

    blank_rows = []
    for entry in entries:
        char = entry["char"]
        group(char, "source_entry", entry["entry_id"])
        for form in {char, entry["fields"].get("traditional"), entry["fields"].get("simplified")}:
            if form and len(form) == 1:
                group(char, "modern_form", form)
        for glyph in entry["source_glyphs_without_matching_asset"]:
            group(char, "svg", glyph["sha256"])
        retained = []
        for glyph in entry["glyphs"]:
            row_index = index[char, glyph["stage"], glyph["label"], glyph["source_local_path"]]
            pixels = packed[row_index]
            group(char, "svg", glyph["sha256"])
            if pixels == WHITE_PATCH:
                blank_rows.append({"char": char, "entry_id": entry["entry_id"], "raster_index": row_index})
                continue
            group(char, "raster", pixels)
            retained.append({**glyph, "raster_index": row_index,
                             "raster_sha256": hashlib.sha256(pixels).hexdigest()})
        entry["glyphs"] = retained

    families = defaultdict(list)
    for char in parents:
        families[root(char)].append(char)
    assignments = {}
    for members in families.values():
        digest = hashlib.sha256("\0".join([NAMESPACE, *sorted(members)]).encode()).hexdigest()
        bucket = int(digest, 16) % 100
        split = "train" if bucket < 80 else "development" if bucket < 90 else "sealed"
        for char in members:
            assignments[char] = (split, digest)

    relation_splits = {}
    for (relation, value), owner in owners.items():
        relation_splits[relation, value] = assignments[owner][0]
    partitions = {split: [] for split in SPLITS}
    for entry in entries:
        split, family = assignments[entry["char"]]
        relations = [("source_entry", entry["entry_id"])]
        forms = [entry["char"], entry["fields"].get("traditional"), entry["fields"].get("simplified")]
        relations.extend(("modern_form", form) for form in forms if form and len(form) == 1)
        source_glyphs = entry["glyphs"] + entry["source_glyphs_without_matching_asset"]
        relations.extend(("svg", glyph["sha256"]) for glyph in source_glyphs)
        relations.extend(("raster", packed[glyph["raster_index"]]) for glyph in entry["glyphs"])
        if any(relation_splits[key] != split for key in relations):
            raise ValueError("A historical source identity crosses partitions")
        partitions[split].append({**entry, "family_sha256": family})
    return partitions, blank_rows, max(map(len, families.values()))


def prepare(source, legacy_root, cache_path, earlier_run, output):
    if output.exists():
        raise FileExistsError(output)
    recovered = load(source / "report.json")
    entries_path = source / "entries.json"
    if sha(entries_path) != recovered["entries_sha256"]:
        raise ValueError("Recovered historical entry manifest changed")
    entries = load(entries_path)
    records, packed, provenance = raster_cache(legacy_root, cache_path)
    prior = earlier_overlap(records, packed, earlier_run)
    partitions, blank_rows, largest_family = group_entries(entries, records, packed)
    output.mkdir(parents=True)
    metrics = {}
    for split, rows in partitions.items():
        path = output / f"{split}.json"
        save_json(path, rows)
        glyphs = [glyph for row in rows for glyph in row["glyphs"]]
        metrics[split] = {
            "families": len({row["family_sha256"] for row in rows}),
            "characters": len({row["char"] for row in rows}),
            "source_entries": len(rows),
            "glyph_records": len(glyphs),
            "unique_binary_rasters": len({glyph["raster_sha256"] for glyph in glyphs}),
            "glyph_records_by_stage": dict(Counter(glyph["stage"] for glyph in glyphs)),
            "entries_with_meaning_pronunciation_and_glyphs": sum(
                bool(row["glyphs"]) and "main_pronunciation" in row["fields"]
                and "original_meaning" in row["fields"] for row in rows
            ),
            "file_sha256": sha(path),
        }
    save_json(output / "blank_rasters.json", blank_rows)
    report = {
        "namespace": NAMESPACE,
        "source_entries_sha256": recovered["entries_sha256"],
        "raster_cache": provenance,
        "grouped_by": ["query character", "traditional and simplified forms", "source entry ID",
                       "identical SVG bytes", "identical nonblank 32×32 binary pixels"],
        "largest_family_characters": largest_family,
        "partitions": metrics,
        "blank_rasters_excluded": len(blank_rows),
        "blank_rasters_sha256": sha(output / "blank_rasters.json"),
        "earlier_character_only_split_overlap": prior,
        "cross_partition_source_or_exact_raster_overlap": False,
        "near_duplicate_raster_comparison_performed": False,
        "inherited_pretraining_exposure_known": False,
        "new_rasterization_or_neural_evaluation_performed": False,
        "source_code_sha256": sha(Path(__file__)),
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/historic_entry_metadata_v1")
    parser.add_argument("--legacy-root", type=Path, default=ROOT.parent / "incoder")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/cache/v34_historic_glyph_rasters_32.pt")
    parser.add_argument("--earlier-run", type=Path, default=ROOT / "artifacts/continuous_glyph_codec_v34_20260814")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/historical_glyph_splits_v1")
    args = parser.parse_args()
    prepare(args.source, args.legacy_root, args.cache, args.earlier_run, args.output)

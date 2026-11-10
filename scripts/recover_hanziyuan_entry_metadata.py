#!/usr/bin/env python3
"""Link cached Hanziyuan descriptions to the existing historical SVG inventory."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from ilm.etymology.hanziyuan import _decode_data_uri, parse_hanziyuan_entries
from ilm.image_edit.baseline import ROOT, save_json, sha


def load(path):
    return json.loads(path.read_text())


def recover(database, cache, inventory, output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    report = load(inventory / "report.json")
    inventory_file = inventory / "records.json"
    if sha(inventory_file) != report["records_sha256"]:
        raise ValueError("Curated glyph inventory changed")
    curated = load(inventory_file)
    glyph_index = {
        (row["char"], row["stage"], row["label"], row["sha256"]): row
        for row in curated
    }

    before = database.stat()
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        chars = connection.execute("SELECT char, pinyin, main_meaning FROM chars ORDER BY char").fetchall()
    cached = {
        path.stem.removeprefix("hanziyuan_ajax_"): path
        for path in cache.glob("hanziyuan_ajax_*.html")
    }
    field_counts = Counter()
    cache_sources = []
    records = []
    missing = []
    linked = set()
    for char, _, _ in chars:
        path = cached.get(char)
        if path is None:
            missing.append({"char": char, "reason": "missing_cached_fragment"})
            continue
        payload = path.read_bytes()
        cache_sources.append({"char": char, "file": path.name, "bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest()})
        entries = parse_hanziyuan_entries(payload.decode("utf-8"))
        if not entries:
            missing.append({"char": char, "reason": "no_source_etymology_entries"})
            continue
        if any(entry.query_char != char for entry in entries):
            missing.append({"char": char, "reason": "source_character_mismatch"})
            continue
        for entry in entries:
            assets = []
            absent = []
            seen = set()
            for glyph in entry.glyphs:
                if not glyph.src.startswith("data:image/svg+xml"):
                    continue
                svg, _ = _decode_data_uri(glyph.src)
                digest = hashlib.sha256(svg).hexdigest()
                key = (char, glyph.stage, glyph.label, digest)
                if key in seen:
                    continue
                seen.add(key)
                description = {"stage": glyph.stage, "label": glyph.label, "sha256": digest}
                row = glyph_index.get(key)
                if row is None:
                    absent.append(description)
                else:
                    linked.add(key)
                    assets.append({**description, "source_local_path": row["source_local_path"]})
            records.append({"char": char, "entry_id": entry.entry_id, "fields": entry.fields,
                            "glyphs": assets, "source_glyphs_without_matching_asset": absent})
            field_counts.update(entry.fields.keys())

    after = database.stat()
    output.mkdir(parents=True)
    save_json(output / "entries.json", records)
    save_json(output / "cached_sources.json", cache_sources)
    save_json(output / "unavailable.json", missing)
    chars_with_entries = {row["char"] for row in records}
    chars_with_assets = {row["char"] for row in records if row["glyphs"]}
    summary = {
        "database_character_rows": len(chars),
        "database_rows_with_pinyin": sum(bool(pinyin) for _, pinyin, _ in chars),
        "database_rows_with_meaning": sum(bool(meaning) for _, _, meaning in chars),
        "cached_fragments_examined": len(cache_sources),
        "cached_fragment_bytes": sum(row["bytes"] for row in cache_sources),
        "characters_with_source_entries": len(chars_with_entries),
        "characters_with_matched_glyphs": len(chars_with_assets),
        "characters_with_multiple_source_entries": sum(
            count > 1 for count in Counter(row["char"] for row in records).values()
        ),
        "source_entries": len(records),
        "entries_with_matched_glyphs": sum(bool(row["glyphs"]) for row in records),
        "entries_with_pinyin_meaning_and_matched_glyphs": sum(
            bool(row["glyphs"]) and "main_pronunciation" in row["fields"]
            and "original_meaning" in row["fields"] for row in records
        ),
        "source_field_counts": dict(field_counts),
        "curated_glyph_records": len(curated),
        "unique_curated_glyphs_linked_by_content": len(linked),
        "source_glyph_entry_links_without_matching_asset": sum(
            len(row["source_glyphs_without_matching_asset"]) for row in records
        ),
        "unavailable_character_reasons": dict(Counter(row["reason"] for row in missing)),
        "curated_inventory_sha256": report["records_sha256"],
        "entries_sha256": sha(output / "entries.json"),
        "cached_sources_sha256": sha(output / "cached_sources.json"),
        "unavailable_sha256": sha(output / "unavailable.json"),
        "parser_sha256": sha(ROOT / "ilm/etymology/hanziyuan.py"),
        "recovery_script_sha256": sha(Path(__file__)),
        "database_stat_unchanged": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "database_opened_read_only": True,
        "source_linguistic_claims_independently_verified": False,
        "network_requests": 0,
        "model_calls": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(output / "report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    legacy = ROOT.parent / "incoder/data/historic"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=legacy / "etymology.sqlite3")
    parser.add_argument("--cache", type=Path, default=legacy / "cache")
    parser.add_argument("--inventory", type=Path, default=ROOT / "artifacts/historic_stage_checked_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/historic_entry_metadata_v1")
    args = parser.parse_args()
    recover(args.db, args.cache, args.inventory, args.output)

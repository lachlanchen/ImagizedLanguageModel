#!/usr/bin/env python3
"""CPU-only inventory of a new preference source under the existing cell policy.

This inspects source rendering, not a checkpoint or generated model output.
No source string, existing renderer, live trial or acceptance gate is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "references/evidence/chinese_preference_raster_registration_20261116.json"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FIELDS = ("system", "prompt", "chosen", "rejected")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Cannot observe available host RAM")


def check(start: float) -> None:
    if time.monotonic()-start > 120:
        raise RuntimeError("120-second source-audit ceiling exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 > 1:
        raise RuntimeError("1-GiB peak RSS ceiling exceeded")
    if available_gib() < 24:
        raise RuntimeError("24-GiB available host reserve breached")


def records(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def raster(font, char: str):
    """Original geometry and an independently sized canvas at the same origin."""
    small = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(small)
    x = (32-draw.textlength(char, font=font))/2
    draw.text((x, 26), char, font=font, fill=0, anchor="ls")
    # Integer translation preserves subpixel origin; this canvas checks overflow.
    large = Image.new("L", (128, 128), 255)
    ImageDraw.Draw(large).text((x+48, 74), char, font=font, fill=0, anchor="ls")
    native = np.asarray(small) < 128
    reference = np.asarray(large) < 128
    if reference[[0, -1]].any() or reference[:, [0, -1]].any():
        raise RuntimeError("Expanded reference canvas is itself insufficient")
    if not np.array_equal(native, reference[48:80, 48:80]):
        raise RuntimeError("Translated crop differs from original cell")
    ys, xs = np.nonzero(reference)
    bbox = None if not len(xs) else [int(xs.min()-48), int(ys.min()-48), int(xs.max()-48), int(ys.max()-48)]
    outside = reference.copy()
    outside[48:80, 48:80] = False
    return native, {
        "full_ink_bbox_in_cell_coordinates_inclusive": bbox,
        "ink_pixels_in_cell": int(native.sum()),
        "ink_pixels_outside_cell": int(outside.sum()),
        "clipped": bool(outside.any()),
        "touches_cell_frame": bool(native[[0,-1]].any() or native[:,[0,-1]].any()),
        "raster_sha256": hashlib.sha256(np.packbits(native, bitorder="big").tobytes()).hexdigest(),
        "translated_crop_equals_original": True,
    }


def audit(output: Path) -> dict:
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("Use CPU-only execution and a new evidence directory")
    if available_gib() < 48:
        raise RuntimeError("48-GiB entry reserve required alongside the live fit")
    os.nice(10)
    registration = json.loads(REGISTRATION.read_text())
    for path, expected in registration["input_sha256"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("Registered input changed: " + path)
    if sha(Path(__file__)) != registration["script_sha256"] or sha(FONT) != registration["font_sha256"]:
        raise ValueError("Audit implementation or shared font changed")
    source = ROOT/registration["pairs_file"]
    output.mkdir(parents=True)
    frequencies = {field: Counter() for field in FIELDS}
    n = 0
    for row in records(source):
        for field in FIELDS:
            frequencies[field].update(row["raw"][field])
        n += 1
        if n % 128 == 0:
            check(started)
    if n != 10735:
        raise ValueError("Complete prepared source required")
    total = sum(frequencies.values(), Counter())
    with TTFont(FONT, fontNumber=2, lazy=True) as face:
        coverage = {cp for cp, name in face.getBestCmap().items() if name != ".notdef"}
    font = ImageFont.truetype(str(FONT), 26, index=2)
    glyphs = {}
    masks = {}
    collisions = defaultdict(list)
    for number, char in enumerate(sorted(total)):
        category = unicodedata.category(char)
        invisible = char.isspace() or category == "Cf"
        entry = {"codepoint": f"U+{ord(char):04X}", "category": category,
                 "occurrences": total[char], "by_field": {k: v[char] for k,v in frequencies.items()},
                 "legacy_maps_to_blank": invisible,
                 "formatting_distinguished_from_space": not invisible or char == " ",
                 "combining_mark_requires_layout_review": category.startswith("M")}
        if invisible:
            entry["status"] = "legacy_blank"
        elif ord(char) not in coverage:
            entry["status"] = "unsupported"
        else:
            mask, measured = raster(font, char)
            entry.update(measured)
            entry["status"] = "clipped" if measured["clipped"] else "empty_visible" if not mask.any() else "visible_in_cell"
            masks[char] = mask
            if mask.any():
                collisions[measured["raster_sha256"]].append(entry["codepoint"])
        glyphs[char] = entry
        if number % 64 == 0:
            check(started)
    unique_counts = Counter(e["status"] for e in glyphs.values())
    occurrence_counts = Counter()
    for entry in glyphs.values():
        occurrence_counts[entry["status"]] += entry["occurrences"]
    field_reasons = {field: Counter() for field in FIELDS}
    pair_reasons = Counter()
    partitions = defaultdict(Counter)
    cap_candidates = {str(cap): Counter() for cap in (128,513,640)}
    for i,row in enumerate(records(source)):
        reasons = set()
        for field in FIELDS:
            field_bad = set()
            for char in set(row["raw"][field]):
                entry = glyphs[char]
                if entry["status"] in {"unsupported", "clipped", "empty_visible"}:
                    field_bad.add(entry["status"])
                if not entry["formatting_distinguished_from_space"]:
                    field_bad.add("formatting_collapsed_to_blank")
                if entry["combining_mark_requires_layout_review"]:
                    field_bad.add("combining_mark_layout_unreviewed")
                if entry.get("touches_cell_frame"):
                    field_bad.add("cell_frame_touched")
            field_reasons[field].update(field_bad)
            reasons.update(field_bad)
        pair_reasons.update(reasons)
        eligible = row["structurally_eligible_not_qualified"] and not reasons
        partitions[row["prospective_partition"]]["conservative_inventory_candidates"] += eligible
        partitions[row["prospective_partition"]]["all_rows"] += 1
        for cap,tally in cap_candidates.items():
            if eligible and max(len(row["raw"]["chosen"]),len(row["raw"]["rejected"])) <= int(cap):
                tally[row["prospective_partition"]] += 1
        if i % 128 == 0:
            check(started)
    collision_rows = [values for values in collisions.values() if len(values)>1]
    detail = {"glyphs": {e["codepoint"]: e for e in glyphs.values()},
              "identical_nonblank_raster_groups": collision_rows}
    detail_path = output/"glyph_inventory.json"
    detail_path.write_text(json.dumps(detail,ensure_ascii=False,indent=2)+"\n")
    # Deterministic source-only examples; no neural output is displayed.
    picks = []
    for status in ("clipped", "empty_visible", "visible_in_cell", "unsupported"):
        picks.extend([c for c in sorted(glyphs) if glyphs[c]["status"]==status][:6])
    for char in "中文言语語":
        if char in glyphs and char not in picks:
            picks.append(char)
    width = 1200
    panel = Image.new("RGB", (width, 95+330*((len(picks)+4)//5)), "white")
    pen = ImageDraw.Draw(panel)
    label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
    pen.text((20,12),"SOURCE FONT AUDIT ONLY - no generated model answers",font=label,fill="black")
    for i,char in enumerate(picks):
        entry = glyphs[char];x=20+(i%5)*236;y=60+(i//5)*330
        pen.text((x,y),entry["codepoint"]+" "+entry["status"],font=label,fill="black")
        if char in masks:
            cell = Image.fromarray(np.where(masks[char],0,255).astype(np.uint8)).resize((224,224),Image.Resampling.NEAREST).convert("RGB")
            panel.paste(cell,(x,y+32));pen.rectangle((x,y+32,x+223,y+255),outline="red",width=1)
        else:
            pen.text((x,y+85),"No supported glyph",font=label,fill="red")
        pen.text((x,y+270),f"occurrences: {entry['occurrences']}",font=label,fill="black")
    panel_path=output/"source_font_samples.png";panel.save(panel_path)
    check(started)
    for path, expected in registration["input_sha256"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("Registered input changed during audit: " + path)
    if sha(FONT) != registration["font_sha256"]:
        raise ValueError("Shared font changed during audit")
    report = {"schema":"chinese-preference-source-raster-audit-v1",
        "registration_sha256":sha(REGISTRATION),"script_sha256":sha(Path(__file__)),
        "source_rows":n,"font":{"sha256":sha(FONT),"face":2,"name":list(font.getname()),"pixels":26,"cell":[32,32],"baseline_y":26,"pillow_version":PIL_VERSION,"layout_engine":str(font.layout_engine)},
        "unique_codepoints":len(glyphs),"unique_status_counts":dict(unique_counts),
        "occurrence_status_counts":dict(occurrence_counts),"field_row_reasons":{k:dict(v) for k,v in field_reasons.items()},
        "pair_row_reasons_overlapping":dict(pair_reasons),"prospective_partition_inventory":{k:dict(v) for k,v in partitions.items()},
        "inventory_candidates_with_both_answers_within_codepoint_cap":{k:dict(v) for k,v in cap_candidates.items()},
        "identical_nonblank_raster_groups":len(collision_rows),"glyph_inventory_sha256":sha(detail_path),"source_samples_sha256":sha(panel_path),
        "full_source_strings_unchanged":True,"all_registered_inputs_unchanged":True,
        "source_rows_rendered_as_complete_images":0,"grapheme_or_contextual_shaping_qualified":False,
        "semantic_labels_qualified":False,"model_calls":0,"optimizer_updates":0,"preference_fit_admitted":False,
        "elapsed_seconds":time.monotonic()-started,"peak_process_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "ceilings":{"seconds":120,"peak_rss_gib":1,"entry_available_host_gib":48,"runtime_available_host_gib":24},
        "candidate_count_is_not_rendering_or_language_qualification":True,"goal_achieved":False}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.output),ensure_ascii=False))

#!/usr/bin/env python3
"""Prepare image-only historical reading and stage selection from attested forms."""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.continuous_glyph_codec_data import historic_character_partition
from scripts.infer_compact_visual_span import read_image, save_grid
from scripts.prepare_compact_visual_span_evaluation import write_case
from scripts.prepare_visual_sequence_corpus import FONT, Rasterizer


STAGES = {"oracle": "甲骨文", "bronze": "金文", "seal": "篆文"}
READINGS_PER_STAGE = 6
PAIRS_PER_STAGE_COMBINATION = 3


def load(path):
    return json.loads(path.read_text())


def rank(text):
    return hashlib.sha256(("historical-visual-evaluation-v1\0" + text).encode()).hexdigest()


def text_ink(renderer, text):
    packed = b"".join(renderer.tile(char) for char in text)
    values = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big").reshape(-1, 1, 32, 32)
    return torch.from_numpy(values)


def load_sources(splits, cache):
    report_path = splits / "report.json"
    report = load(report_path)
    verified = load(splits / "independent_verification.json")
    if (verified["report_sha256"] != sha(report_path)
            or any(verified["cross_partition_keys"].values())
            or sha(cache) != report["raster_cache"]["cache_sha256"]):
        raise ValueError("Expected the authenticated historical split and original raster cache")
    partitions = {}
    for split, partition in report["partitions"].items():
        path = splits / f"{split}.json"
        if sha(path) != partition["file_sha256"]:
            raise ValueError("Historical source partition changed")
        partitions[split] = load(path)
    prior_train = {glyph["raster_sha256"] for rows in partitions.values() for row in rows
                   if historic_character_partition(row["char"]) == "train" for glyph in row["glyphs"]}
    payload = torch.load(cache, weights_only=True, map_location="cpu", mmap=True)
    pixels = payload["pixels"]
    if (pixels.shape != (report["raster_cache"]["records"], 1, 32, 32)
            or pixels.dtype != torch.uint8):
        raise ValueError("Historical raster tensor differs from its source receipt")
    return partitions["sealed"], prior_train, pixels


def candidates(rows, prior_train, renderer):
    # A shared exact shape with multiple recorded modern answers is ambiguous
    # for the single-image reading task. Restrict those cases before sampling.
    modern_forms = defaultdict(set)
    for row in rows:
        modern = row["fields"].get("traditional", "")
        if len(modern) == 1:
            for glyph in row["glyphs"]:
                modern_forms[glyph["raster_sha256"]].add(modern)
    selected = []
    for row in sorted(rows, key=lambda item: rank(item["family_sha256"] + item["entry_id"])):
        modern = row["fields"].get("traditional", "")
        if len(modern) != 1:
            continue
        try:
            renderer.tile(modern)
        except ValueError:
            continue
        staged = {}
        glyphs = sorted(row["glyphs"], key=lambda glyph: rank(glyph["label"] + glyph["sha256"]))
        for glyph in glyphs:
            stage, digest = glyph["stage"], glyph["raster_sha256"]
            if stage in STAGES and digest not in prior_train and modern_forms[digest] == {modern}:
                staged.setdefault(stage, glyph)
        if staged:
            selected.append((row, modern, staged))
    return selected


def choose(records, required_stages, count, used):
    chosen = []
    for row, modern, staged in records:
        family = row["family_sha256"]
        if family in used or not all(stage in staged for stage in required_stages):
            continue
        if len({staged[stage]["raster_sha256"] for stage in required_stages}) != len(required_stages):
            continue
        chosen.append((row, modern, [staged[stage] for stage in required_stages]))
        used.add(family)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Only {len(chosen)} eligible families for {required_stages}")


def write_source(output, legacy, pixels, row, glyph):
    source = (legacy / glyph["source_local_path"]).resolve()
    if not source.is_relative_to(legacy.resolve()) or sha(source) != glyph["sha256"]:
        raise ValueError("Historical SVG differs from its registered source")
    white = pixels[glyph["raster_index"]:glyph["raster_index"] + 1]
    actual = hashlib.sha256(np.packbits(white.numpy(), bitorder="big").tobytes()).hexdigest()
    if len(white) != 1 or not bool((white == 0).any()) or actual != glyph["raster_sha256"]:
        raise ValueError("A historical source raster changed or is empty")
    ink = 1 - white
    path = output / f"{glyph['raster_sha256']}.png"
    if not path.is_file():
        save_grid(ink, path)
    return ink, {"entry_id": row["entry_id"], "query_character": row["char"], "stage": glyph["stage"],
                 "label": glyph["label"], "svg_sha256": glyph["sha256"], "raster_sha256": actual,
                 "file": path.name, "image_sha256": sha(path)}


def prepare(splits, cache, legacy, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA during historical image preparation")
    torch.set_num_threads(2)
    rows, prior_train, pixels = load_sources(splits, cache)
    renderer = Rasterizer()
    available = candidates(rows, prior_train, renderer)
    reading_families, selection_families = set(), set()
    readings = [example for stage in STAGES for example in
                choose(available, (stage,), READINGS_PER_STAGE, reading_families)]
    selections = [example for pair in combinations(STAGES, 2) for example in
                  choose(available, pair, PAIRS_PER_STAGE_COMBINATION, selection_families)]
    output.mkdir(parents=True, exist_ok=False)
    inputs, answers = output / "inputs", output / "targets"
    inputs.mkdir()
    answers.mkdir()
    cases, references = [], []
    source_files = {}
    prefix = text_ink(renderer, "字形：")
    suffix = text_ink(renderer, "\n問題：這個古文字對應哪個繁體字？只輸出一字。\n回答：")
    for index, (row, modern, (glyph,)) in enumerate(readings, 1):
        key = f"read_{index:02d}"
        ink, source = write_source(answers, legacy, pixels, row, glyph)
        source_files[source["raster_sha256"]] = source
        for condition, image in (("normal", ink), ("glyph_blank", torch.zeros_like(ink))):
            cases.append(write_case(inputs, f"{key}_{condition}", condition, torch.cat([prefix, image, suffix])))
        target = answers / f"{key}.png"
        save_grid(text_ink(renderer, modern), target)
        references.append({"id": key, "kind": "reading", "source": source, "family": row["family_sha256"],
                           "modern": modern, "target_file": target.name, "target_image_sha256": sha(target)})
    for index, (row, _, glyphs) in enumerate(selections, 1):
        choices = [write_source(answers, legacy, pixels, row, glyph) for glyph in glyphs]
        choices.sort(key=lambda pair: rank(pair[1]["raster_sha256"]))
        blocks = [text_ink(renderer, "字形資料：")]
        for ink, source in choices:
            blocks.extend((text_ink(renderer, STAGES[source["stage"]] + "："), ink, text_ink(renderer, "\n")))
            source_files[source["raster_sha256"]] = source
        blocks.append(text_ink(renderer, "問題："))
        ending = text_ink(renderer, "\n回答：")
        longest_stage_name = max(len(STAGES[source["stage"]]) for _, source in choices)
        controls = []
        for arm, (_, target) in zip(("a", "b"), choices):
            key = f"select_{index:02d}_{arm}"
            stage = STAGES[target["stage"]]
            padding = " " * (longest_stage_name - len(stage))
            question = text_ink(renderer, f"請輸出{stage}的字形。只輸出一個字形。{padding}")
            for condition, content in (("normal", question), ("question_blank", torch.zeros_like(question))):
                case = write_case(inputs, f"{key}_{condition}", condition, torch.cat([*blocks, content, ending]))
                cases.append(case)
                if condition == "question_blank":
                    controls.append(case["image_sha256"])
            references.append({"id": key, "kind": "selection", "source": target, "family": row["family_sha256"],
                               "pair": index, "stage_choices": [value for _, value in choices],
                               "target_file": target["file"], "target_image_sha256": target["image_sha256"]})
        if len(set(controls)) != 1:
            raise ValueError("A blanked question pair must expose identical pixels")
    manifest = inputs / "manifest.json"
    save_json(manifest, {"inputs": cases})
    save_json(output / "references.json", {"cases": references, "unique_historical_rasters": list(source_files.values())})
    receipt = {"manifest_sha256": sha(manifest), "references_sha256": sha(output / "references.json"),
               "historical_split_report_sha256": sha(splits / "report.json"), "raster_cache_sha256": sha(cache),
               "renderer_font_sha256": sha(FONT), "normal_reading_by_stage": dict(Counter(r[2][0]["stage"] for r in readings)),
               "paired_stage_selection_contexts": len(selections), "unique_historical_rasters": len(source_files),
               "source_families": len(reading_families | selection_families), "normal_inputs": len(references),
               "paired_question_blank_controls_bit_identical": len(selections),
               "all_input_roundtrips_bit_exact": len(cases), "exact_old_historical_training_raster_overlap": 0,
               "model_calls": 0, "source_sha256": sha(Path(__file__))}
    save_json(output / "preparation.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=ROOT / "artifacts/historical_glyph_splits_v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/cache/v34_historic_glyph_rasters_32.pt")
    parser.add_argument("--legacy", type=Path, default=ROOT.parent / "incoder")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/historical_visual_evaluation_v1")
    args = parser.parse_args()
    prepare(args.splits, args.cache, args.legacy, args.output)

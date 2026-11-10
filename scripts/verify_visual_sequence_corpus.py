#!/usr/bin/env python3
"""Verify prepared glyph sequences and inspect actual training inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus


def verify(directory):
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    identities = {split: set() for split in ("train", "development")}
    totals = Counter()
    previews = {}
    for shard in report["shards"]:
        split = shard["split"]
        for name, digest in shard["files"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"Corpus shard changed: {name}")
        paths = {name.rsplit("_", 1)[-1]: directory / name for name in shard["files"]}
        sources = json.loads(paths["sources.json"].read_text())
        with np.load(paths["pixels.npz"], allow_pickle=False) as visual, np.load(
            paths["teacher.npz"], allow_pickle=False
        ) as teacher:
            ink = visual["ink"]
            glyphs = visual["record_glyph_offsets"]
            spans = visual["record_span_offsets"]
            units = teacher["record_unit_offsets"]
            glyph_ends, unit_ends = visual["span_glyph_ends"], teacher["span_unit_ends"]
            prompt_spans, prompt_glyphs = visual["prompt_spans"], visual["prompt_glyphs"]
            valid = (
                len(sources) == shard["records"] == len(glyphs) - 1 == len(spans) - 1 == len(units) - 1
                and ink.shape == (shard["glyphs"], 128) and ink.dtype == np.uint8
                and glyphs[-1] == len(ink)
                and spans[-1] == len(glyph_ends) == len(unit_ends) == shard["spans"]
                and units[-1] == len(teacher["unit_ids"]) == shard["native_units"]
                and np.all(np.diff(glyph_ends) > 0) and np.all(np.diff(unit_ends) > 0)
                and np.array_equal(glyph_ends[spans[1:] - 1], glyphs[1:])
                and np.array_equal(unit_ends[spans[1:] - 1], units[1:])
            )
            if not valid:
                raise ValueError(f"Misaligned complete sequences in {paths['pixels.npz'].name}")
            for index, source in enumerate(sources):
                first, last = glyphs[index:index + 2]
                prefix = prompt_glyphs[index]
                prefix_spans = prompt_spans[index]
                boundary = glyph_ends[spans[index] + prefix_spans - 1] - first if prefix_spans else 0
                if prefix != boundary or not 0 <= prefix < last - first:
                    raise ValueError("Prompt boundary crosses a visual span")
                physical_input = ink[first:first + prefix if prefix else last]
                identities[split].update({
                    ("source_group", source["kind"], source["group"]),
                    ("source_text", source["source_text_sha256"]),
                    ("physical_input", hashlib.sha256(physical_input.tobytes()).hexdigest()),
                })
                if split == "train" and last - first < 420:
                    previews.setdefault(source["kind"], totals[split + "/records"] + index)
        for key in ("records", "glyphs", "spans", "native_units"):
            totals[f"{split}/{key}"] += shard[key]
    if identities["train"] & identities["development"]:
        raise ValueError("A source group or exact input crosses the corpus partitions")

    images = []
    for split in identities:
        dataset = VisualSequenceCorpus(directory, split=split, expected_report_sha256=sha(report_path))
        if len(dataset) != totals[split + "/records"]:
            raise ValueError("Training loader length differs from the verified corpus")
        if split != "train":
            continue
        for kind, index in previews.items():
            sample = dataset[index]
            teacher = dataset.offline_teacher_alignment(index)
            if len(sample["span_glyph_ends"]) != len(teacher["span_unit_ends"]):
                raise ValueError("Training sample and teacher spans differ")
            tiles = sample["ink"][:sample["prompt_glyphs"] or len(sample["ink"]), 0].numpy()
            columns = 24
            panel = Image.new("L", (columns * 32, 30 + 32 * ((len(tiles) + columns - 1) // columns)), 255)
            ImageDraw.Draw(panel).text((5, 8), f"Training input / {kind} / record {index}", fill=0)
            for i, tile in enumerate(tiles):
                panel.paste(Image.fromarray(255 * (1 - tile)), (i % columns * 32, 30 + i // columns * 32))
            path = directory / f"preview_{kind}.png"
            panel.save(path)
            images.append({"kind": kind, "index": index, "file": path.name, "sha256": sha(path)})
    receipt = {"report_sha256": sha(report_path), "verified_shards": len(report["shards"]),
               "totals": dict(totals), "cross_partition_exact_input_or_source_group_overlap": 0,
               "training_input_previews": images}
    save_json(directory / "verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visual_sequence_corpus_v1")
    verify(parser.parse_args().data)

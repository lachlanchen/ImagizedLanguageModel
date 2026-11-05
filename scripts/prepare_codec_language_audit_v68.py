#!/usr/bin/env python3
"""Prepare source-exact autonomous language questions and pixel-only controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer

ROOT = Path(__file__).resolve().parents[1]
DEV_SHA = "14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369"
PAIR_SHA = "3bcb2e0534290ed9cf3dd1a4e378bbaa5a7eb4903dce0b4b0befb8f3ce58029d"
SOURCES = ("scripts/prepare_codec_language_audit_v68.py",
    "references/codec_language_v68_audit_preparation.md",
    "references/codec_language_learning_v68_protocol.md",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/full_text_renderer.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def white_question(renderer, text, prefix, question):
    layout, metadata = renderer._layout(text)
    before = layout.get_pixel_extents()
    start = len(prefix.encode("utf-8"))
    end = start + len(question.encode("utf-8"))
    attributes = renderer.Pango.AttrList()
    foreground = renderer.Pango.attr_foreground_new(65535, 65535, 65535)
    foreground.start_index, foreground.end_index = start, end
    attributes.insert(foreground)
    layout.set_attributes(attributes)
    after = layout.get_pixel_extents()
    shape = lambda rectangles: [(r.x, r.y, r.width, r.height) for r in rectangles]
    if shape(before) != shape(after):
        raise ValueError("question-color intervention changed physical layout")
    surface = renderer.cairo.ImageSurface(renderer.cairo.FORMAT_RGB24, metadata["width_px"], 32)
    context = renderer.cairo.Context(surface)
    context.set_source_rgb(1, 1, 1)
    context.paint()
    context.set_source_rgb(0, 0, 0)
    context.move_to(metadata["draw_x"], metadata["draw_y"])
    renderer.PangoCairo.show_layout(context, layout)
    surface.flush()
    pixels = np.frombuffer(surface.get_data(), np.uint8).reshape(32, surface.get_stride() // 4, 4)
    result = np.where(pixels[:, :metadata["width_px"], :3].mean(-1) >= 128, 255, 0).astype(np.uint8)
    return result, dict(question_utf8_start=start, question_utf8_end=end,
        intervention="white foreground attribute; unchanged full layout", layout_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/general_instructions/codec_language_audit_v68_20261105")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only preparation required")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    begun = time.perf_counter()
    sources = {name:sha(ROOT / name) for name in SOURCES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, digest in sources.items():
        if hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)).hexdigest() != digest:
            raise ValueError("commit the audit preparation before execution")
    cache = CompleteWritingImageData(ROOT / "data/general_instructions/complete_lines_development_20261105",
        purpose="development", expected_report_sha256=DEV_SHA)
    source_rows = {}
    for kind, path in cache.report["source_paths"].items():
        if sha(ROOT / path) != cache.report["source_sha256"][kind]:
            raise ValueError("original development source changed")
        source_rows[kind] = {r["id"]:r for r in map(json.loads, (ROOT / path).read_text().splitlines())}
    pairs_file = ROOT / "data/visual_qa/cmrc2018_short/counterfactual_pairs.jsonl"
    if sha(pairs_file) != PAIR_SHA:
        raise ValueError("original paired-question source changed")
    paired = [json.loads(x) for x in pairs_file.read_text().splitlines()]
    if len(paired) != 393:
        raise ValueError("original pair count differs")
    groups = {}
    for row in paired:
        ids = sorted(row["record_ids"])
        if not all("cmrc/" + i + ".pt" in cache._allowed for i in ids):
            continue
        a, b = (source_rows["cmrc"][i] for i in ids)
        if not {r["text"] for r in a["references"]}.isdisjoint({r["text"] for r in b["references"]}):
            continue
        if a["context_hash"] != b["context_hash"] or a["article_title"] != b["article_title"]:
            raise ValueError("paired article/context identity differs")
        groups.setdefault(a["article_title"], []).append(ids)
    articles = sorted(groups)
    generator = np.random.default_rng(20266830)
    chosen_articles = sorted(generator.choice(articles, 64, replace=False).tolist())
    chosen_pairs = [sorted(groups[article])[0] for article in chosen_articles]
    selected = [("cmrc", i) for pair in chosen_pairs for i in pair]
    categories = {}
    for identity, record in source_rows["dolly"].items():
        if "dolly/" + identity + ".pt" in cache._allowed:
            categories.setdefault(record["category"], []).append(identity)
    if len(categories) != 8:
        raise ValueError("all eight original English categories required")
    for category in sorted(categories):
        identities = sorted(categories[category])
        selected += [("dolly", i) for i in sorted(generator.choice(identities, 8, replace=False).tolist())]
    if len(selected) != 192 or len(set(selected)) != 192:
        raise ValueError("all fixed unique questions required")
    renderer = CompleteLineRenderer()
    if renderer.provenance != cache.report["renderer"]:
        raise ValueError("original complete-line layout required")
    cases, excluded = [], []
    for kind, identity in selected:
        row = source_rows[kind][identity]
        text = row["prompt"] if kind == "cmrc" else row["prompt_flat"]
        original, metadata = renderer.render(text, maximum_blocks=240)
        name = kind + "/" + identity + ".pt"
        strip, boundary = cache.get(name)
        expected = ((strip[0, 0, :, :boundary * 64] + 1) * 127.5).numpy().astype(np.uint8)
        target = ((strip[0, 0, :, boundary * 64:-128] + 1) * 127.5).numpy().astype(np.uint8)
        if not np.array_equal(original, expected):
            raise ValueError("original source prompt differs from actual cached pixels")
        if kind == "cmrc":
            prefix = "资料：" + row["context"] + "问题："
            if text != prefix + row["question"] + "回答：":
                raise ValueError("actual Chinese prompt construction differs")
            blank, control = white_question(renderer, text, prefix, row["question"])
            references = [r["text"] for r in row["references"]]
        else:
            blank = np.full_like(original, 255)
            control = dict(intervention="same-shape all-white English prompt", layout_unchanged=True)
            references = [row["answer_flat"]]
        if blank.shape != original.shape or np.any(blank < original) or np.array_equal(blank, original):
            raise ValueError("blanking must remove ink without reflow or added ink")
        images = []
        for index, reference in enumerate(references):
            try:
                pixels, _ = renderer.render(reference, maximum_blocks=128)
            except (ValueError, OverflowError) as error:
                if index == 0:
                    raise
                excluded.append(dict(id=identity, alternative=index, reason=str(error)))
                continue
            if index == 0 and not np.array_equal(pixels, target):
                raise ValueError("original first reference differs from cached pixels")
            images.append(torch.from_numpy(pixels.copy()))
        relative = Path("cases") / (kind + "_" + identity + ".pt")
        path = args.out / relative
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(prompt=torch.from_numpy(original.copy()), blank_prompt=torch.from_numpy(blank.copy()),
            references=images), path)
        cases.append(dict(source=kind, id=identity, file=str(relative), sha256=sha(path),
            original_cache_file=name, original_cache_sha256=cache.report["files_sha256"][name],
            article=row.get("article_title"), category=row.get("category", "chinese_reading"),
            control=control, prompt_shape=list(original.shape), reference_shapes=[list(x.shape) for x in images]))
    report = dict(source_commit=revision, sources=sources, development_report_sha256=DEV_SHA,
        original_source_sha256=cache.report["source_sha256"], paired_source_sha256=PAIR_SHA,
        eligible_chinese_articles=len(articles), chosen_pairs=chosen_pairs, cases=cases,
        excluded_alternative_references=excluded, renderer=renderer.provenance, seed=20266830,
        examples=len(cases), chinese_questions=128, english_questions=64,
        original_prompts_and_first_references_exact=True, sealed_examples=0, model_calls=0,
        training_only=False, evaluation_only=True, development_reused=True,
        language_quality_claim=False, elapsed_seconds=time.perf_counter() - begun)
    if sources != {name:sha(ROOT / name) for name in SOURCES}:
        raise ValueError("preparation implementation changed while running")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(examples=192, chinese_pairs=64, english_categories=len(categories),
        excluded_alternatives=len(excluded), report_sha256=sha(args.out / "report.json"),
        elapsed_seconds=report["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()

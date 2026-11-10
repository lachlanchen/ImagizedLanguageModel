#!/usr/bin/env python3
"""Prepare complete visual passages and QA with separate offline teacher alignment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from ilm.image_edit.baseline import ROOT, save_json, sha


BOOKS = ROOT / "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
CMRC = ROOT / "data/raw/cmrc2018"
NATURAL = ROOT / "data/visual_qa/natural_first_images_v1"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
NAMESPACE = "visual-sequence-corpus-v1"


def load(path):
    return json.loads(path.read_text())


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_hash(text):
    return text_hash("".join(char for char in text if not char.isspace()))


def documents():
    books = [json.loads(line) for line in BOOKS.read_text().splitlines()]
    titles = {row["source_title"] for row in books}
    held_titles = set(sorted(titles, key=lambda title: text_hash(NAMESPACE + "\0" + title))[:2])
    for row in books:
        split = "development" if row["source_title"] in held_titles else "train"
        yield split, "book", row["id"], row["source_title"], "", row["text"]

    natural_report = load(NATURAL / "report.json")
    if sha(NATURAL / "records.json") != natural_report["records_sha256"]:
        raise ValueError("Existing natural held contexts changed")
    held_contexts = {row["context_hash"] for row in load(NATURAL / "records.json")
                     if row["split"] == "held_context"}
    official = load(CMRC / "cmrc2018_dev.json")["data"]
    dev_titles = {article["title"] for article in official}
    dev_contexts = {context_hash(p["context"]) for article in official for p in article["paragraphs"]}
    for article in load(CMRC / "cmrc2018_train.json")["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            key = context_hash(context)
            if article["title"] in dev_titles or key in dev_contexts:
                continue
            split = "development" if key in held_contexts else "train"
            for question in paragraph["qas"]:
                answer = next((a["text"] for a in question["answers"] if a["text"].strip()
                               and context[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]), "")
                prompt = f"資料：{context}\n問題：{question['question']}\n回答："
                yield split, "qa", question["id"], key, prompt, answer


def aligned_spans(text, tokenizer, byte_decoder):
    """Collect native units until their joint bytes end at a character boundary."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    pending = bytearray()
    glyph_ends = []
    teacher_ends = []
    count = 0
    for index, token in enumerate(encoded.tokens, start=1):
        try:
            pending.extend(byte_decoder[char] for char in token)
        except KeyError as error:
            raise ValueError("source token has no byte representation") from error
        try:
            value = pending.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data":
                continue
            raise ValueError("source sequence has invalid UTF-8") from error
        if not text.startswith(value, count) or not value:
            raise ValueError("teacher spans do not reconstruct source text")
        count += len(value)
        glyph_ends.append(count)
        teacher_ends.append(index)
        pending.clear()
    if pending or count != len(text):
        raise ValueError("teacher spans do not cover the complete source text")
    return encoded.ids, glyph_ends, teacher_ends


class Rasterizer:
    def __init__(self):
        self.font = ImageFont.truetype(str(FONT), 26, index=2)
        with TTFont(FONT, fontNumber=2) as face:
            self.coverage = set(face.getBestCmap())
        self.cache = {}

    def tile(self, char):
        if char not in self.cache:
            canvas = Image.new("L", (32, 32), 255)
            if not char.isspace() and unicodedata.category(char) != "Cf":
                if ord(char) not in self.coverage:
                    raise ValueError(f"font lacks U+{ord(char):04X}")
                draw = ImageDraw.Draw(canvas)
                x = (32 - draw.textlength(char, font=self.font)) / 2
                draw.text((x, 26), char, fill=0, font=self.font, anchor="ls")
                if canvas.getextrema()[0] >= 128:
                    raise ValueError(f"empty glyph U+{ord(char):04X}")
            self.cache[char] = np.packbits(np.asarray(canvas) < 128, bitorder="big").tobytes()
        return self.cache[char]


def sequence(prompt, answer, tokenizer, byte_decoder, renderer):
    if not answer:
        raise ValueError("no valid complete target")
    prompt_ids, prompt_glyphs, prompt_units = aligned_spans(prompt, tokenizer, byte_decoder)
    target_ids, target_glyphs, target_units = aligned_spans(answer, tokenizer, byte_decoder)
    text = prompt + answer
    pixels = b"".join(renderer.tile(char) for char in text)
    return {
        "pixels": pixels,
        "glyph_ends": prompt_glyphs + [len(prompt) + end for end in target_glyphs],
        "unit_ends": prompt_units + [len(prompt_ids) + end for end in target_units],
        "units": prompt_ids + target_ids,
        "prompt_glyphs": len(prompt),
        "prompt_spans": len(prompt_glyphs),
        "glyphs": len(text),
    }


def write_shard(output, split, number, records):
    name = f"{split}_{number:04d}"
    pixels = []
    glyph_bounds = [0]
    span_bounds = [0]
    unit_bounds = [0]
    span_glyph_ends = []
    span_unit_ends = []
    teacher_units = []
    metadata = []
    for origin, row in records:
        pixels.append(row["pixels"])
        span_glyph_ends.extend(glyph_bounds[-1] + end for end in row["glyph_ends"])
        span_unit_ends.extend(unit_bounds[-1] + end for end in row["unit_ends"])
        teacher_units.extend(row["units"])
        glyph_bounds.append(glyph_bounds[-1] + row["glyphs"])
        span_bounds.append(len(span_glyph_ends))
        unit_bounds.append(len(teacher_units))
        metadata.append(origin)
    visual = output / f"{name}_pixels.npz"
    teacher = output / f"{name}_teacher.npz"
    sources = output / f"{name}_sources.json"
    np.savez_compressed(visual, ink=np.frombuffer(b"".join(pixels), dtype=np.uint8).reshape(-1, 128),
                        record_glyph_offsets=np.asarray(glyph_bounds, dtype=np.int64),
                        record_span_offsets=np.asarray(span_bounds, dtype=np.int64),
                        span_glyph_ends=np.asarray(span_glyph_ends, dtype=np.int64),
                        prompt_glyphs=np.asarray([row["prompt_glyphs"] for _, row in records], dtype=np.int32),
                        prompt_spans=np.asarray([row["prompt_spans"] for _, row in records], dtype=np.int32))
    np.savez_compressed(teacher, unit_ids=np.asarray(teacher_units, dtype=np.int32),
                        record_unit_offsets=np.asarray(unit_bounds, dtype=np.int64),
                        span_unit_ends=np.asarray(span_unit_ends, dtype=np.int64))
    save_json(sources, metadata)
    return {"split": split, "records": len(records), "glyphs": glyph_bounds[-1],
            "spans": span_bounds[-1], "native_units": unit_bounds[-1],
            "files": {path.name: sha(path) for path in (visual, teacher, sources)}}


def prepare(output):
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    from ilm.visual_lm.grounded_image_weights import MODEL_SHA

    if output.exists():
        raise FileExistsError(output)
    _, tokenizer, _ = archive_and_tokenizer()
    byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
    renderer = Rasterizer()
    output.mkdir(parents=True)
    pending = {split: [] for split in ("train", "development")}
    numbers = Counter()
    totals = Counter()
    rejected = []
    shards = []
    groups = set()
    for split, kind, identifier, group, prompt, answer in documents():
        try:
            row = sequence(prompt, answer, tokenizer, byte_decoder, renderer)
        except ValueError as error:
            rejected.append({"split": split, "kind": kind, "id": identifier, "reason": str(error)})
            continue
        pending[split].append(({"kind": kind, "id": identifier, "group": group,
                                "source_text_sha256": text_hash(prompt + answer)}, row))
        groups.add((split, kind, group))
        counts = {"records": 1, "glyphs": row["glyphs"], "spans": len(row["glyph_ends"]),
                  "native_units": len(row["units"]), "target_glyphs": len(answer),
                  "multi_unit_spans": int((np.diff([0, *row["unit_ends"]]) > 1).sum())}
        for key, count in counts.items():
            totals[f"{split}/{kind}/{key}"] += count
        if len(pending[split]) >= 128:
            shards.append(write_shard(output, split, numbers[split], pending[split]))
            numbers[split] += 1
            pending[split].clear()
    for split, rows in pending.items():
        if rows:
            shards.append(write_shard(output, split, numbers[split], rows))
    for split, kind, _ in groups:
        totals[f"{split}/{kind}/source_groups"] += 1
    save_json(output / "rejections.json", rejected)
    sources = [BOOKS, BOOKS.with_suffix(".jsonl.provenance.json"), CMRC / "cmrc2018_train.json",
               CMRC / "cmrc2018_dev.json", NATURAL / "records.json"]
    report = {"source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
              "source_teacher_archive_sha256": MODEL_SHA, "font_sha256": sha(FONT), "font_face": 2,
              "source_code_sha256": sha(Path(__file__)), "totals": dict(totals), "shards": shards,
              "rejected_records": len(rejected), "rejections_sha256": sha(output / "rejections.json"),
              "pixels": "32×32 binary ink; 128 bytes per glyph; high-bit first",
              "teacher_unit_ids_stored_separately_for_training": True,
              "official_development_answers_used": False, "teacher_model_forward_performed": False}
    save_json(output / "report.json", report)
    print(json.dumps({"totals": totals, "shards": len(shards), "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/visual_sequence_corpus_v1")
    prepare(parser.parse_args().output)

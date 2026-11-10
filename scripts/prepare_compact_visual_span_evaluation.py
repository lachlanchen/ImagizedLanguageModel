#!/usr/bin/env python3
"""Freeze paired visual QA and book continuations from held-out source groups."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.infer_compact_visual_span import read_image, save_grid
from scripts.prepare_visual_sequence_corpus import documents, text_hash


NAMESPACE = "compact-visual-span-evaluation-v1"
PAIR_COUNT = 16
BOOKS_PER_TITLE = 6


@dataclass(frozen=True)
class Record:
    index: int
    identifier: str
    group: str
    prompt: str
    answer: str


def order(text):
    return text_hash(NAMESPACE + "\0" + text)


def source_records(corpus):
    report = json.loads((corpus.directory / "report.json").read_text())
    verification = json.loads((corpus.directory / "verification.json").read_text())
    if (verification["report_sha256"] != corpus.report_sha256
            or verification["cross_partition_exact_input_or_source_group_overlap"] != 0):
        raise ValueError("Use the verified group-disjoint visual corpus")
    for name, digest in report["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Original corpus source changed: {name}")
    entries = {(kind, identifier): (group, prompt, answer)
               for split, kind, identifier, group, prompt, answer in documents() if split == "development"}
    grouped = {"qa": defaultdict(list), "book": defaultdict(list)}
    for shard_index, shard in enumerate(corpus.shards):
        name, = [name for name in shard["files"] if name.endswith("_sources.json")]
        path = corpus.directory / name
        if Path(name).name != name or sha(path) != shard["files"][name]:
            raise ValueError("Visual evaluation source index changed")
        rows = json.loads(path.read_text())
        if len(rows) != shard["records"]:
            raise ValueError("Held-out source index does not cover the visual shard")
        for offset, row in enumerate(rows):
            kind, identifier = row["kind"], row["id"]
            group, prompt, answer = entries[kind, identifier]
            if group != row["group"] or text_hash(prompt + answer) != row["source_text_sha256"]:
                raise ValueError("Source text does not match the authenticated raster")
            grouped[kind][group].append(Record(corpus.offsets[shard_index] + offset, identifier, group, prompt, answer))
    return grouped


def question_parts(record):
    passage, delimiter, tail = record.prompt.rpartition("\n問題：")
    if not delimiter or not passage.startswith("資料：") or not tail.endswith("\n回答："):
        raise ValueError("Unexpected visual QA template")
    return passage, tail.removesuffix("\n回答："), len(passage) + len(delimiter)


def normalized(text):
    return "".join(text.split())


def select_pairs(groups):
    pairs = []
    for group in sorted(groups, key=order):
        ordered = sorted(groups[group], key=lambda row: order(row.identifier))
        candidates = []
        for row in ordered:
            passage, question, _ = question_parts(row)
            answer = normalized(row.answer)
            if answer and answer not in normalized(question) and len(row.answer) <= 128:
                candidates.append((row, passage, question, answer))
        for left, right in combinations(candidates, 2):
            a, passage_a, question_a, answer_a = left
            b, passage_b, question_b, answer_b = right
            if (passage_a == passage_b and question_a != question_b
                    and answer_a not in answer_b and answer_b not in answer_a):
                pairs.append((a, b))
                break
        if len(pairs) == PAIR_COUNT:
            return pairs
    raise ValueError(f"Only {len(pairs)} eligible held-out QA pairs")


def book_cut(text):
    start = max(64, len(text) - 128)
    end = len(text) - 64
    boundaries = [cut for cut in range(start, end + 1) if text[cut - 1] in "。！？；"]
    if not boundaries:
        return None
    return min(boundaries, key=lambda cut: (abs(len(text) - cut - 96), cut))


def select_books(groups):
    if len(groups) != 2:
        raise ValueError("Expected the two whole works held out of training")
    chosen = []
    for title in sorted(groups, key=order):
        available = [(row, cut) for row in sorted(groups[title], key=lambda row: order(row.identifier))
                     if (cut := book_cut(row.answer)) is not None]
        if len(available) < BOOKS_PER_TITLE:
            raise ValueError(f"Insufficient continuation passages for {title}")
        chosen.extend(available[:BOOKS_PER_TITLE])
    return chosen


def write_case(directory, identifier, condition, tiles):
    name = identifier + ".png"
    path = directory / name
    save_grid(tiles, path)
    if not torch.equal(read_image(path, len(tiles)), tiles):
        raise ValueError("PNG conversion changed input ink or tile order")
    return {"id": identifier, "condition": condition, "image": name, "glyphs": len(tiles), "image_sha256": sha(path)}


def prepare(data, output):
    torch.set_num_threads(2)
    corpus = VisualSequenceCorpus(data, split="development")
    groups = source_records(corpus)
    pairs, books = select_pairs(groups["qa"]), select_books(groups["book"])
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    cases, references = [], []
    for index, pair in enumerate(pairs, 1):
        for arm, record in zip(("a", "b"), pair):
            key = f"qa_{index:02d}_{arm}"
            row = corpus[record.index]
            passage, question, question_start = question_parts(record)
            cut = row["prompt_glyphs"]
            if cut != len(record.prompt) or len(row["ink"]) - cut != len(record.answer):
                raise ValueError("QA raster boundaries differ from source")
            prompt = row["ink"][:cut]
            question_blank = prompt.clone()
            question_blank[question_start:question_start + len(question)] = 0
            passage_blank = prompt.clone()
            passage_blank[len("資料："):len(passage)] = 0
            for condition, tiles in (("normal", prompt), ("question_blank", question_blank), ("passage_blank", passage_blank)):
                cases.append(write_case(inputs, f"{key}_{condition}", condition, tiles))
            references.append({"id": key, "kind": "qa", "group": record.group, "source_id": record.identifier,
                               "paired_group": index, "question": question, "answer": record.answer})
    for index, (record, cut) in enumerate(books, 1):
        key = f"book_{index:02d}"
        row = corpus[record.index]
        if row["prompt_glyphs"] != 0 or len(row["ink"]) != len(record.answer):
            raise ValueError("Book raster does not match its source passage")
        prompt = row["ink"][:cut]
        cases.append(write_case(inputs, f"{key}_normal", "normal", prompt))
        cases.append(write_case(inputs, f"{key}_prefix_blank", "prefix_blank", torch.zeros_like(prompt)))
        references.append({"id": key, "kind": "book", "group": record.group, "source_id": record.identifier,
                           "prefix_glyphs": cut, "answer": record.answer[cut:]})
    manifest = inputs / "manifest.json"
    save_json(manifest, {"inputs": cases})
    save_json(output / "references.json", {"reference_rows": references})
    receipt = {
        "visual_corpus_report_sha256": corpus.report_sha256,
        "corpus_verification_sha256": sha(corpus.directory / "verification.json"),
        "input_manifest_sha256": sha(manifest), "references_sha256": sha(output / "references.json"),
        "paired_qa_groups": len(pairs), "qa_questions": 2 * len(pairs),
        "whole_held_out_books": sorted(groups["book"]), "book_continuations": len(books),
        "all_inputs": len(cases), "normal_inputs": sum(row["condition"] == "normal" for row in cases),
        "pixel_roundtrips_exact": len(cases), "model_calls": 0,
        "source_sha256": {name: sha(ROOT / name) for name in (
            "scripts/prepare_compact_visual_span_evaluation.py", "scripts/infer_compact_visual_span.py",
            "ilm/visual_lm/visual_sequence_corpus.py", "scripts/prepare_visual_sequence_corpus.py",
        )},
    }
    save_json(output / "preparation.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visual_sequence_corpus_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/compact_visual_span_evaluation_v1")
    args = parser.parse_args()
    prepare(args.data, args.output)

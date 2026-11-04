#!/usr/bin/env python3
"""Prepare provenance-pinned, answer-blind short CMRC paragraphs for visual QA."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import requests

from ilm.visual_lm.visual_cell_data import visual_cell_character_supported

REVISION = "c0eb1b6ba219847457e6af3180da722bbeb656af"
BASE = f"https://raw.githubusercontent.com/ymcui/cmrc2018/{REVISION}/"
SOURCES = {
    "cmrc2018_train.json": (
        "squad-style-data/cmrc2018_train.json",
        "cd56f7d38f6f4a6f92df2616479bbf134a37bebf",
    ),
    "cmrc2018_dev.json": (
        "squad-style-data/cmrc2018_dev.json",
        "94c1f2df14a669b0b7b836665d6339a34df8aa97",
    ),
    "LICENCE": ("LICENCE", "3b7b82d0da2db857eda1a798dbd908ea136f07b5"),
}
OUT = Path("data/visual_qa/cmrc2018_short")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def visible(value):
    return "".join(c for c in value if not c.isspace())


def main():
    if (OUT / "preparation_report.json").exists():
        raise FileExistsError("completed preparation is immutable")
    raw = Path("data/raw/cmrc2018")
    raw.mkdir(parents=True, exist_ok=True)
    source_receipts = {}
    for name, (remote, git_hash) in SOURCES.items():
        path = raw / name
        if not path.exists():
            response = requests.get(BASE + remote, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        payload = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if actual != git_hash:
            raise ValueError(f"raw source differs from pinned Git blob: {name}")
        source_receipts[name] = {
            "url": BASE + remote,
            "git_blob": git_hash,
            "sha256": digest(payload),
            "bytes": len(payload),
        }
    prepared = {}
    counts = {}
    for split, name in [
        ("train", "cmrc2018_train.json"),
        ("development", "cmrc2018_dev.json"),
    ]:
        source = json.loads((raw / name).read_text())
        rows = []
        reasons = Counter()
        for article in source["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                original = paragraph["context"]
                context = visible(original)
                context_hash = digest(context.encode())
                for qa in paragraph["qas"]:
                    reasons["source_records"] += 1
                    question = visible(qa["question"])
                    prompt = f"资料：{context}问题：{question}回答："
                    if not question or not context:
                        reasons["empty_input"] += 1
                        continue
                    if len(prompt) > 480:
                        reasons["prompt_over_480"] += 1
                        continue
                    if any(not visual_cell_character_supported(c) for c in prompt):
                        reasons["unsupported_input_glyph"] += 1
                        continue
                    answers = []
                    for ref in qa["answers"]:
                        original_answer = ref["text"]
                        start = ref["answer_start"]
                        answer = visible(original_answer)
                        if (
                            original[start : start + len(original_answer)]
                            != original_answer
                        ):
                            reasons["invalid_source_reference"] += 1
                            continue
                        mapped = len(visible(original[:start]))
                        if not answer or len(answer) > 16:
                            continue
                        if any(not visual_cell_character_supported(c) for c in answer):
                            continue
                        if answer in question:
                            continue
                        if context[mapped : mapped + len(answer)] != answer:
                            raise ValueError("whitespace offset remapping failed")
                        entry = {
                            "text": answer,
                            "start": mapped,
                            "end": mapped + len(answer),
                        }
                        if entry not in answers:
                            answers.append(entry)
                    if not answers:
                        reasons["no_eligible_reference"] += 1
                        continue
                    rows.append(
                        {
                            "id": qa["id"],
                            "split": split,
                            "article_title": title,
                            "context_hash": context_hash,
                            "context": context,
                            "question": question,
                            "prompt": prompt,
                            "references": answers,
                            "original_context": original,
                            "original_question": qa["question"],
                            "source_revision": REVISION,
                            "rights": "CC-BY-SA-4.0",
                            "source": "CMRC 2018, Cui et al. (EMNLP-IJCNLP 2019)",
                        }
                    )
        prepared[split] = rows
        counts[split] = dict(reasons)
    dev_contexts = {x["context_hash"] for x in prepared["development"]}
    dev_titles = {x["article_title"] for x in prepared["development"]}
    before = len(prepared["train"])
    prepared["train"] = [
        x
        for x in prepared["train"]
        if x["context_hash"] not in dev_contexts
        and x["article_title"] not in dev_titles
    ]
    counts["train"]["cross_split_article_or_context_removed"] = before - len(
        prepared["train"]
    )
    train_contexts = {x["context_hash"] for x in prepared["train"]}
    train_titles = {x["article_title"] for x in prepared["train"]}
    assert not train_contexts.intersection(
        dev_contexts
    ) and not train_titles.intersection(dev_titles)
    groups = defaultdict(list)
    for row in prepared["development"]:
        groups[row["context_hash"]].append(row)
    pairs = []
    for context_hash, group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x["id"])
        selected = None
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a["question"] == b["question"]:
                    continue
                if {r["text"] for r in a["references"]}.isdisjoint(
                    {r["text"] for r in b["references"]}
                ):
                    selected = {
                        "context_hash": context_hash,
                        "record_ids": [a["id"], b["id"]],
                    }
                    break
            if selected:
                break
        if selected:
            pairs.append(selected)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    summary = {}
    for split, rows in prepared.items():
        if len({x["id"] for x in rows}) != len(rows):
            raise ValueError("duplicate source question IDs")
        for row in rows:
            assert len(row["prompt"]) <= 480
            for ref in row["references"]:
                assert (
                    row["context"][ref["start"] : ref["end"]] == ref["text"]
                    and len(ref["text"]) <= 16
                    and ref["text"] not in row["question"]
                )
        path = OUT / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        files[path.name] = digest(path.read_bytes())
        lengths = sorted(len(x["prompt"]) for x in rows)
        summary[split] = {
            "records": len(rows),
            "articles": len({x["article_title"] for x in rows}),
            "paragraphs": len({x["context_hash"] for x in rows}),
            "prompt_cells_min": min(lengths),
            "prompt_cells_median": lengths[len(lengths) // 2],
            "prompt_cells_max": max(lengths),
            "maximum_reference_cells": max(
                len(r["text"]) for x in rows for r in x["references"]
            ),
        }
    path = OUT / "counterfactual_pairs.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs)
    )
    files[path.name] = digest(path.read_bytes())
    (OUT / "LICENCE").write_bytes((raw / "LICENCE").read_bytes())
    report = {
        "dataset": "CMRC2018 short full paragraphs",
        "source_revision": REVISION,
        "source_receipts": source_receipts,
        "license": "CC-BY-SA-4.0",
        "citation": "Cui et al. A Span-Extraction Dataset for Chinese Machine Reading Comprehension. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1600/",
        "filters": counts,
        "summary": summary,
        "counterfactual_paragraph_pairs": len(pairs),
        "files_sha256": files,
        "cross_split_title_overlap": 0,
        "cross_split_context_overlap": 0,
        "answer_aware_cropping": False,
        "test_labels_downloaded": False,
        "qa_training_performed": False,
        "preparation_source_sha256": digest(Path(__file__).read_bytes()),
        "plan_sha256": digest(
            Path("references/visual_reading_qa_data_plan.md").read_bytes()
        ),
    }
    (OUT / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

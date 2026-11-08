#!/usr/bin/env python3
"""Audit real paired-question source bindings and image geometry on CPU."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.data import display_text
from ilm.image_edit.paired_data import copy_cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pages", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prep = json.loads((args.pages / "preparation.json").read_text())
    manifest = args.pages / "pairs.jsonl"
    if sha(manifest) != prep["manifest_sha256"] or sha(args.source) != prep["source_sha256"]:
        raise ValueError("prepared manifest or source text changed")
    for name, digest in prep["implementation_sha256"].items():
        if sha(ROOT / name) != digest: raise ValueError("rendering source changed: " + name)
    originals = [json.loads(s) for s in args.source.read_text().splitlines() if s]
    source = {r["id"]: r for r in originals}
    if len(source) != len(originals): raise ValueError("duplicate source question IDs")
    copies = {r["id"]: r for r in copy_cases()}
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s]
    if len({r["id"] for r in rows}) != len(rows): raise ValueError("duplicate prepared IDs")
    pairs = defaultdict(list); audit = []
    for row in rows:
        if row["task"] == "qa":
            original = source[row["id"]]; context = display_text(original["context"])
            question = display_text(original["question"]); target = original["references"][0]["text"]
            prompt = f"资料：{context}\n问题：{question}\n回答："
            if (target != row["reference_text"] or target not in context or original["article_title"] != row["article"]
                    or hashlib.sha256(context.encode()).hexdigest() != row["pair"]):
                raise ValueError("question, passage or target binding differs")
            pairs[row["pair"]].append(row)
            item = dict(id=row["id"], split=row["split"], article=row["article"], pair=row["pair"],
                question_characters=len(question), passage_characters=len(context), answer_characters=len(target),
                answer_literal_in_question=target in question, accepted_answers=sorted({r["text"] for r in original["references"]}))
        elif row["task"] == "copy":
            prompt = copies[row["id"]]["prompt"]; target = copies[row["id"]]["reference"]
            if target != row["reference_text"]: raise ValueError("original copying target changed")
            item = dict(id=row["id"], split=row["split"])
        else: raise ValueError("unknown prepared task")
        for key, text, size in (("input", prompt, (1024, 1024)), ("target", target, (512, 256))):
            path = args.pages / row[key]; geometry = row[key + "_geometry"]
            if (sha(path) != row[key + "_sha256"] or hashlib.sha256(text.encode()).hexdigest() != geometry["text_sha256"]
                    or geometry["non_newline_characters"] != len(text.replace("\n", ""))):
                raise ValueError("actual page or source text differs: " + row["id"])
            with Image.open(path) as image:
                if image.size != size: raise ValueError("declared geometry changed")
                box = ImageChops.difference(image.convert("RGB"), Image.new("RGB", size, "white")).getbbox()
                if not box or not (0 < box[0] < box[2] < image.width and 0 < box[1] < box[3] < image.height):
                    raise ValueError("actual ink reaches the canvas edge")
                if key == "input": item["actual_ink_bbox"] = list(box)
        audit.append(item)
    paired = []
    for key, group in pairs.items():
        if len(group) != 2 or len({r["split"] for r in group}) != 1 or len({r["article"] for r in group}) != 1:
            raise ValueError("complete same-passage pair required")
        a, b = [source[r["id"]] for r in group]
        if a["question"] == b["question"]: raise ValueError("questions in a pair are identical")
        overlap = {r["text"] for r in a["references"]} & {r["text"] for r in b["references"]}
        paired.append(dict(pair=key, ids=[r["id"] for r in group], split=group[0]["split"], accepted_answers_overlap=sorted(overlap)))
    articles = {s: {r["article"] for r in rows if r["task"] == "qa" and r["split"] == s} for s in ("train", "validation")}
    if articles["train"] & articles["validation"]: raise ValueError("article leakage")
    summary = {}
    for split in ("train", "validation"):
        group = [r for r in audit if "pair" in r and r["split"] == split]
        if not group: raise ValueError("both question splits required")
        summary[split] = dict(questions=len(group), pairs=sum(r["split"] == split for r in paired), articles=len(articles[split]),
            literal_answer_in_question=[r["id"] for r in group if r["answer_literal_in_question"]],
            accepted_overlap_pairs=[r["ids"] for r in paired if r["split"] == split and r["accepted_answers_overlap"]],
            passage_character_range=[min(r["passage_characters"] for r in group), max(r["passage_characters"] for r in group)],
            answer_character_range=[min(r["answer_characters"] for r in group), max(r["answer_characters"] for r in group)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise ValueError("preserve earlier audit; select a new explicit output")
    save_json(args.output, dict(source_sha256=sha(args.source), manifest_sha256=sha(manifest),
        prepared_pairs=len(rows), verified_image_files=2 * len(rows), rows=audit, pairs=paired, summary=summary,
        task_split_counts={"/".join(k): v for k, v in Counter((r["task"], r["split"]) for r in rows).items()},
        all_image_and_text_bindings_verified=True, actual_ink_inside_canvas=True, article_disjoint=True,
        source_font=prep["font"], neural_calls=0, qa_quality_claim=False,
        limitations="Checks file/text bindings and edge geometry, not OCR legibility, internal glyph defects, native VAE information retention, model memory fit or semantic understanding.",
        native_large_page_memory_and_representation_unvalidated=True, implementation_sha256=sha(__file__)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

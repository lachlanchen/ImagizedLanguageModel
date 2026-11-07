#!/usr/bin/env python3
"""Prepare pinned, evaluation-only English passage pairs without model calls."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import save_json, sha
from ilm.image_edit.data import font_and_coverage, page
from score_native_visual_access import canonical

REVISION = "240e165ab706d95bd4323653bb92421f446208cf"
BLOB = "e9a3f913ad1468ebe105b891334ca7b0bc0e2510"
BYTES = 4854279
URL = f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{REVISION}/dataset/dev-v1.1.json"
SEED = 20269321


def acquire(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=30) as response:
            payload = response.read(BYTES + 1)
        if len(payload) != BYTES or hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != BLOB:
            raise ValueError("official source bytes differ")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if len(payload) != BYTES or hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != BLOB:
        raise ValueError("cached official source differs")
    data = json.loads(payload)
    if data["version"] != "1.1":
        raise ValueError("SQuAD 1.1 required")
    return data


def valid_questions(paragraph):
    context = paragraph["context"]
    questions = []
    for q in sorted(paragraph["qas"], key=lambda r: r["id"]):
        answers = []
        for answer in q["answers"]:
            start, text = answer["answer_start"], answer["text"]
            if not text or context[start:start + len(text)] != text:
                raise ValueError("official annotated answer offset differs")
            value = dict(text=text, start=start, end=start + len(text))
            if value not in answers:
                answers.append(value)
        if answers and q["question"].strip():
            questions.append(dict(id=q["id"], question=q["question"], references=answers))
    return questions


def distinct(a, b):
    if a["question"] == b["question"]:
        return False
    if {canonical(x["text"], "en") for x in a["references"]} & {canonical(x["text"], "en") for x in b["references"]}:
        return False
    return not any(max(x["start"], y["start"]) < min(x["end"], y["end"]) for x in a["references"] for y in b["references"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/raw/squad_v1.1/dev-v1.1.json"))
    p.add_argument("--training", type=Path, default=Path("artifacts/native_visual_alignment/prepared_v1/training.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = acquire(args.source)
    training = json.loads(args.training.read_text())
    candidates = []
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            digest = hashlib.sha256(paragraph["context"].encode()).hexdigest()
            order = hashlib.sha256(f"{SEED}:{article['title']}:{digest}".encode()).hexdigest()
            candidates.append((order, article["title"], digest, paragraph))
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ("inputs", "references", "erased"):
        (args.output / folder).mkdir()
    inputs, references, selected = [], [], []
    used_titles = set()
    rejections = Counter()
    font_provenance = None
    font_rows = []
    for _, title, digest, paragraph in sorted(candidates, key=lambda r: r[0]):
        if title in used_titles:
            continue
        context = paragraph["context"]
        if any(context in row["transcript"] for row in training):
            rejections["current_training_context_overlap"] += 1
            continue
        found = None
        for a, b in itertools.combinations(valid_questions(paragraph), 2):
            if not distinct(a, b):
                continue
            rows = [dict(id="squad/" + q["id"], question=q["question"], references=q["references"],
                prompt=f'Passage:\n{context}\nQuestion: {q["question"]}\nAnswer:', reference=q["references"][0]["text"]) for q in (a, b)]
            try:
                font, provenance = font_and_coverage(rows)
                pages = [page(row["prompt"], (1024, 1024), font) for row in rows]
                answers = [page(row["reference"], (512, 256), font) for row in rows]
                _, context_geometry = page("Passage:\n" + context, (1024, 1024), font)
            except ValueError as error:
                rejections[str(error)] += 1
                continue
            found = rows, pages, answers, provenance, context_geometry
            break
        if found is None:
            rejections["no_valid_fitting_distinct_pair"] += 1
            continue
        rows, pages, answers, provenance, context_geometry = found
        if font_provenance is not None and provenance["font_file_sha256"] != font_provenance["font_file_sha256"]:
            raise ValueError("font changed during preparation")
        font_provenance = provenance
        font_rows.extend(rows)
        index = len(selected)
        pair = "squad/" + digest
        latent_seed, readout_seed = 20269421 + index, 20269521 + index
        question_top = 32 + 48 * context_geometry["lines"]
        context_top = 80  # one fixed Passage: header line before the source paragraph
        erased_questions = []
        for j, (row, (image, geometry), (answer, answer_geometry)) in enumerate(zip(rows, pages, answers)):
            original = np.asarray(image).copy()
            question_erased = original.copy()
            question_erased[question_top:] = 255
            if not np.array_equal(question_erased[:question_top], original[:question_top]):
                raise ValueError("question erasure altered context pixels")
            erased_questions.append(question_erased)
            passage_erased = original.copy()
            passage_erased[context_top:question_top] = 255
            if not np.array_equal(passage_erased[:context_top], original[:context_top]) or not np.array_equal(passage_erased[question_top:], original[question_top:]):
                raise ValueError("passage erasure altered the question or outside pixels")
            normal_name = f"inputs/{index:02}_{j}.png"
            erased_name = f"erased/{index:02}_{j}_passage.png"
            reference_name = f"references/{index:02}_{j}.png"
            image.save(args.output / normal_name)
            Image.fromarray(passage_erased).save(args.output / erased_name)
            answer.save(args.output / reference_name)
            for condition, name, identifier in (("normal", normal_name, row["id"]), ("passage_erased", erased_name, "passage_erased/" + row["id"])):
                inputs.append(dict(id=identifier, source_id=row["id"], pair=pair, condition=condition,
                    input=name, input_sha256=sha(args.output / name), seed=latent_seed, readout_seed=readout_seed))
            references.append(dict(id=row["id"], pair=pair, language="en", article=title,
                references=row["references"], primary=row["reference"], reference_image=reference_name,
                reference_sha256=sha(args.output / reference_name), input_geometry=geometry,
                answer_geometry=answer_geometry, prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest()))
        if not np.array_equal(*erased_questions):
            raise ValueError("paired question-erased pages must be identical")
        name = f"erased/{index:02}_question.png"
        Image.fromarray(erased_questions[0]).save(args.output / name)
        inputs.append(dict(id="question_erased/" + pair, pair=pair, condition="question_erased", input=name,
            input_sha256=sha(args.output / name), seed=latent_seed, readout_seed=readout_seed))
        selected.append(dict(pair=pair, article=title, ids=[r["id"] for r in rows], context_sha256=digest,
            question_band_top=question_top, passage_band_top=context_top))
        used_titles.add(title)
        if len(selected) == 16:
            break
    if len(selected) != 16 or len(inputs) != 80 or len(references) != 32:
        raise ValueError("complete sixteen-article English cohort unavailable")
    _, global_provenance = font_and_coverage(font_rows)
    for key in ("font_file_sha256", "font_face_index", "font_size"):
        if global_provenance[key] != font_provenance[key]:
            raise ValueError("whole-cohort font differs")
    font_provenance = global_provenance
    save_json(args.output / "inputs.json", inputs)
    save_json(args.output / "references.json", references)
    files = {str(path.relative_to(args.output)): sha(path) for path in sorted(args.output.rglob("*.png"))}
    save_json(args.output / "preparation.json", dict(cases=80, normal=32, question_erased=16, passage_erased=32,
        articles=16, pairs=16, selected=selected, source_revision=REVISION, source_git_blob=BLOB,
        source_url=URL, source_sha256=sha(args.source), source_bytes=BYTES,
        training_labels_sha256=sha(args.training), exact_current_training_context_overlap=0,
        inputs_sha256=sha(args.output / "inputs.json"), references_sha256=sha(args.output / "references.json"),
        files_sha256=files, font=font_provenance, rejection_counts=dict(rejections), seed=SEED,
        model_calls=0, optimizer_updates=0, training_data_modified=False,
        all_source_characters_retained=True, answer_aware_cropping=False,
        all_erasure_outside_pixels_identical=True, paired_question_erased_pages_identical=True,
        foundation_pretraining_contamination_ruled_out=False, official_benchmark_replication=False,
        implementation_sha256=sha(__file__), copyright="SQuAD 1.1, Rajpurkar et al., EMNLP 2016; CC-BY-SA-4.0.",
        general_ilm_goal_complete=False))
    print("Prepared 32 natural English questions and 48 erasure controls from 16 articles; no model calls.")


if __name__ == "__main__":
    main()

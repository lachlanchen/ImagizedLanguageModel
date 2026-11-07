#!/usr/bin/env python3
"""Score complete English transfer and two erasures against all alternatives."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import canonical
from score_native_visual_alignment import edit_distance

INITIALIZER = "e2937d2fa986fac0e7ca1964607d5dca06194b698b6a25ad0644635f585427fb"


def read(path):
    return json.loads(Path(path).read_text())


def judge(output, reference):
    actual = canonical(output["text"], "en")
    alternatives = {canonical(r["text"], "en") for r in reference["references"]}
    return dict(text=output["text"], stopped=output["stopped"],
        canonical_exact=actual in alternatives, exact_stopped=actual in alternatives and output["stopped"])


def score(inputs, references, outputs):
    if len(inputs) != 80 or [r["id"] for r in inputs] != [r["id"] for r in outputs]:
        raise ValueError("all eighty calls in the fixed input order required")
    refs = {r["id"]: r for r in references}
    results = {}
    groups = {}
    for item, output in zip(inputs, outputs):
        for key in ("id", "pair", "condition", "input_sha256", "seed", "readout_seed"):
            if item[key] != output[key]:
                raise ValueError("input image, condition or matched seed differs")
        results[item["id"]] = output
        groups.setdefault(item["pair"], []).append(item)
    cases, pairs = [], []
    for pair, items in groups.items():
        normal = [r for r in items if r["condition"] == "normal"]
        passage = {r["source_id"]: r for r in items if r["condition"] == "passage_erased"}
        erased = [r for r in items if r["condition"] == "question_erased"]
        if len(normal) != 2 or len(passage) != 2 or len(erased) != 1:
            raise ValueError("complete five-call passage pair required")
        blank = results[erased[0]["id"]]
        selected = []
        for item in normal:
            ref = refs[item["id"]]
            answer = results[item["id"]]
            context_missing = results[passage[item["id"]]["id"]]
            row = dict(id=item["id"], pair=pair, article=ref["article"], alternatives=[r["text"] for r in ref["references"]],
                normal=judge(answer, ref), passage_erased=judge(context_missing, ref), question_erased=judge(blank, ref),
                changed_after_question_erasure=canonical(answer["text"], "en") != canonical(blank["text"], "en"))
            cases.append(row)
            selected.append(row)
        pairs.append(dict(pair=pair, both_correct=all(r["normal"]["exact_stopped"] for r in selected),
            normal_outputs_differ=canonical(selected[0]["normal"]["text"], "en") != canonical(selected[1]["normal"]["text"], "en"),
            shared_question_erased_matches_either=any(r["question_erased"]["exact_stopped"] for r in selected)))
    if len(cases) != 32 or len(pairs) != 16:
        raise ValueError("complete sixteen-pair English cohort required")
    summary = dict(normal_cases=32, normal_exact_stopped=sum(r["normal"]["exact_stopped"] for r in cases),
        passage_erased_cases=32, passage_erased_exact_stopped=sum(r["passage_erased"]["exact_stopped"] for r in cases),
        question_erased_independent_calls=16, both_correct_pairs=sum(r["both_correct"] for r in pairs),
        normal_outputs_differ_pairs=sum(r["normal_outputs_differ"] for r in pairs),
        normal_answers_changed_after_question_erasure=sum(r["changed_after_question_erasure"] for r in cases),
        shared_question_erased_matches_either=sum(r["shared_question_erased_matches_either"] for r in pairs),
        correct_normal_retained_without_passage=sum(r["normal"]["exact_stopped"] and r["passage_erased"]["exact_stopped"] for r in cases),
        correct_normal_lost_without_passage=sum(r["normal"]["exact_stopped"] and not r["passage_erased"]["exact_stopped"] for r in cases))
    return dict(summary=summary, cases=cases, pairs=pairs)


def calibrate(args, preparation, inputs, references):
    if sha(args.source) != preparation["source_sha256"]:
        raise ValueError("official source changed")
    source = read(args.source)
    lookup = {"squad/" + q["id"]: (paragraph["context"], q["question"])
              for article in source["data"] for paragraph in article["paragraphs"] for q in paragraph["qas"]}
    normal = {r["id"]: r for r in inputs if r["condition"] == "normal"}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    for ref in references:
        context, question = lookup[ref["id"]]
        prompt = f"Passage:\n{context}\nQuestion: {question}\nAnswer:"
        if hashlib.sha256(prompt.encode()).hexdigest() != ref["prompt_sha256"]:
            raise ValueError("original complete source text differs")
        observations = {}
        for kind, path, expected in (("input", args.prepared / normal[ref["id"]]["input"], prompt),
                                      ("reference", args.prepared / ref["reference_image"], ref["primary"])):
            result = subprocess.run(["tesseract", str(path), "stdout", "-l", "eng", "--psm", "6"],
                env=dict(os.environ, OMP_THREAD_LIMIT="1"), capture_output=True, text=True, check=True)
            actual, target = canonical(result.stdout, "en"), canonical(expected, "en")
            observations[kind] = dict(text=result.stdout, expected=expected, source_sha256=sha(path),
                canonical_exact=actual == target, edit_errors=edit_distance(actual, target), reference_characters=len(target))
        records.append(dict(id=ref["id"], **observations))
        save_json(args.output / "records.json", records)
    result = dict(cases=32, input_ocr_exact=sum(r["input"]["canonical_exact"] for r in records),
        reference_ocr_exact=sum(r["reference"]["canonical_exact"] for r in records),
        input_character_edit_error=sum(r["input"]["edit_errors"] for r in records) / sum(r["input"]["reference_characters"] for r in records),
        elapsed_seconds=time.monotonic() - started, inputs_sha256=preparation["inputs_sha256"],
        prepared_references_sha256=preparation["references_sha256"], records_sha256=sha(args.output / "records.json"),
        implementation_sha256=sha(__file__), tesseract_version=subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0],
        cpu_threads_per_ocr=1, gpu_used=False, student_model_calls=0, optimizer_updates=0,
        cases_filtered_using_ocr=False, student_conditioning_uses_ocr=False, general_ilm_goal_complete=False)
    save_json(args.output / "report.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("calibrate", "complete"), default="complete")
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_english_transfer/prepared_v2"))
    p.add_argument("--before", type=Path, default=Path("artifacts/native_english_transfer/before_v1"))
    p.add_argument("--after", type=Path, default=Path("artifacts/native_english_transfer/after_v1"))
    p.add_argument("--alignment", type=Path, default=Path("artifacts/native_visual_alignment"))
    p.add_argument("--source", type=Path, default=Path("data/raw/squad_v1.1/dev-v1.1.json"))
    p.add_argument("--calibration", type=Path, default=Path("artifacts/native_english_transfer/calibration_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    preparation = read(args.prepared / "preparation.json")
    inputs = read(args.prepared / "inputs.json")
    refs = read(args.prepared / "references.json")
    if sha(args.prepared / "inputs.json") != preparation["inputs_sha256"] or sha(args.prepared / "references.json") != preparation["references_sha256"]:
        raise ValueError("prospective cohort changed")
    for name, digest in preparation["files_sha256"].items():
        if sha(args.prepared / name) != digest:
            raise ValueError("source raster changed")
    if args.stage == "calibrate":
        return calibrate(args, preparation, inputs, refs)
    learning = read(args.alignment / "learning_v1/report.json")
    if learning["updates"] != 2560:
        raise ValueError("finish the registered alignment experiment first")
    if read(args.alignment / "controller_status.json")["state"] != "completed":
        raise ValueError("finish all registered assessments before the transfer follow-up")
    digest = sha(args.alignment / "learning_v1/checkpoint.pt")
    if digest != learning["checkpoint_sha256"]:
        raise ValueError("completed alignment checkpoint changed")
    scores = {}
    for stage, folder, checkpoint in (("before", args.before, INITIALIZER), ("after", args.after, digest)):
        report = read(folder / "report.json")
        if report["cases"] != 80 or report["mode"] != "answer" or report["cap"] != 96 or report["teacher_answers_loaded"]:
            raise ValueError("complete autonomous transfer condition required")
        if report["inputs_sha256"] != preparation["inputs_sha256"] or report["checkpoint_sha256"] != checkpoint or report["implementation"] != learning["metadata"]["implementation"]:
            raise ValueError("model/code/input bindings differ")
        scores[stage] = score(inputs, refs, read(folder / "outputs.json"))
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != sha(target):
            raise ValueError("evidence copy differs")
        files[name] = sha(target)

    for name in (*preparation["files_sha256"], "preparation.json", "inputs.json", "references.json"):
        retain(args.prepared / name, name)
    for stage, folder in (("before", args.before), ("after", args.after)):
        for name in ("outputs.json", "report.json"):
            retain(folder / name, "reports/" + stage + "_" + name)
    calibration = None
    if (args.calibration / "report.json").exists():
        calibration = read(args.calibration / "report.json")
        if calibration["inputs_sha256"] != preparation["inputs_sha256"] or calibration["records_sha256"] != sha(args.calibration / "records.json"):
            raise ValueError("source OCR calibration differs")
        for name in ("report.json", "records.json"):
            retain(args.calibration / name, "reports/source_calibration_" + name)
    save_json(args.output / "result.json", dict(models=scores, files_sha256=files, scorer_sha256=sha(__file__),
        calls_per_model=80, total_calls=160, reference_alternatives_preserved=True, source_calibration=calibration,
        canonicalization="NFC and collapsed English whitespace; case, punctuation, word boundaries and quotes preserved; exact stopped requires native EOS.",
        foundation_pretraining_contamination_ruled_out=False, official_benchmark_replication=False,
        matched_optimizer_continuation_arm_executed=False, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, language_quality_qualified=False, general_ilm_goal_complete=False,
        copyright=preparation["copyright"]))
    print(json.dumps({key: value["summary"] for key, value in scores.items()}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preserve complete visual-alignment assessments, without repairing output."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import canonical, score_cohort, score_controls

INITIALIZER = "e2937d2fa986fac0e7ca1964607d5dca06194b698b6a25ad0644635f585427fb"


def read(path):
    return json.loads(Path(path).read_text())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def transcriptions(inputs, refs, outputs):
    if len(inputs) != 8 or [r["id"] for r in inputs] != [r["id"] for r in refs] or [r["id"] for r in inputs] != [r["id"] for r in outputs]:
        raise ValueError("complete fixed eight-page transcription cohort required")
    cases = []
    for item, ref, output in zip(inputs, refs, outputs):
        if output["input_sha256"] != item["input_sha256"]:
            raise ValueError("transcription source image differs")
        actual = canonical(output["text"], ref["language"])
        expected = canonical(ref["expected"], ref["language"])
        errors = edit_distance(actual, expected)
        cases.append(dict(id=ref["id"], task=ref["task"], language=ref["language"],
            expected=ref["expected"], text=output["text"], stopped=output["stopped"],
            canonical_exact=actual == expected, exact_stopped=actual == expected and output["stopped"],
            edit_errors=errors, reference_characters=len(expected),
            character_edit_error=errors / len(expected) if expected else None))
    groups = {}
    for task in ("copy", "qa"):
        selected = [r for r in cases if r["task"] == task]
        groups[task] = dict(cases=len(selected), exact_stopped=sum(r["exact_stopped"] for r in selected),
            stopped=sum(r["stopped"] for r in selected),
            aggregate_character_edit_error=sum(r["edit_errors"] for r in selected) / sum(r["reference_characters"] for r in selected))
    return dict(cases=cases, groups=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("before", "complete"), required=True)
    p.add_argument("--run", type=Path, default=Path("artifacts/native_visual_alignment"))
    p.add_argument("--previous", type=Path, default=Path("artifacts/native_visual_access"))
    p.add_argument("--previous-evidence", type=Path, default=Path("references/evidence/native_visual_access_result/result.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepared = args.run / "prepared_v1"
    preparation = read(prepared / "preparation.json")
    prior = read(args.previous_evidence)
    for name, digest in prior["files_sha256"].items():
        if sha(args.previous_evidence.parent / name) != digest:
            raise ValueError("retained predecessor evidence changed")
    inputs = read(prepared / "transcription_screen_inputs.json")
    refs = read(prepared / "transcription_screen_references.json")
    if sha(prepared / "transcription_screen_inputs.json") != preparation["screen_inputs_sha256"] or sha(prepared / "transcription_screen_references.json") != preparation["screen_references_sha256"]:
        raise ValueError("prepared transcription inputs/references changed")
    before = read(args.run / "transcription_before_v1/report.json")
    if before["checkpoint_sha256"] != INITIALIZER or before["inputs_sha256"] != preparation["screen_inputs_sha256"] or before["cap"] != 1024 or before["mode"] != "transcribe":
        raise ValueError("matched before-transcription control required")
    cohorts = dict(transcription_before=transcriptions(inputs, refs, read(args.run / "transcription_before_v1/outputs.json")))
    sources = [(args.run / "transcription_before_v1" / name, "reports/transcription_before_" + name) for name in ("outputs.json", "report.json")]
    sources += [(prepared / "preparation.json", "reports/preparation.json"), (prepared / "transcription_screen_references.json", "reports/transcription_references.json")]
    training = None
    if args.stage == "complete":
        training = read(args.run / "learning_v1/report.json")
        updates = [json.loads(s) for s in (args.run / "learning_v1/updates.jsonl").read_text().splitlines()]
        schedule = read(args.run / "learning_v1/schedule.json")
        if training["updates"] != 2560 or training["metadata"]["initializer_sha256"] != INITIALIZER or len(updates) != 2560:
            raise ValueError("complete declared alignment run required")
        if training["metadata"]["preparation_sha256"] != sha(prepared / "preparation.json") or training["metadata"]["training_sha256"] != sha(prepared / "training.json"):
            raise ValueError("declared prepared training data changed")
        if training["metadata"]["schedule_sha256"] != sha(args.run / "learning_v1/schedule.json"):
            raise ValueError("saved optimizer schedule changed")
        if [{k: r[k] for k in ("update", "task", "id")} for r in updates] != schedule or [r["update"] for r in updates] != list(range(1, 2561)):
            raise ValueError("complete optimizer/schedule observation differs")
        if any(not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"]) or not all(v["finite"] for v in r["gradients"].values()) for r in updates):
            raise ValueError("nonfinite optimizer observation")
        if Counter(r["task"] for r in updates) != {"dense": 1536, "qa": 512, "copy": 512}:
            raise ValueError("curriculum totals differ")
        if set(r["id"] for r in updates) != set(training["metadata"]["training_ids"]) or len(set(r["id"] for r in updates)) != 384:
            raise ValueError("training IDs incomplete or different")
        declared = read(prepared / "training.json")
        for task, passes in (("dense", 4), ("qa", 2), ("copy", 4)):
            counts = Counter(r["id"] for r in updates if r["task"] == task)
            expected_ids = {r["id"] for r in declared if task == "dense" or r["task"] == task}
            if set(counts) != expected_ids or set(counts.values()) != {passes}:
                raise ValueError("complete-pass curriculum coverage differs")
        digest = sha(args.run / "learning_v1/checkpoint.pt")
        if digest != training["checkpoint_sha256"]:
            raise ValueError("completed numeric checkpoint changed")
        for stage in ("transcription_after_v1", "full_validation_v1", "question_readout_v1"):
            report = read(args.run / stage / "report.json")
            if report["checkpoint_sha256"] != digest or report["implementation"] != training["metadata"]["implementation"]:
                raise ValueError("assessment model/code binding differs")
            for name in ("outputs.json", "report.json"):
                sources.append((args.run / stage / name, "reports/" + stage + "_" + name))
        after = read(args.run / "transcription_after_v1/report.json")
        if after["inputs_sha256"] != preparation["screen_inputs_sha256"] or after["cap"] != 1024 or after["mode"] != "transcribe":
            raise ValueError("after-transcription condition differs")
        cohorts["transcription_after"] = transcriptions(inputs, refs, read(args.run / "transcription_after_v1/outputs.json"))
        full_inputs = args.previous / "full_validation_inputs.json"
        control_inputs = args.previous / "question_controls_v1/inputs.json"
        if read(args.run / "full_validation_v1/report.json")["inputs_sha256"] != sha(full_inputs) or read(args.run / "question_readout_v1/report.json")["inputs_sha256"] != sha(control_inputs):
            raise ValueError("immutable answer-assessment index changed")
        for stage, count in (("full_validation_v1", 64), ("question_readout_v1", 48)):
            report = read(args.run / stage / "report.json")
            if report["mode"] != "answer" or report["cap"] != 96 or report["cases"] != count or report["teacher_answers_loaded"]:
                raise ValueError("autonomous answer-assessment condition differs")
        cohorts["full_validation"] = score_cohort(read(full_inputs), prior["cohorts"]["full_validation"]["cases"], read(args.run / "full_validation_v1/outputs.json"))
        cohorts["paired_question_controls"] = score_controls(read(control_inputs), read(args.previous / "question_controls_v1/references.json"), read(args.run / "question_readout_v1/outputs.json"))
        for name in ("report.json", "updates.jsonl", "schedule.json"):
            sources.append((args.run / "learning_v1" / name, "reports/training_" + name))
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for path, name in sources:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(target) != sha(path):
            raise ValueError("retained observation differs")
        files[name] = sha(target)
    save_json(args.output / "result.json", dict(stage=args.stage, cohorts=cohorts, training=training,
        files_sha256=files, scorer_sha256=sha(__file__),
        reused_input_raster_evidence="references/evidence/native_visual_access_result/validation/ and erased/; source pages are unchanged",
        input_raster_source_result_sha256=sha(args.previous_evidence),
        character_error="Levenshtein distance divided by reference characters after NFC/declared whitespace canonicalization; insertion errors can exceed one. Order-sensitive, without case/punctuation/quote repair.",
        matched_continuation_arm_executed=False, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, language_quality_qualified=False, general_ilm_goal_complete=False,
        copyright="CMRC 2018 passage text/images: CC-BY-SA-4.0, Cui et al., EMNLP-IJCNLP 2019; copy strings synthetic; native source/model NUS Show Lab, Apache-2.0."))
    print(json.dumps({key: value.get("groups", {k: v for k, v in value.items() if k not in ("cases", "pairs")}) for key, value in cohorts.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

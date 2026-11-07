#!/usr/bin/env python3
"""Preserve and score complete auxiliary reading observations without repair."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import unicodedata

from ilm.image_edit.baseline import save_json, sha


def read(path):
    return json.loads(Path(path).read_text())


def canonical(text, language):
    text = unicodedata.normalize("NFC", text)
    return "".join(text.split()) if language == "zh" else " ".join(text.split())


def compare(output, reference):
    prediction = canonical(output["text"], reference["language"])
    expected = canonical(reference["expected"], reference["language"])
    a, b = Counter(prediction), Counter(expected)
    overlap = sum((a & b).values())
    denominator = len(prediction) + len(expected)
    return dict(id=reference["id"], expected=reference["expected"], text=output["text"],
        stopped=output["stopped"], canonical_exact=prediction == expected,
        exact_stopped=prediction == expected and output["stopped"],
        character_multiset_f1=2 * overlap / denominator if denominator else 1.0,
        language=reference["language"])


def totals(rows):
    return dict(cases=len(rows), canonical_exact=sum(r["canonical_exact"] for r in rows),
        exact_stopped=sum(r["exact_stopped"] for r in rows), stopped=sum(r["stopped"] for r in rows),
        mean_character_multiset_f1=sum(r["character_multiset_f1"] for r in rows) / len(rows) if rows else None)


def score_cohort(inputs, references, outputs):
    if len(inputs) != len(references) or [r["id"] for r in inputs] != [r["id"] for r in outputs]:
        raise ValueError("complete fixed input/output order is required")
    if [r["id"] for r in inputs] != [r["id"] for r in references]:
        raise ValueError("reference order differs")
    results = []
    for item, ref, output in zip(inputs, references, outputs):
        if item["input_sha256"] != output["input_sha256"]:
            raise ValueError("readout input binding differs")
        result = compare(output, ref)
        result.update(task=ref["task"], split=ref["split"])
        results.append(result)
    groups = defaultdict(list)
    for result in results:
        groups[result["split"] + "/" + result["task"]].append(result)
        groups[result["split"] + "/" + result["task"] + "/" + result["language"]].append(result)
    return dict(cases=results, groups={name: totals(rows) for name, rows in groups.items()})


def score_controls(inputs, references, outputs):
    if len(inputs) != 48 or [r["id"] for r in inputs] != [r["id"] for r in outputs]:
        raise ValueError("complete fixed 48-call intervention order required")
    refs = {r["id"]: r for r in references}
    groups = defaultdict(list)
    for item, output in zip(inputs, outputs):
        if any(item[k] != output[k] for k in ("id", "pair", "condition", "seed", "readout_seed", "input_sha256")):
            raise ValueError("paired control input or seed differs")
        groups[item["pair"]].append(output)
    pairs, normal = [], []
    for pair, values in groups.items():
        clean = [r for r in values if r["condition"] == "normal"]
        erased = [r for r in values if r["condition"] == "erased"]
        if len(clean) != 2 or len(erased) != 1 or refs[clean[0]["id"]]["expected"] == refs[clean[1]["id"]]["expected"]:
            raise ValueError("complete distinct-answer pair required")
        scores = [compare(r, refs[r["id"]]) for r in clean]
        blank = [compare(erased[0], refs[r["id"]]) for r in clean]
        normal.extend(scores)
        changed = [canonical(r["text"], "zh") != canonical(erased[0]["text"], "zh") for r in clean]
        pairs.append(dict(pair=pair, normal=scores, shared_erased_text=erased[0]["text"],
            shared_erased_stopped=erased[0]["stopped"], both_questions_exact=all(r["exact_stopped"] for r in scores),
            normal_outputs_differ=canonical(clean[0]["text"], "zh") != canonical(clean[1]["text"], "zh"),
            normal_answers_changed_after_erasure=changed,
            erased_exact_against_either_reference=any(r["exact_stopped"] for r in blank),
            correct_normal_answers_unchanged_after_erasure=sum(s["exact_stopped"] and not c for s, c in zip(scores, changed))))
    if len(pairs) != 16:
        raise ValueError("all sixteen passage pairs required")
    return dict(pairs=pairs, normal=totals(normal), both_questions_exact=sum(r["both_questions_exact"] for r in pairs),
        normal_outputs_differ=sum(r["normal_outputs_differ"] for r in pairs),
        changed_normal_answers=sum(sum(r["normal_answers_changed_after_erasure"]) for r in pairs),
        erased_calls=16, erased_matches_either_reference=sum(r["erased_exact_against_either_reference"] for r in pairs),
        correct_normal_answers_unchanged_after_erasure=sum(r["correct_normal_answers_unchanged_after_erasure"] for r in pairs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("artifacts/native_visual_access"))
    p.add_argument("--diagnostic", type=Path, default=Path("artifacts/native_unified_probe/prepare_v1"))
    p.add_argument("--manifest", type=Path, default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--full", type=Path)
    p.add_argument("--controls", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    training = read(args.run / "learning_v1/report.json")
    updates_path = args.run / "learning_v1/updates.jsonl"
    updates = [json.loads(s) for s in updates_path.read_text().splitlines()]
    original = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    training_rows = [r for r in original if r["split"] == "train"]
    allowed = {r["id"] for r in training_rows}
    if training["updates"] != 4096 or [r["update"] for r in updates] != list(range(1, 4097)):
        raise ValueError("complete contiguous 4096-update observation required")
    if set(r["id"] for r in updates) != allowed or len(allowed) != 384:
        raise ValueError("training IDs incomplete or leaked")
    if any(not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"]) or
           not all(v["finite"] for v in r["gradients"].values()) for r in updates):
        raise ValueError("nonfinite training observation")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(path, name, digest=None):
        actual = sha(path)
        if digest is not None and actual != digest:
            raise ValueError("retained evidence changed")
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha(destination) != actual:
            raise ValueError("evidence copy differs")
        files[name] = actual
        return name

    screen_inputs = read(args.diagnostic / "inputs.json")
    screen_refs = read(args.diagnostic / "references.json")
    cohorts = {}
    checkpoint_sha = sha(args.run / "learning_v1/checkpoint.pt")
    for name in ("before_v1", "after_v1"):
        report = read(args.run / name / "report.json")
        if report["checkpoint_sha256"] != (checkpoint_sha if name == "after_v1" else None):
            raise ValueError("screen checkpoint binding differs")
        if report["inputs_sha256"] != sha(args.diagnostic / "inputs.json"):
            raise ValueError("screen input index binding differs")
        cohorts[name] = score_cohort(screen_inputs, screen_refs, read(args.run / name / "outputs.json"))
        for file in ("outputs.json", "report.json"):
            retain(args.run / name / file, f"reports/{name}_{file}")
    for file in ("report.json", "updates.jsonl"):
        retain(args.run / "learning_v1" / file, "reports/training_" + file)
    source_rows = {r["id"]: r for r in original}
    for i, (item, ref) in enumerate(zip(screen_inputs, screen_refs)):
        retain(args.diagnostic / item["input"], f"screen/{i:02}_input.png", item["input_sha256"])
        retain(args.diagnostic / ref["target"], f"screen/{i:02}_reference.png", ref["target_sha256"])
    if args.full:
        index_path = args.run / "full_validation_inputs.json"
        refs_path = args.run / "full_validation_references.json"
        inputs, refs = read(index_path), read(refs_path)
        held_out = {r["id"] for r in original if r["split"] == "validation"}
        if {r["id"] for r in inputs} != held_out:
            raise ValueError("full validation does not match the declared held-out IDs")
        refs = [dict(r, split=source_rows[r["id"]]["split"]) for r in refs]
        report = read(args.full / "report.json")
        if len(inputs) != 64 or report["inputs_sha256"] != sha(index_path) or report["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("complete full-cohort bindings differ")
        cohorts["full_validation"] = score_cohort(inputs, refs, read(args.full / "outputs.json"))
        for file in ("outputs.json", "report.json"):
            retain(args.full / file, "reports/full_validation_" + file)
        for i, (item, ref) in enumerate(zip(inputs, refs)):
            source = source_rows[item["id"]]
            retain(index_path.parent / item["input"], f"validation/{i:02}_input.png", item["input_sha256"])
            retain(args.manifest.parent / source["target"], f"validation/{i:02}_reference.png", source["target_sha256"])
    if args.controls:
        prepared = args.run / "question_controls_v1"
        inputs = read(prepared / "inputs.json")
        report = read(args.controls / "report.json")
        if report["inputs_sha256"] != sha(prepared / "inputs.json") or report["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("intervention bindings differ")
        cohorts["paired_question_controls"] = score_controls(inputs, read(prepared / "references.json"), read(args.controls / "outputs.json"))
        for file in ("outputs.json", "report.json"):
            retain(args.controls / file, "reports/question_controls_" + file)
        retain(prepared / "preparation.json", "reports/question_controls_preparation.json")
        for i, r in enumerate(x for x in inputs if x["condition"] == "erased"):
            retain(prepared / r["input"], f"erased/{i:02}.png", r["input_sha256"])
    result = dict(cohorts=cohorts, training=training, checkpoint_sha256=checkpoint_sha,
        complete_finite_updates=4096, unique_training_ids=384, validation_ids_optimized=0,
        files_sha256=files, exporter_sha256=sha(__file__),
        canonicalization="NFC; Chinese whitespace removed; English whitespace collapsed; case, punctuation, quotation marks and all written characters retained. Exact stopped additionally requires native EOS.",
        character_f1="Multiset character overlap after canonicalization; insensitive to ordering, and not a language-qualification metric.",
        repeated_screen_cases_not_pooled_with_full_validation=True,
        auxiliary_unicode_head=True, answer_image_generation_evaluated=False,
        copyright="CMRC 2018 passage images and reference answers: CC-BY-SA-4.0, Cui et al., EMNLP-IJCNLP 2019. Copy strings are synthetic; native Show-o2 model/source: NUS Show Lab, Apache-2.0.",
        language_quality_qualified=False, general_ilm_goal_complete=False)
    save_json(args.output / "result.json", result)
    print(json.dumps({name: value.get("groups", {k: v for k, v in value.items() if k != "pairs"}) for name, value in cohorts.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

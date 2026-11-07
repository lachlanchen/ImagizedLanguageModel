#!/usr/bin/env python3
"""Verify the complete bilingual state-alignment run and autonomous answers."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import compare, totals, score_controls
from score_native_english_transfer import score as score_english
from train_native_state_alignment import binding, BUDGET


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_state_alignment"))
    p.add_argument("--learning", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    learn = args.learning or args.root / "learning_v1"
    after_path = args.after or args.root / "after_v1"
    for directory in (learn, after_path, args.root / "cache_v1", args.root / "preflight_v1", args.root / "copy_before_v1"):
        if read(directory / "status.json")["state"] != "completed":
            raise ValueError("wait for terminal completed stage: " + str(directory))
    report = read(learn / "report.json")
    if report["updates"] != BUDGET or report["metadata"]["implementation"] != binding():
        raise ValueError("complete unchanged learner required")
    if sha(learn / "checkpoint.pt") != report["checkpoint_sha256"]:
        raise ValueError("final numeric model changed")
    preparation = read(args.root / "prepared_v1/preparation.json")
    cache = read(args.root / "cache_v1/manifest.json")
    if cache["implementation"] != binding() or sha(args.root / "prepared_v1/training.json") != preparation["training_sha256"] or cache["training_sha256"] != preparation["training_sha256"] or sha(args.root / "cache_v1/manifest.json") != report["metadata"]["cache_manifest_sha256"]:
        raise ValueError("training data/state cache changed")
    plan = read(learn / "schedule.json")
    journal = [json.loads(line) for line in (learn / "updates.jsonl").read_text().splitlines()]
    if len(journal) != BUDGET or [r["update"] for r in journal] != list(range(1, BUDGET + 1)):
        raise ValueError("full contiguous optimizer observation required")
    if [r["id"] for r in journal] != [r["id"] for r in plan] or sha(learn / "schedule.json") != report["metadata"]["schedule_sha256"]:
        raise ValueError("observed optimizer differs from fixed curriculum")
    counts = Counter(r["id"] for r in journal)
    if set(counts) != set(report["metadata"]["training_ids"]) or set(counts.values()) != {4} or len(counts) != 640:
        raise ValueError("four complete passes over all 640 IDs required")
    for row in journal:
        if not all(math.isfinite(row[k]) for k in ("loss", "answer_ce", "state_mse", "gradient_norm")) or len(row["gradients"]) != 317 or not all(v["finite"] for v in row["gradients"].values()):
            raise ValueError("optimizer/gradient record failed")
    preflight = read(args.root / "preflight_v1/report.json")
    if preflight["updates"] != 3 or preflight["disabled_source_prefix_error"] != 0 or not preflight["zero_correction_prefix_exact"] or preflight["causal_history_control"]["normalized_max_error"] > 2e-5:
        raise ValueError("real source/causal preflight required")
    source_proof = read(args.root / "source_replay_complete_v1/report.json")
    if source_proof["cases"] != 4 or source_proof["readout_calls"] != 8 or not source_proof["all_exact"] or source_proof["implementation"] != binding():
        raise ValueError("complete additional zero-correction source replay required")
    if read(args.root / "source_replay_complete_v1/status.json")["state"] != "completed":
        raise ValueError("wait for terminal additional source check")
    assessment = args.root / "assessment_v1"
    if source_proof["evaluator_sha256"] != sha("scripts/check_native_state_source.py") or source_proof["inputs_sha256"] != sha(assessment / "en_inputs.json"):
        raise ValueError("additional source evaluator or cohort changed")
    if len(source_proof["checks"]) != 4 or not all(r["source_prefix_bit_exact"] and r["source_mask_bit_exact"] and r["retained_source_replay_exact"] and r["ids"] == r["corrected_ids"] for r in source_proof["checks"]):
        raise ValueError("additional source case evidence incomplete")
    assessment_preparation = read(assessment / "preparation.json")
    if sha(assessment / "inputs.json") != assessment_preparation["inputs_sha256"]:
        raise ValueError("fixed assessment inventory changed")
    for cohort, digest in assessment_preparation["cohort_inputs_sha256"].items():
        if sha(assessment / (cohort + "_inputs.json")) != digest:
            raise ValueError("fixed cohort inventory changed")
    for name, digest in assessment_preparation["files_sha256"].items():
        if sha(assessment / name) != digest:
            raise ValueError("fixed assessment raster changed")
    inputs = read(assessment / "inputs.json")
    before_report = read(args.root / "copy_before_v1/report.json")
    after_report = read(after_path / "report.json")
    if before_report["cases"] != 32 or before_report["checkpoint_sha256"] is not None or before_report["inputs_sha256"] != sha(assessment / "copy_inputs.json"):
        raise ValueError("native source copy comparison changed")
    if after_report["cases"] != 160 or after_report["checkpoint_sha256"] != report["checkpoint_sha256"] or after_report["inputs_sha256"] != sha(assessment / "inputs.json"):
        raise ValueError("complete final autonomous assessment required")
    if before_report["implementation"] != binding() or after_report["teacher_states_loaded"] or after_report["original_transcripts_loaded"] or after_report["implementation"] != binding():
        raise ValueError("autonomous image input assessment changed")
    outputs = read(after_path / "outputs.json")
    if [r["id"] for r in inputs] != [r["id"] for r in outputs]:
        raise ValueError("all 160 responses in fixed order required")
    for row, out in zip(inputs, outputs):
        for key in ("input_sha256", "coarse_input_sha256", "seed", "readout_seed", "cohort"):
            if row[key] != out[key]:
                raise ValueError("input/view/seed differs")
    copy_refs = read(args.root / "prepared_v1/copy_references.json")
    if sha(args.root / "prepared_v1/copy_references.json") != preparation["copy_references_sha256"]:
        raise ValueError("fixed copy references changed")
    copy_before = read(args.root / "copy_before_v1/outputs.json")
    copy_inputs = read(assessment / "copy_inputs.json")
    if len(copy_before) != 32 or [r["id"] for r in copy_before] != [r["id"] for r in copy_inputs]:
        raise ValueError("source copy response order differs")
    for row, out in zip(copy_inputs, copy_before):
        if any(row[k] != out[k] for k in ("input_sha256", "coarse_input_sha256", "seed", "readout_seed")):
            raise ValueError("source copying pixels/view/seed differ")
    before_copies = [compare(o, r) for o, r in zip(copy_before, copy_refs)]
    after_copies = [compare(o, r) for o, r in zip([r for r in outputs if r["cohort"] == "copy"], copy_refs)]
    before, after = dict(copy=dict(summary=totals(before_copies), cases=before_copies)), dict(copy=dict(summary=totals(after_copies), cases=after_copies))
    for cohort, refs_path, source_path, scorer in (
            ("zh", Path("artifacts/native_chinese_interface/prepared_v1/references.json"), Path("references/evidence/native_chinese_interface_result/reports/outputs.json"), score_controls),
            ("en", Path("artifacts/native_english_transfer/prepared_v2/references.json"), Path("references/evidence/native_reader_geometry_result/reports/source_432_outputs.json"), score_english)):
        source_receipt = read(source_path.parent.parent / "result.json")
        if sha(source_path) != source_receipt["files_sha256"][str(source_path.relative_to(source_path.parent.parent))]:
            raise ValueError("retained native source responses changed")
        reference_export = Path("references/evidence") / ("native_chinese_interface_result" if cohort == "zh" else "native_english_transfer_result") / "result.json"
        if sha(refs_path) != read(reference_export)["files_sha256"]["references.json"]:
            raise ValueError("fixed annotated reference alternatives changed")
        rows = [r for r in inputs if r["cohort"] == cohort]
        native_rows = [dict(r, input_sha256=r["coarse_input_sha256"]) for r in rows]
        old = read(source_path)
        if cohort == "zh":
            old = [r for r in old if r["arm"] == "source_432"]
        before[cohort] = scorer(native_rows, read(refs_path), old)
        after[cohort] = scorer(rows, read(refs_path), [r for r in outputs if r["cohort"] == cohort])
    gates = dict(copying=after["copy"]["summary"]["exact_stopped"] >= 30,
        chinese_answers=after["zh"]["normal"]["exact_stopped"] >= 16,
        chinese_pairs=after["zh"]["both_questions_exact"] >= 4,
        chinese_gain=after["zh"]["normal"]["exact_stopped"] >= before["zh"]["normal"]["exact_stopped"] + 4,
        english_answers=after["en"]["summary"]["normal_exact_stopped"] >= 11,
        english_pairs=after["en"]["summary"]["both_correct_pairs"] >= 3)
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != sha(target):
            raise ValueError("retained evidence changed")
        files[name] = sha(target)

    for directory, prefix, names in ((learn, "reports/training_", ("report.json", "schedule.json")),
            (args.root / "source_replay_complete_v1", "reports/additional_source_", ("report.json",)),
            (args.root / "preflight_v1", "reports/preflight_", ("report.json", "updates.jsonl")),
            (args.root / "copy_before_v1", "reports/before_", ("report.json", "outputs.json")),
            (after_path, "reports/after_", ("report.json", "outputs.json")),
            (args.root / "prepared_v1", "data/training_", ("preparation.json", "english_selected.json", "copy_references.json")),
            (assessment, "data/assessment_", ("preparation.json", "inputs.json", "copy_inputs.json", "zh_inputs.json", "en_inputs.json"))):
        for name in names:
            retain(directory / name, prefix + name)
    compact = args.output / "reports/training_updates_compact.jsonl"
    with compact.open("w") as f:
        for row in journal:
            values = row["gradients"]
            r = {k: v for k, v in row.items() if k != "gradients"}
            r["gradient_summary"] = dict(total=len(values), finite=sum(v["finite"] for v in values.values()), nonzero=sum(v["nonzero"] for v in values.values()))
            f.write(json.dumps(r) + "\n")
    files[str(compact.relative_to(args.output))] = sha(compact)
    for i, row in enumerate([r for r in inputs if r["cohort"] == "copy"]):
        retain(assessment / row["input"], f"copy/{i:02d}_input.png")
    save_json(args.output / "result.json", dict(before=before, after=after, gates=gates,
        reader_prerequisites_passed=all(gates.values()), image_writing_stage_admitted=all(gates.values()),
        files_sha256=files, training_raw_journal_sha256=sha(learn / "updates.jsonl"),
        raw_gradient_journal_retained_locally=True, public_optimizer_journal_is_compact=True,
        implementation_sha256=sha(__file__), training=report, actual_updates=BUDGET,
        all_training_ids_seen_four_times=True, copied_native_baseline_calls=128, new_before_copy_calls=32, new_after_calls=160,
        auxiliary_unicode_head=True, image_answer_generation_demonstrated=False, general_ilm_goal_complete=False,
        copyright=preparation["copyright"]))
    print(json.dumps(dict(gates=gates, all_reader_prerequisites_passed=all(gates.values()),
        copy=after["copy"]["summary"], zh=after["zh"]["normal"], en=after["en"]["summary"]), indent=2))


if __name__ == "__main__":
    main()

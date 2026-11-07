#!/usr/bin/env python3
"""Score the complete attention learner against its exactly replayed source."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import compare, totals, score_controls
from score_native_english_transfer import score as score_english
from train_native_reader_attention import binding, SOURCE, BUDGET


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_reader_attention"))
    p.add_argument("--learning", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    learning = args.learning or args.root / "learning_v1"
    after = args.after or args.root / "after_v1"
    for folder in (learning, after, args.root / "cache_v1", args.root / "preflight_v1"):
        if read(folder / "status.json")["state"] != "completed":
            raise ValueError("wait for completed stage: " + str(folder))
    training = read(learning / "report.json")
    cache = read(args.root / "cache_v1/manifest.json")
    if training["updates"] != BUDGET or training["metadata"]["implementation"] != binding() or cache["implementation"] != binding():
        raise ValueError("completed unchanged learner required")
    if sha(learning / "checkpoint.pt") != training["checkpoint_sha256"] or training["metadata"]["cache_manifest_sha256"] != sha(args.root / "cache_v1/manifest.json") or cache["source_checkpoint_sha256"] != SOURCE:
        raise ValueError("final model/source/cache changed")
    source_preparation = read("artifacts/native_state_alignment/prepared_v1/preparation.json")
    if cache["training_sha256"] != source_preparation["training_sha256"] or sha("artifacts/native_state_alignment/prepared_v1/training.json") != cache["training_sha256"]:
        raise ValueError("bilingual training inventory changed")
    journal = [json.loads(line) for line in (learning / "updates.jsonl").read_text().splitlines()]
    plan = read(learning / "schedule.json")
    if len(journal) != BUDGET or [r["update"] for r in journal] != list(range(1, BUDGET + 1)) or [r["id"] for r in journal] != [r["id"] for r in plan] or sha(learning / "schedule.json") != training["metadata"]["schedule_sha256"]:
        raise ValueError("complete contiguous declared optimizer schedule required")
    counts = Counter(r["id"] for r in journal)
    if len(counts) != 640 or set(counts.values()) != {4} or set(counts) != set(training["metadata"]["training_ids"]):
        raise ValueError("four complete passes per training ID required")
    for row in journal:
        if not all(math.isfinite(row[k]) for k in ("loss", "gradient_norm")) or len(row["gradients"]) != 224 or not all(v["finite"] for v in row["gradients"].values()):
            raise ValueError("finite complete attention gradients required")
        i = row["update"] - 1
        rate = 5e-5 * (i + 1) / 64 if i < 64 else 5e-6 + .5 * 4.5e-5 * (1 + math.cos(math.pi * (i - 64) / (BUDGET - 64)))
        if abs(row["lr"] - rate) > 1e-12:
            raise ValueError("learning-rate schedule changed")
    preflight = read(args.root / "preflight_v1/report.json")
    if preflight["updates"] != 3 or preflight["metadata"]["implementation"] != binding() or preflight["disabled_source_logit_error"] != 0 or preflight["changed_initially_zero_tensors"] != 112 or preflight["causal_history_control"]["normalized_max_error"] > 2e-5:
        raise ValueError("real gradient/source/causal preflight required")
    if len(preflight["source_replay"]) != 4 or not all(r["exact"] and r["disabled_ids"] == r["enabled_ids"] for r in preflight["source_replay"]):
        raise ValueError("all four disabled/zero source replays required")
    prior_root = Path("references/evidence/native_canvas_control_result")
    source = read(prior_root / "result.json")
    if source["checkpoint_sha256"] != SOURCE or source["actual_calls"] != 480 or source["legacy_decision_replays_exact"] != 160:
        raise ValueError("completed canvas source calibration required")
    for name, digest in source["files_sha256"].items():
        if sha(prior_root / name) != digest:
            raise ValueError("source calibration evidence changed")
    inputs_path = Path("artifacts/native_canvas_control/prepared_v1/legacy_inputs.json")
    inputs = read(inputs_path)
    report = read(after / "report.json")
    if report["cases"] != 160 or report["checkpoint_sha256"] != training["checkpoint_sha256"] or report["inputs_sha256"] != sha(inputs_path) or report["implementation"] != binding() or report["teacher_states_loaded"] or report["original_transcripts_loaded"]:
        raise ValueError("complete autonomous image-only final assessment required")
    outputs = read(after / "outputs.json")
    if [r["id"] for r in outputs] != [r["id"] for r in inputs]:
        raise ValueError("all 160 autonomous responses in fixed order required")
    for row, out in zip(inputs, outputs):
        if any(row[k] != out[k] for k in ("input_sha256", "seed", "readout_seed", "cohort")) or sha(inputs_path.parent / row["input"]) != row["input_sha256"]:
            raise ValueError("assessment source/seed changed")
    refs_path = Path("artifacts/native_state_alignment/prepared_v1/copy_references.json")
    if sha(refs_path) != source_preparation["copy_references_sha256"]:
        raise ValueError("copying references changed")
    refs = read(refs_path)
    copies = [r for r in outputs if r["cohort"] == "copy"]
    if [r["id"] for r in copies] != [r["id"] for r in refs]:
        raise ValueError("copy reference order changed")
    copying = [compare(r, ref) for r, ref in zip(copies, refs)]
    result = dict(copy=dict(summary=totals(copying), cases=copying,
        languages={lang: totals([r for r in copying if r["language"] == lang]) for lang in ("en", "zh")}))
    for cohort, reference, export, scorer in (
        ("zh", "artifacts/native_chinese_interface/prepared_v1/references.json", "native_chinese_interface_result", score_controls),
        ("en", "artifacts/native_english_transfer/prepared_v2/references.json", "native_english_transfer_result", score_english)):
        if sha(reference) != read(Path("references/evidence") / export / "result.json")["files_sha256"]["references.json"]:
            raise ValueError("fixed annotated QA alternatives changed")
        result[cohort] = scorer([r for r in inputs if r["cohort"] == cohort], read(reference), [r for r in outputs if r["cohort"] == cohort])
    before = source["arms"]["legacy"]
    gates = dict(copying=result["copy"]["summary"]["exact_stopped"] >= 30,
        chinese_answers=result["zh"]["normal"]["exact_stopped"] >= 16,
        chinese_pairs=result["zh"]["both_questions_exact"] >= 4,
        chinese_gain=result["zh"]["normal"]["exact_stopped"] >= before["zh"]["normal"]["exact_stopped"] + 4,
        english_answers=result["en"]["summary"]["normal_exact_stopped"] >= 11,
        english_pairs=result["en"]["summary"]["both_correct_pairs"] >= 3)
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[name] = sha(target)
        if files[name] != sha(path):
            raise ValueError("public evidence copy differs")

    for folder, prefix, names in ((learning, "reports/training_", ("report.json", "schedule.json")),
        (args.root / "preflight_v1", "reports/preflight_", ("report.json", "updates.jsonl")),
        (after, "reports/after_", ("report.json", "outputs.json")),
        (args.root / "cache_v1", "inputs/", ("manifest.json",))):
        for name in names:
            retain(folder / name, prefix + name)
    compact = args.output / "reports/training_updates_compact.jsonl"
    with compact.open("w") as f:
        for row in journal:
            g = row["gradients"]
            f.write(json.dumps(dict({k: v for k, v in row.items() if k != "gradients"}, gradient_summary=dict(total=len(g),
                finite=sum(v["finite"] for v in g.values()), nonzero=sum(v["nonzero"] for v in g.values())))) + "\n")
    files[str(compact.relative_to(args.output))] = sha(compact)
    save_json(args.output / "result.json", dict(before=before, after=result, gates=gates, reader_prerequisites_passed=all(gates.values()),
        image_writing_stage_admitted=all(gates.values()), training=training, actual_updates=BUDGET, new_assessment_calls=160,
        source_calibration_result_sha256=sha(prior_root / "result.json"), files_sha256=files, scorer_sha256=sha(__file__),
        training_raw_journal_sha256=sha(learning / "updates.jsonl"), public_optimizer_journal_is_compact=True,
        all_training_ids_seen_four_times=True, auxiliary_unicode_head=True, image_answer_generation_demonstrated=False,
        general_ilm_goal_complete=False, copyright=source["copyright"]))
    print(json.dumps(dict(gates=gates, copy=result["copy"]["summary"], zh=result["zh"]["normal"], en=result["en"]["summary"]), indent=2))


if __name__ == "__main__":
    main()

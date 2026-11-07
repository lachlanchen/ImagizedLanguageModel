#!/usr/bin/env python3
"""Retain completed data/cache/optimizer controls without claiming QA success."""
import argparse
import json
from pathlib import Path
import shutil

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import compare, totals
from train_native_state_alignment import binding


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_state_alignment"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cache = read(args.root / "cache_v1/manifest.json")
    prepared = read(args.root / "prepared_v1/preparation.json")
    preflight = read(args.root / "preflight_v1/report.json")
    journal = [json.loads(s) for s in (args.root / "preflight_v1/updates.jsonl").read_text().splitlines()]
    if cache["implementation"] != binding() or preflight["metadata"]["implementation"] != binding() or cache["training_sha256"] != prepared["training_sha256"]:
        raise ValueError("completed control data/code bindings differ")
    if len(cache["records"]) != 640 or len(journal) != 3 or [r["update"] for r in journal] != [1, 2, 3]:
        raise ValueError("complete data/preflight observation required")
    for row in cache["records"]:
        if sha(args.root / "cache_v1" / row["sample"]) != row["sample_sha256"]:
            raise ValueError("cached numeric source state changed")
    if len(journal[-1]["gradients"]) != 317 or not all(r["finite"] and r["nonzero"] for r in journal[-1]["gradients"].values()):
        raise ValueError("third-update gradient coverage incomplete")
    if preflight["disabled_source_prefix_error"] != 0 or preflight["causal_history_control"]["normalized_max_error"] > 2e-5:
        raise ValueError("source/causal control failed")
    before = read(args.root / "copy_before_v1/outputs.json")
    refs = read(args.root / "prepared_v1/copy_references.json")
    if len(before) != 32 or [r["id"] for r in before] != [r["id"] for r in refs]:
        raise ValueError("complete fixed copying baseline required")
    copies = [compare(o, r) for o, r in zip(before, refs)]
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    sources = (("prepared_v1/preparation.json", "training_preparation.json"),
        ("prepared_v1/english_selected.json", "english_selected.json"),
        ("cache_v1/manifest.json", "cache_manifest.json"),
        ("assessment_v1/preparation.json", "assessment_preparation.json"),
        ("assessment_v1/inputs.json", "assessment_inputs.json"),
        ("preflight_v1/report.json", "preflight_report.json"),
        ("preflight_v1/updates.jsonl", "preflight_updates.jsonl"),
        ("copy_before_v1/report.json", "copy_before_report.json"),
        ("copy_before_v1/outputs.json", "copy_before_outputs.json"))
    for source, name in sources:
        shutil.copy2(args.root / source, args.output / name)
        if sha(args.root / source) != sha(args.output / name):
            raise ValueError("execution evidence copy changed")
        files[name] = sha(args.output / name)
    save_json(args.output / "result.json", dict(files_sha256=files, source_copy_baseline=dict(summary=totals(copies), cases=copies),
        cache_examples_verified=640, cache_elapsed_seconds=cache["elapsed_seconds"],
        real_preflight_updates=3, finite_nonzero_gradients_on_third_update=317,
        trainable_parameters=preflight["trainable_parameters"], complete_model_parameters=preflight["complete_model_parameters"],
        causal_history_control=preflight["causal_history_control"], disabled_source_prefix_error=0,
        initially_zero_tensors_changed=preflight["changed_initially_zero_tensors"],
        first_four_native_readouts_replayed=True, four_case_corrected_path_replay_pending=True,
        preflight_scope_deviation_recorded=True, completed_language_learning_result=False,
        autonomous_image_answers_demonstrated=False, general_ilm_goal_complete=False,
        exporter_sha256=sha(__file__), copyright=prepared["copyright"]))
    print(json.dumps(dict(copy_source_baseline=totals(copies), preflight_updates=3,
        third_update_gradient_coverage=317, additional_source_replay_pending=True), indent=2))


if __name__ == "__main__":
    main()

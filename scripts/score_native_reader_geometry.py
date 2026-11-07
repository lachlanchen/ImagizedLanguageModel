#!/usr/bin/env python3
"""Retain all native-resolution readouts, matched controls and codec checks."""
import argparse
import json
from pathlib import Path
import shutil

from ilm.image_edit.baseline import save_json, sha
from ilm.native_unified.showo2_source import MODEL_SHA256
from score_native_english_transfer import score, INITIALIZER


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_english_transfer"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original = args.root / "prepared_v2"
    prepared = args.root / "native432_prepared_v1"
    preparation = read(prepared / "preparation.json")
    original_preparation = read(original / "preparation.json")
    if preparation["source_inputs_sha256"] != sha(original / "inputs.json") or preparation["references_sha256"] != sha(original / "references.json"):
        raise ValueError("changed original English cohort")
    if sha(prepared / "inputs.json") != preparation["inputs_sha256"]:
        raise ValueError("changed resized cohort")
    for name, digest in preparation["files_sha256"].items():
        if sha(prepared / name) != digest:
            raise ValueError("changed resized image")
    if sha(prepared / "recognition.json") != preparation["recognition_sha256"]:
        raise ValueError("changed source readability calibration")
    original_inputs = read(original / "inputs.json")
    inputs = read(prepared / "inputs.json")
    refs = read(original / "references.json")
    for old, new in zip(original_inputs, inputs):
        if new["original_input_sha256"] != old["input_sha256"]:
            raise ValueError("source image pairing differs")
        for key in ("id", "source_id", "pair", "condition", "seed", "readout_seed"):
            if (key in new) != (key in old) or new.get(key) != old.get(key):
                raise ValueError("source condition or matched seed differs")
    specs = (("source_1024", "unadapted_hq_v1", original_inputs),
             ("source_432", "unadapted_hq_native432_v1", inputs),
             ("predecessor_432", "predecessor_native432_v1", inputs),
             ("dense_432", "dense_native432_v1", inputs))
    models = {}
    files = {}
    args.output.mkdir(parents=True, exist_ok=False)

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != sha(target):
            raise ValueError("evidence copy changed")
        files[name] = sha(target)

    for key, folder, rows in specs:
        path = args.root / folder
        report = read(path / "report.json")
        if report["cases"] != 80 or report["cap"] != 96 or report["teacher_answers_loaded"]:
            raise ValueError("incomplete autonomous condition")
        if report["inputs_sha256"] != sha((original if key == "source_1024" else prepared) / "inputs.json"):
            raise ValueError("model input binding differs")
        if key.startswith("source"):
            if report["source_model_sha256"] != MODEL_SHA256 or report["adapters_added"] or report["optimizer_updates"]:
                raise ValueError("frozen original source required")
            if report["implementation_sha256"] != sha("scripts/probe_native_unadapted_reader.py"):
                raise ValueError("unadapted implementation changed")
        else:
            checkpoint = INITIALIZER if key == "predecessor_432" else read("artifacts/native_visual_alignment/learning_v1/report.json")["checkpoint_sha256"]
            if report["checkpoint_sha256"] != checkpoint:
                raise ValueError("adapter checkpoint differs")
        if read(path / "status.json")["state"] != "completed":
            raise ValueError("wait for terminal inference")
        models[key] = score(rows, refs, read(path / "outputs.json"))
        models[key]["normal_stopped"] = sum(r["normal"]["stopped"] for r in models[key]["cases"])
        for name in ("outputs.json", "report.json"):
            retain(path / name, "reports/" + key + "_" + name)
    for name in (*preparation["files_sha256"], "inputs.json", "preparation.json", "recognition.json"):
        retain(prepared / name, name)
    codec = args.root / "native432_codec_v1"
    report = read(codec / "report.json")
    if report["inputs_sha256"] != preparation["inputs_sha256"] or report["codec_ocr"]["cases"] != 32 or sha(codec / "recognition.json") != report["records_sha256"]:
        raise ValueError("incomplete codec readability calibration")
    for row in read(codec / "recognition.json"):
        if sha(codec / row["image"]) != row["source_sha256"]:
            raise ValueError("codec reconstruction changed")
        retain(codec / row["image"], "codec/" + row["image"])
    for name in ("report.json", "recognition.json"):
        retain(codec / name, "codec/" + name)
    for name in ("probe_native_unadapted_reader.py", "check_native_reader_geometry.py", "score_native_reader_geometry.py"):
        retain(Path("scripts") / name, "implementation/" + name)
    save_json(args.output / "result.json", dict(models=models, files_sha256=files,
        source_readability=preparation["source_ocr"], codec_readability=report["codec_ocr"],
        original_input_evidence="../native_english_transfer_result/result.json",
        original_input_result_sha256=sha("references/evidence/native_english_transfer_result/result.json"),
        original_references_sha256=sha(original / "references.json"), source_model_sha256=MODEL_SHA256,
        implementation_sha256=sha(__file__), calls_per_model=80, model_calls=320,
        same_cases_and_matched_seeds=True, full_square_pixels_resized=True, content_cropped=False,
        cases_filtered_using_ocr=False, optimizer_updates_in_diagnostic=0,
        official_benchmark_replication=False, unique_failure_cause_proved=False,
        auxiliary_unicode_head=True, image_answer_generation_demonstrated=False,
        language_quality_qualified=False, general_ilm_goal_complete=False,
        copyright=original_preparation["copyright"]))
    print(json.dumps({key: value["summary"] for key, value in models.items()}, indent=2))


if __name__ == "__main__":
    main()

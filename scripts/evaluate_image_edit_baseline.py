#!/usr/bin/env python3
"""Offline OCR observation, calibrated before scoring the fixed image baseline.

This evaluator is not imported by generation. Retain the raw images and OCR:
reference calibration is not a guarantee on arbitrary generated typography.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unicodedata

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.data import fixed_cases
from scripts.image_edit_language_baseline import prepared_pages


def normalize(text, language):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()) if language == "en" else "".join(text.split())


def engine():
    binary = shutil.which("tesseract")
    if binary is None:
        raise RuntimeError("reuse an installed Tesseract with chi_sim and eng; do not silently change evaluator")
    version = subprocess.check_output([binary, "--version"], text=True, stderr=subprocess.STDOUT)
    langs = subprocess.check_output([binary, "--list-langs"], text=True, stderr=subprocess.STDOUT)
    if not all(name in langs.splitlines() for name in ("chi_sim", "eng")):
        raise RuntimeError("the fixed offline recognition languages are absent")
    models = Path(langs.split('"')[1])
    return binary, dict(binary_sha256=sha(binary), version=version, languages=["chi_sim", "eng"],
                        psm=6, traineddata_sha256={name: sha(models / (name + ".traineddata")) for name in ("chi_sim", "eng")})


def recognize(binary, path):
    result = subprocess.run([binary, str(path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                            capture_output=True, text=True, timeout=60, check=True)
    return result.stdout, result.stderr


def calibrate(data, out):
    prepared = prepared_pages(data)
    cases = fixed_cases()
    binary, source = engine()
    rows = []
    for case, page in zip(cases, prepared["cases"]):
        if case["id"] != page["id"] or hashlib.sha256(case["reference"].encode()).hexdigest() != page["reference_geometry"]["text_sha256"]:
            raise ValueError("fixed reference text and prepared image binding differ")
        name = page["files"]["reference"]
        if sha(data / name) != prepared["files_sha256"][name]:
            raise ValueError("reference PNG changed")
        raw, diagnostic = recognize(binary, data / name)
        rows.append(dict(position=case["position"], id=case["id"], language=case["language"],
                         source_png_sha256=sha(data / name), expected=case["reference"],
                         raw_transcript=raw, ocr_stderr=diagnostic,
                         normalized=normalize(raw, case["language"]),
                         exact=normalize(raw, case["language"]) == normalize(case["reference"], case["language"])))
    return dict(engine=source, prepared_pages_sha256=sha(data / "pages.json"), rows=rows,
                exact=sum(row["exact"] for row in rows), cases=20,
                normalization="NFC; English whitespace collapsed, Chinese/numeric whitespace removed; punctuation and case preserved",
                generated_images_opened=False, generator_model_loaded=False,
                scope="Reference-page OCR calibration, not language or arbitrary-font qualification")


def score(data, generations, calibration):
    prepared = prepared_pages(data)
    binding = json.loads(calibration.read_text())
    if binding["stage"] != "calibrate" or binding["result"]["prepared_pages_sha256"] != sha(data / "pages.json"):
        raise ValueError("owning reference calibration required")
    generated = json.loads((generations / "report.json").read_text())
    source = generated["result"]
    if (generated["stage"] != "cohort" or source["complete_generations"] != 40
            or source["prepared_pages_sha256"] != sha(data / "pages.json")
            or source["reference_images_used_by_generator"]):
        raise ValueError("the full autonomous 40-output cohort must finish first")
    binary, current = engine()
    if current != binding["result"]["engine"]:
        raise ValueError("offline recognition engine changed after reference calibration")
    if binding["evaluator_source_sha256"] != sha(__file__):
        raise ValueError("evaluator changed after reference calibration")
    rows = []
    for record in source["generations"]:
        i, route = record["position"], record["route"]
        directory = generations / f"{i:02}_{route}"
        if sha(directory / "generation.json") != record["result_sha256"]:
            raise ValueError("owning generation receipt changed")
        actual = json.loads((directory / "generation.json").read_text())
        if (actual["reference_answer_used"] or actual["per_question_text_condition"] or actual["output_repair"]
                or sha(directory / "answer.png") != actual["answer_png_sha256"]
                or sha(directory / "answer_rgb.pt") != actual["raw_rgb_sha256"]
                or actual["fixed_condition_sha256"] != source["fixed_condition_sha256"]):
            raise ValueError("autonomous image or fixed-condition boundary differs")
        case = binding["result"]["rows"][i]
        raw, diagnostic = recognize(binary, directory / "answer.png")
        normalized = normalize(raw, case["language"])
        rows.append(dict(position=i, id=case["id"], route=route, raw_transcript=raw,
                         normalized=normalized, ocr_stderr=diagnostic,
                         answer_png_sha256=actual["answer_png_sha256"], expected=case["expected"],
                         ocr_exact=normalized == normalize(case["expected"], case["language"])))
    if {(r["position"], r["route"]) for r in rows} != {(i, route) for i in range(20) for route in ("normal", "blank")}:
        raise ValueError("complete normal/blank case coverage required")
    summaries = {}
    for group, indices in (("copy", range(12)), ("qa", range(12, 20)), ("all", range(20))):
        summaries[group] = {route: dict(cases=len(indices), ocr_exact=sum(r["ocr_exact"] for r in rows
                              if r["position"] in indices and r["route"] == route)) for route in ("normal", "blank")}
    return dict(rows=rows, summaries=summaries, engine=current,
                reference_calibration_exact=binding["result"]["exact"],
                reference_calibration_sha256=sha(calibration), generation_report_sha256=sha(generations / "report.json"),
                raw_images_require_visual_review=True, generator_model_loaded=False,
                evaluation_only=True, language_quality_claim=False,
                scope="Exploratory OCR exact match; prior external-pretraining overlap and generated-font recognition remain limitations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "score"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("hide CUDA explicitly for the separate offline evaluator")
    if args.output.exists():
        raise FileExistsError("preserve prior or partial observations")
    args.output.mkdir(parents=True)
    if args.stage == "score" and (args.generations is None or args.calibration is None):
        parser.error("score requires --generations and --calibration")
    result = calibrate(args.data, args.output) if args.stage == "calibrate" else score(args.data, args.generations, args.calibration)
    save_json(args.output / "report.json", dict(stage=args.stage, result=result,
              evaluator_source_sha256=sha(__file__), local_optimizer_updates=0,
              language_quality_claim=False, general_ilm_goal_complete=False))
    print(json.dumps({k: result[k] for k in ("exact", "cases") if k in result} or result.get("summaries"), indent=2))

#!/usr/bin/env python3
"""Independently score every V68 raw answer and verify its source bindings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_language_audit import AUDIT_DATA_SHA, CodecLanguageAuditData
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_close(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("verification field identity differs")
        for key in a:
            assert_close(a[key], b[key])
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("verification array length differs")
        for left, right in zip(a, b):
            assert_close(left, right)
    elif isinstance(a, float):
        if not np.isclose(a, b, rtol=0, atol=1e-12):
            raise ValueError("independent measurement differs")
    elif a != b:
        raise ValueError("independent value differs")


def physical_scores(prediction, references, stopped):
    candidates = []
    for value in references:
        target = value.numpy() == 0
        width = min(target.shape[-1], prediction.shape[-1])
        common = int(np.logical_and(target[:, :width], prediction[:, :width]).sum())
        total = int(target.sum()) + int(prediction.sum())
        candidates.append(dict(foreground_f1=2 * common / total if total else 1.,
            exact=bool(stopped and target.shape == prediction.shape and np.array_equal(target, prediction))))
    return dict(foreground_f1=max(x["foreground_f1"] for x in candidates),
        exact=any(x["exact"] for x in candidates), blank=not bool(prediction.any()),
        blocks=prediction.shape[-1] // 64, stopped=bool(stopped), references=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only score verification required")
    torch.set_num_threads(4)
    audit = args.run / "development"
    report = json.loads((audit / "report.json").read_text())
    cases = json.loads((audit / "cases.json").read_text())
    identity = json.loads((audit / "identity.json").read_text())
    if identity != report["identity"] or sha(audit / "cases.json") != report["cases_sha256"]:
        raise ValueError("final audit identity changed")
    if identity["prepared_data_sha256"] != AUDIT_DATA_SHA or identity["maximum_blocks"] != 128:
        raise ValueError("registered data or fixed generation capacity changed")
    if sha(args.run / "training_report.json") != identity["training_report_sha256"]:
        raise ValueError("completed training report changed")
    training = json.loads((args.run / "training_report.json").read_text())
    if (training["update"] != 10000 or training["development_examples"] != 0
            or training["final_trainable_sha256"] != identity["checkpoint_sha256"]
            or sha(args.run / "actor_trainable_final.pt") != identity["checkpoint_sha256"]):
        raise ValueError("only the fixed complete training state may be evaluated")
    for name, digest in training["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("accepted training implementation changed")
    for name, digest in training["prerequisites"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("accepted training prerequisite changed")
    for name, digest in identity["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("frozen evaluator implementation changed")
        original = subprocess.check_output(["git", "show", report["source_commit"] + ":" + name], cwd=ROOT)
        if hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("evaluator Git provenance differs")
    data = CodecLanguageAuditData(ROOT / "data/general_instructions/codec_language_audit_v68_20261105")
    cache = CompleteWritingImageData(ROOT / "data/general_instructions/complete_lines_development_20261105",
        purpose="development", expected_report_sha256=data.report["development_report_sha256"])
    if len(cases) != 192:
        raise ValueError("all fixed question scores required")
    scores, files = {}, {}
    for index, row in enumerate(data.cases):
        key = row["source"] + "/" + row["id"]
        references = data.references(index)
        original, boundary = cache.get(row["original_cache_file"])
        expected = ((original[0, 0, :, boundary * 64:-128] + 1) * 127.5).to(torch.uint8)
        if not torch.equal(references[0], expected):
            raise ValueError("original first-reference pixels changed")
        scores[key] = {}
        for route in ("normal", "blank"):
            relative = Path("predictions") / route / (row["source"] + "_" + row["id"] + ".pt")
            path = audit / relative
            saved = torch.load(path, map_location="cpu", weights_only=True)
            prompt = data.prompt(index, blank=route == "blank")
            header = dict(id=row["id"], source=row["source"], route=route,
                prompt_pixels_sha256=hashlib.sha256(prompt.numpy().tobytes()).hexdigest(),
                checkpoint_sha256=identity["checkpoint_sha256"])
            if saved["header"] != header:
                raise ValueError("raw generation bound to another input or checkpoint")
            result = saved["result"]
            count = len(result["images"])
            if (count > 128 or result["stopped"] == result["capped"]
                    or (result["capped"] and count != 128)
                    or len(result["events"]) != count + int(result["stopped"])):
                raise ValueError("invalid actual stream or termination")
            for block, event in enumerate(result["events"]):
                stopping = result["stopped"] and block == count
                probability = event["stop_probability"]
                if (event["block"] != block or event["stopped"] != stopping
                        or not np.isfinite(probability) or not 0 <= probability <= 1
                        or stopping != (probability >= .5)):
                    raise ValueError("raw learned stopping decision differs")
            images = []
            for value in result["images"]:
                if value.dtype != torch.float32 or value.shape != (1, 1, 32, 64):
                    raise ValueError("raw generated block shape/type differs")
                array = value[0, 0].numpy()
                if not np.isin(array, [-1., 1.]).all():
                    raise ValueError("raw generated pixels were modified")
                images.append(array < 0)
            prediction = np.concatenate(images, -1) if images else np.empty((32, 0), bool)
            measured = physical_scores(prediction, references, result["stopped"])
            assert_close(measured, cases[key][route])
            scores[key][route] = measured
            files[str(relative)] = sha(path)
            image_path = path.with_suffix(".png")
            if count:
                with Image.open(image_path) as picture:
                    value = np.asarray(picture.convert("L"))
                if not np.isin(value, [0, 255]).all() or not np.array_equal(value == 0, prediction):
                    raise ValueError("published raw PNG differs from the original stream")
                files[str(image_path.relative_to(audit))] = sha(image_path)
            elif image_path.exists():
                raise ValueError("an empty answer cannot have a synthetic PNG placeholder")
    if files != report["raw_files_sha256"]:
        raise ValueError("raw generation inventory or bytes changed")
    means = {}
    for source in ("cmrc", "dolly"):
        values = [scores[r["source"] + "/" + r["id"]] for r in data.cases if r["source"] == source]
        means[source] = {route:{metric:float(np.mean([x[route][metric] for x in values]))
            for metric in ("foreground_f1", "exact", "blank", "blocks", "stopped")}
            for route in ("normal", "blank")}
    assert_close(means, report["measurements"])
    categories = {}
    for category in sorted({r["category"] for r in data.cases if r["source"] == "dolly"}):
        values = [scores["dolly/" + r["id"]]["normal"] for r in data.cases
            if r["source"] == "dolly" and r["category"] == category]
        categories[category] = {metric:float(np.mean([x[metric] for x in values]))
            for metric in ("foreground_f1", "exact", "blank", "stopped")}
    assert_close(categories, report["english_categories"])
    pairs = [dict(ids=pair, both_correct=all(scores["cmrc/" + i]["normal"]["exact"] for i in pair))
        for pair in data.report["chosen_pairs"]]
    assert_close(pairs, report["pairs"])
    both = float(np.mean([x["both_correct"] for x in pairs]))
    assert_close(both, report["paired_both_correct"])
    articles = sorted({r["article"] for r in data.cases if r["source"] == "cmrc"})
    cluster = np.array([np.mean([scores["cmrc/" + r["id"]]["normal"]["foreground_f1"]
        - scores["cmrc/" + r["id"]]["blank"]["foreground_f1"]
        for r in data.cases if r["source"] == "cmrc" and r["article"] == article]) for article in articles])
    bootstrap = np.random.default_rng(20266831).choice(cluster, size=(2000, 64), replace=True).mean(1)
    effect = dict(mean=float(cluster.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
        articles=64, replicates=2000, seed=20266831)
    assert_close(effect, report["question_effect"])
    normal = means["cmrc"]["normal"]
    gates = dict(chinese_exact=normal["exact"] >= .5, chinese_foreground_f1=normal["foreground_f1"] >= .8,
        both_correct_pairs=both >= .35, question_gain=effect["mean"] >= .1,
        question_gain_ci_positive=effect["ci95"][0] > 0, learned_stopping=normal["stopped"] >= .9)
    if gates != report["gates"] or all(gates.values()) != report["bounded_chinese_development_pass"]:
        raise ValueError("registered development decision changed")
    if report["general_language_qualified"] or report["general_ilm_goal_complete"]:
        raise ValueError("bounded development cannot close the general image-language goal")
    result = dict(report_sha256=sha(audit / "report.json"), cases_sha256=sha(audit / "cases.json"),
        verifier_sha256=sha(Path(__file__)), prepared_data_sha256=AUDIT_DATA_SHA,
        original_first_references_verified=192, raw_generations_verified=384,
        raw_files_sha256=files, independently_recomputed_measurements=means,
        independently_recomputed_question_effect=effect, paired_both_correct=both,
        gates=gates, bounded_chinese_development_pass=all(gates.values()),
        all_verification_checks_pass=True, general_language_qualified=False, general_ilm_goal_complete=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("completed independent receipt is immutable")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "raw_files_sha256"}, indent=2))


if __name__ == "__main__":
    main()

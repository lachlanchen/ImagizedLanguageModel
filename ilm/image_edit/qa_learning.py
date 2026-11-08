"""Fixed data and schedules for the first complete native QA learning run."""
import hashlib
import json
from pathlib import Path
import random
import subprocess

from .baseline import ROOT, sha

SNAPSHOT = ROOT / "artifacts/external/image_edit_learning/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d"
CONDITION = ROOT / "artifacts/image_edit_baseline/condition/condition.pt"
TEACHER = ROOT / "artifacts/external/writing_reader/hub/models--stepfun-ai--GOT-OCR-2.0-hf/snapshots/d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
INITIAL = ROOT / "artifacts/ink_mass_comparison_v1/ink_mass_training/checkpoint.pt"
INITIAL_SHA = "166409db4f87af87393576195251527f1fcf54d5fe12812d75f6d1491674717d"
PAGES = ROOT / "artifacts/image_edit_learning/pages_v2"
CONTROLS = ROOT / "artifacts/paired_language_controls_v1"
SOURCE = ROOT / "data/visual_qa/cmrc2018_short/train.jsonl"
COEFFICIENT = .4035679875746343
FILES = ("ilm/image_edit/qa_learning.py", "scripts/train_native_paired_qa.py",
    "scripts/run_native_paired_qa.py", "scripts/assess_native_paired_qa.py",
    "references/native_paired_qa_learning_protocol.md", "ilm/image_edit/baseline.py",
    "ilm/image_edit/paired.py", "ilm/image_edit/autonomous.py", "ilm/image_edit/recognition.py",
    "ilm/image_edit/ink_mass.py", "scripts/infer_paired_image_cohort_v2.py",
    "scripts/score_paired_writing.py", "scripts/evaluate_image_edit_baseline.py")


def identity():
    if subprocess.check_output(["git", "status", "--porcelain", "--", *FILES], cwd=ROOT, text=True):
        raise ValueError("commit the exact learning implementation and protocol first")
    return {p: sha(ROOT / p) for p in FILES}


def dataset():
    from PIL import Image
    rows = [json.loads(x) for x in (PAGES / "pairs.jsonl").read_text().splitlines()]
    chosen = [r for r in rows if r["task"] == "qa" and r["split"] == "train"][:32]
    held = [r for r in rows if r["task"] == "qa" and r["split"] == "validation"]
    source = {r["id"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    if len(chosen) != 32 or len(held) != 32 or {r["article"] for r in chosen} & {r["article"] for r in held}:
        raise ValueError("complete article-disjoint 32/32 cohort required")
    for subset in (chosen, held):
        pairs = {}
        for r in subset:
            pairs.setdefault(r["pair"], []).append(r)
            if r["reference_text"] not in [a["text"] for a in source[r["id"]]["references"]]:
                raise ValueError("canonical target not accepted by original source")
            for key, size in (("input", (1024, 1024)), ("target", (512, 256))):
                path = PAGES / r[key]
                if sha(path) != r[key + "_sha256"] or Image.open(path).size != size:
                    raise ValueError("actual complete image bytes or geometry differ")
        if len(pairs) != 16 or any(len(v) != 2 for v in pairs.values()):
            raise ValueError("all sixteen complete question pairs required")
        for pair in pairs.values():
            answers = [{a["text"] for a in source[r["id"]]["references"]} for r in pair]
            if answers[0] & answers[1]:
                raise ValueError("paired accepted answers must be disjoint")
    return chosen, held


def schedule(rows):
    order, noise = random.Random(20268902), random.Random(20268903)
    result = []
    for epoch in range(4):
        ids = [r["id"] for r in rows]; order.shuffle(ids)
        result.extend([dict(update=len(result) + i + 1, epoch=epoch + 1, id=id,
            noise_seed=noise.randrange(2**31 - 1)) for i, id in enumerate(ids)])
    if [r["update"] for r in result] != list(range(1, 129)):
        raise ValueError("128 consecutive optimizer updates required")
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

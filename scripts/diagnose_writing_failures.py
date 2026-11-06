#!/usr/bin/env python3
"""Score retained writing images by position; never load a model or training data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ink(path):
    with Image.open(path) as image:
        value = np.asarray(image)
    if (value.ndim != 2 or value.shape[0] != 32 or value.shape[1] % 64
            or not np.isin(value, [0, 255]).all()):
        raise ValueError(f"Expected unchanged binary 32-pixel strip: {path}")
    return value == 0


def score(prediction, target):
    width = min(prediction.shape[1], target.shape[1])
    common = int((prediction[:, :width] & target[:, :width]).sum())
    total = int(prediction.sum()) + int(target.sum())
    return dict(f1=2 * common / total if total else 1.,
                exact=bool(np.array_equal(prediction, target)),
                predicted_blocks=prediction.shape[1] // 64,
                reference_blocks=target.shape[1] // 64)


def diagnose(folder):
    receipt = json.loads((folder / "result.json").read_text())
    verification = receipt["verification"]
    if (not verification["all_64_final_outputs_independently_scored"]
            or not receipt["raw_png_bytes_unchanged"] or not receipt["training_only"]
            or receipt["language_quality_claim"] or receipt["general_ilm_goal_complete"]):
        raise ValueError("Require the complete, verified, training-only comparison export")
    files = dict(receipt["raw_png_sha256"])
    files.update({item["path"]: item["sha256"] for item in receipt["all_case_plates"]})
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(folder / relative) != digest:
            raise ValueError(f"Retained evidence identity differs: {name}")
    summaries, cases = {}, []
    for arm in ("control", "metric"):
        summaries[arm] = {}
        original = receipt["arm_reports"][arm]
        for kind, f1_key, exact_key in (
                ("answer", "free_f1", "free_exact"),
                ("assisted", "assisted_f1", "assisted_exact_pixels"),
                ("supplied", "supplied_f1", "supplied_exact_pixels")):
            rows = []
            for index in range(32):
                root = folder / "raw" / f"{arm}_after_{index:02}"
                target = ink(Path(str(root) + "_reference.png"))
                if not np.array_equal(target, ink(folder / "raw" / f"control_after_{index:02}_reference.png")):
                    raise ValueError("Arms have different reference pixels")
                path = Path(str(root) + f"_{kind}.png")
                prediction = ink(path) if path.exists() else np.zeros((32, 0), dtype=bool)
                full = score(prediction, target)
                first = score(prediction[:, :64], target[:, :64])
                later = score(prediction[:, 64:], target[:, 64:]) if target.shape[1] > 64 else None
                expected = original["rows"][index]["free" if kind == "answer" else kind]
                exact_name = "exact" if kind == "answer" else "exact_pixel_stream"
                if (abs(full["f1"] - expected["foreground_f1"]) > 1e-12
                        or full["exact"] != expected[exact_name]):
                    raise ValueError("Independent PNG score differs from the verified original")
                if kind == "answer" and abs(first["f1"] - original["rows"][index]["first_block_f1"]) > 1e-12:
                    raise ValueError("First-block score differs")
                rows.append(dict(case=index, full=full, first=first, later=later))
            later_rows = [row["later"] for row in rows if row["later"] is not None]
            summary = dict(
                cases=len(rows), full_f1=float(np.mean([r["full"]["f1"] for r in rows])),
                full_exact=sum(r["full"]["exact"] for r in rows),
                first_block_f1=float(np.mean([r["first"]["f1"] for r in rows])),
                first_block_exact=sum(r["first"]["exact"] for r in rows),
                later_cases=len(later_rows), later_f1=float(np.mean([r["f1"] for r in later_rows])),
                later_exact=sum(r["exact"] for r in later_rows),
                shorter_than_reference=sum(r["full"]["predicted_blocks"] < r["full"]["reference_blocks"] for r in rows),
                longer_than_reference=sum(r["full"]["predicted_blocks"] > r["full"]["reference_blocks"] for r in rows))
            if (abs(summary["full_f1"] - original["summary"][f1_key]) > 1e-12
                    or summary["full_exact"] != original["summary"][exact_key]):
                raise ValueError("Independent summary differs")
            summaries[arm][kind] = summary
            cases.append(dict(arm=arm, path=kind, rows=rows))
    return dict(
        source_sha256=sha(Path(__file__)), exported_result_sha256=sha(folder / "result.json"),
        verified_png_files=len(files), summaries=summaries, cases=cases,
        scope="Post-hoc spatial diagnosis of the original 32 seen training questions; no new model experiment",
        first_block_definition="First 64 columns (two 32x32 tiles), all 32 cases",
        later_definition="Columns 64 onward, mean per-case F1 over references longer than one block; missing ink counts as error, extra ink is retained",
        full_summary_independently_reproduced=True, model_loaded=False,
        neural_forward_performed=False, new_generation=False, optimizer_updates=0,
        semantic_accuracy_measured=False, unique_learning_cause_identified=False,
        general_ilm_goal_complete=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=Path("references/evidence/decoder_metric_v82_result"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing diagnosis; choose a new output path")
    result = diagnose(args.export.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["summaries"], indent=2))

#!/usr/bin/env python3
"""Bind all held-out image conditions and their exact native coarse views."""
import argparse
import json
import os
from pathlib import Path

from PIL import Image
from ilm.image_edit.baseline import save_json, sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_state_alignment/prepared_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "coarse").mkdir()
    specifications = (("copy", args.prepared, "copy_inputs.json", None),
        ("zh", Path("artifacts/native_chinese_interface/prepared_v1"), "source_1024_inputs.json", Path("artifacts/native_chinese_interface/prepared_v1/source_432_inputs.json")),
        ("en", Path("artifacts/native_english_transfer/prepared_v2"), "inputs.json", Path("artifacts/native_english_transfer/native432_prepared_v1/inputs.json")))
    combined = []
    hashes = {}
    for cohort, root, name, baseline in specifications:
        expected = {r["id"]: r["input_sha256"] for r in read(baseline)} if baseline else {}
        subset = []
        for row in read(root / name):
            path = root / row["input"]
            if sha(path) != row["input_sha256"]:
                raise ValueError("whole held-out image changed")
            target = args.output / f"coarse/{len(combined):03d}.png"
            with Image.open(path) as image:
                if image.size != (1024, 1024):
                    raise ValueError("complete square assessment input required")
                image.convert("RGB").resize((432, 432), Image.Resampling.BICUBIC).save(target)
            if expected and sha(target) != expected[row["id"]]:
                raise ValueError("native baseline low-view pixels changed")
            result = dict(row, cohort=cohort, input=os.path.relpath(path.resolve(), args.output.resolve()),
                coarse_input=str(target.relative_to(args.output)), coarse_input_sha256=sha(target))
            combined.append(result)
            subset.append(result)
            hashes[result["coarse_input"]] = sha(target)
        save_json(args.output / (cohort + "_inputs.json"), subset)
    if len(combined) != 160:
        raise ValueError("all 32 copy, 48 Chinese and 80 English conditions required")
    save_json(args.output / "inputs.json", combined)
    save_json(args.output / "preparation.json", dict(cases=160, inputs_sha256=sha(args.output / "inputs.json"),
        cohort_inputs_sha256={name: sha(args.output / (name + "_inputs.json")) for name in ("copy", "zh", "en")},
        files_sha256=hashes, original_source_pixels_resized=False, native_low_views_derived=True,
        native_zh_en_baseline_views_exact=True, source_input_indexes_contain_answers=False,
        implementation_sha256=sha(__file__), model_calls=0, general_ilm_goal_complete=False))
    print("Bound 160 complete conditions; all 128 native Chinese/English baseline views match exactly.")


if __name__ == "__main__":
    main()
